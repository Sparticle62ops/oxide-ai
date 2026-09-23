use crate::checkpoint::{self, CheckpointFormat};
use crate::dataset::{DatasetManager, Tokenizer, TokenizerKind};
use crate::inference::{InferenceConfig, PSSAInferenceEngine};
use crate::backend::{gemm_cpu_reference, Device};
use crate::pssa::{PSSAConfigV2, PSSALayerV2};
use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct TrainingOptions {
    pub epochs: usize,
    pub latent: usize,
    pub state: usize,
    pub key: usize,
    pub memory: usize,
    pub chunk: usize,
    pub lr: f32,
    pub accumulate: usize,
    pub warmup_steps: usize,
    pub seed: u64,
    /// A global cap across documents; documents are never individually reset to this cap.
    pub max_tokens: Option<usize>,
    pub tokenizer: TokenizerKind,
    pub vocab_size: usize,
    /// Continue training from an existing checkpoint instead of fresh initialization.
    pub resume: Option<String>,
}
impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            epochs: 4,
            latent: 256,
            state: 16,
            key: 32,
            memory: 512,
            chunk: 64,
            lr: 1e-3,
            accumulate: 8,
            warmup_steps: 0,
            seed: 42,
            max_tokens: None,
            tokenizer: TokenizerKind::Bpe,
            vocab_size: 2048,
            resume: None,
        }
    }
}

/// Linear warm-up followed by cosine decay. Update and total are one-based.
pub fn learning_rate_for_update(
    base: f32,
    update: usize,
    total: usize,
    warmup: usize,
) -> Result<f32, String> {
    if !(base.is_finite() && base > 0.0) || total == 0 || update == 0 || update > total {
        return Err("invalid learning-rate schedule inputs".into());
    }
    if warmup >= total && warmup != 0 {
        return Err("warmup-steps must be less than total optimizer updates".into());
    }
    let min = base * 0.01;
    if warmup > 0 && update <= warmup {
        return Ok(base * update as f32 / warmup as f32);
    }
    let progress = if warmup == 0 {
        (update - 1) as f32 / total.saturating_sub(1).max(1) as f32
    } else {
        (update - warmup) as f32 / (total - warmup) as f32
    };
    Ok(min + 0.5 * (base - min) * (1.0 + (std::f32::consts::PI * progress).cos()))
}

struct Parsed {
    flags: HashMap<String, String>,
    positional: Vec<String>,
}
impl Parsed {
    fn parse(args: &[String], allowed: &[&str]) -> Result<Self, String> {
        let allowed: HashSet<&str> = allowed.iter().copied().collect();
        let mut flags = HashMap::new();
        let mut positional = Vec::new();
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            if arg.starts_with('-') {
                if !allowed.contains(arg.as_str()) {
                    return Err(format!("unknown option '{arg}'"));
                }
                if i + 1 >= args.len() || args[i + 1].starts_with('-') {
                    return Err(format!("option '{arg}' requires a value"));
                }
                flags.insert(arg.clone(), args[i + 1].clone());
                i += 2;
            } else {
                positional.push(arg.clone());
                i += 1;
            }
        }
        Ok(Self { flags, positional })
    }
    fn string(&self, long: &str, short: &str) -> Option<&str> {
        self.flags
            .get(long)
            .or_else(|| self.flags.get(short))
            .map(String::as_str)
    }
    fn required_usize(&self, long: &str, short: &str, default: usize) -> Result<usize, String> {
        self.string(long, short).map_or(Ok(default), |x| {
            x.parse()
                .map_err(|_| format!("{long} must be a positive integer"))
        })
    }
    fn usize_nonzero(&self, long: &str, short: &str, default: usize) -> Result<usize, String> {
        let n = self.required_usize(long, short, default)?;
        if n == 0 {
            Err(format!("{long} must be positive"))
        } else {
            Ok(n)
        }
    }
    fn f32(&self, long: &str, short: &str, default: f32) -> Result<f32, String> {
        let n = self.string(long, short).map_or(Ok(default), |x| {
            x.parse().map_err(|_| format!("{long} must be a number"))
        })?;
        if n.is_finite() {
            Ok(n)
        } else {
            Err(format!("{long} must be finite"))
        }
    }
}

trait TrainingObserver {
    fn initialized(
        &mut self,
        model: &PSSALayerV2,
        tokenizer: &Tokenizer,
        options: &TrainingOptions,
    ) -> Result<(), String>;
    fn epoch(
        &mut self,
        model: &PSSALayerV2,
        tokenizer: &Tokenizer,
        options: &TrainingOptions,
        epoch: usize,
        cross_entropy: f64,
        scored_transitions: usize,
        last_lr: f32,
        elapsed_seconds: f32,
    ) -> Result<(), String>;
}

struct NoopTrainingObserver;
impl TrainingObserver for NoopTrainingObserver {
    fn initialized(
        &mut self,
        _model: &PSSALayerV2,
        _tokenizer: &Tokenizer,
        _options: &TrainingOptions,
    ) -> Result<(), String> {
        Ok(())
    }

    fn epoch(
        &mut self,
        _model: &PSSALayerV2,
        _tokenizer: &Tokenizer,
        _options: &TrainingOptions,
        _epoch: usize,
        _cross_entropy: f64,
        _scored_transitions: usize,
        _last_lr: f32,
        _elapsed_seconds: f32,
    ) -> Result<(), String> {
        Ok(())
    }
}

static ARTIFACT_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn atomic_artifact_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or("artifact");
    let nonce = ARTIFACT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = parent.join(format!(".{stem}.{}.{}.tmp", std::process::id(), nonce));
    // Do not remove a file we did not create if a name collision occurs.
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(|e| format!("cannot create temporary artifact '{}': {e}", tmp.display()))?;
    let result = (|| -> io::Result<()> {
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result.map_err(|e| format!("cannot atomically write artifact '{}': {e}", path.display()))
}

struct ArtifactTrainingObserver {
    run_dir: PathBuf,
    data: String,
    output: String,
    metrics: Vec<serde_json::Value>,
}

impl ArtifactTrainingObserver {
    fn new(run_dir: &Path, data: &str, output: &str) -> Self {
        Self {
            run_dir: run_dir.to_path_buf(),
            data: data.to_string(),
            output: output.to_string(),
            metrics: Vec::new(),
        }
    }

    fn checkpoint_path(&self, epoch: usize) -> PathBuf {
        self.run_dir.join(format!("epoch-{epoch:04}.pssa"))
    }

    fn save_checkpoint(&self, model: &PSSALayerV2, epoch: usize) -> Result<(), String> {
        let path = self.checkpoint_path(epoch);
        checkpoint::save_model(model, &path)
            .map_err(|e| format!("cannot save artifact checkpoint '{}': {e}", path.display()))
    }

    fn write_metrics(&self) -> Result<(), String> {
        let path = self.run_dir.join("metrics.json");
        let bytes = serde_json::to_vec_pretty(&self.metrics)
            .map_err(|e| format!("cannot encode artifact metrics '{}': {e}", path.display()))?;
        atomic_artifact_write(&path, &bytes)
    }

    fn add_metric(
        &mut self,
        epoch: usize,
        cross_entropy: Option<f64>,
        scored_transitions: usize,
        updates: usize,
        last_lr: Option<f32>,
        elapsed_seconds: f32,
    ) -> Result<(), String> {
        self.metrics.push(serde_json::json!({
            "epoch": epoch,
            "cross_entropy": cross_entropy,
            "scored_transitions": scored_transitions,
            "optimizer_updates": updates,
            "last_lr": last_lr,
            "elapsed_seconds": elapsed_seconds,
            "checkpoint": self.checkpoint_path(epoch).file_name().and_then(|x| x.to_str()).unwrap_or_default(),
        }));
        self.write_metrics()
    }
}

impl TrainingObserver for ArtifactTrainingObserver {
    fn initialized(
        &mut self,
        model: &PSSALayerV2,
        tokenizer: &Tokenizer,
        options: &TrainingOptions,
    ) -> Result<(), String> {
        let tokenizer_kind = match tokenizer.kind() {
            TokenizerKind::Word => "word",
            TokenizerKind::Bpe => "bpe",
        };
        let run_path = self.run_dir.join("run.json");
        let run = serde_json::json!({
            "schema_version": 1,
            "data_source": &self.data,
            "final_output": &self.output,
            "configuration": {
                "epochs": options.epochs,
                "latent": options.latent,
                "state": options.state,
                "key": options.key,
                "memory": options.memory,
                "chunk": options.chunk,
                "lr": options.lr,
                "accumulate": options.accumulate,
                "warmup_steps": options.warmup_steps,
                "seed": options.seed,
                "max_tokens": options.max_tokens,
                "tokenizer": tokenizer_kind,
                "vocab_size": options.vocab_size,
            },
            "model": {
                "d_vocab": model.cfg.d_vocab,
                "d_latent": model.cfg.d_latent,
                "d_state": model.cfg.d_state,
                "d_mem_key": model.cfg.d_mem_key,
                "mem_capacity": model.cfg.mem_capacity,
                "chunk_len": model.cfg.chunk_len,
            },
            "seed": options.seed,
            "tokenizer": {
                "kind": tokenizer_kind,
                "vocabulary_size": tokenizer.vocab_size,
                "fit_scope": "tokenizer fitted on all supplied training text",
            },
            "scope": {
                "tokenizer": "tokenizer fitted on all supplied training text",
                "max_tokens": "encoded training prefix per epoch, actual transitions lower",
            },
            "state_policy": "recurrent state resets at each epoch and document; episodic memory persists across documents and epochs",
            "memory_policy": "episodic memory is retained in each checkpoint and is not cleared by recurrent-state resets",
            "resume": {
                "supported": false,
                "note": "epoch checkpoints are evidence artifacts; the CLI provides no resume support",
            },
        });
        let bytes = serde_json::to_vec_pretty(&run).map_err(|e| {
            format!(
                "cannot encode run configuration '{}': {e}",
                run_path.display()
            )
        })?;
        atomic_artifact_write(&run_path, &bytes)?;
        self.save_checkpoint(model, 0)?;
        self.add_metric(0, None, 0, model.step_counter, None, 0.0)
    }

    fn epoch(
        &mut self,
        model: &PSSALayerV2,
        _tokenizer: &Tokenizer,
        _options: &TrainingOptions,
        epoch: usize,
        cross_entropy: f64,
        scored_transitions: usize,
        last_lr: f32,
        elapsed_seconds: f32,
    ) -> Result<(), String> {
        self.save_checkpoint(model, epoch)?;
        self.add_metric(
            epoch,
            Some(cross_entropy),
            scored_transitions,
            model.step_counter,
            Some(last_lr),
            elapsed_seconds,
        )
    }
}

pub struct CLIHandler;
impl CLIHandler {
    fn default_data() -> String {
        if std::path::Path::new("data/downloaded.txt").exists() {
            "data/downloaded.txt".into()
        } else {
            "science".into()
        }
    }
    /// Historical name retained for callers; current output is V7.
    pub fn save_model_v2(model: &PSSALayerV2, path: &str) -> io::Result<()> {
        checkpoint::save_model(model, path).map_err(|e| io::Error::other(e.to_string()))
    }
    pub fn load_model_v2(path: &str) -> Result<PSSALayerV2, String> {
        checkpoint::load_checkpoint(path)
            .map(|x| x.model)
            .map_err(|e| e.to_string())
    }

    fn options(parsed: &Parsed) -> Result<TrainingOptions, String> {
        let tokenizer = match parsed.string("--tokenizer", "").unwrap_or("bpe") {
            "bpe" => TokenizerKind::Bpe,
            "word" => TokenizerKind::Word,
            x => return Err(format!("--tokenizer must be bpe or word, got '{x}'")),
        };
        let x = TrainingOptions {
            epochs: parsed.usize_nonzero("--epochs", "-e", 4)?,
            latent: parsed.usize_nonzero("--latent", "", 256)?,
            state: parsed.usize_nonzero("--state", "", 16)?,
            key: parsed.usize_nonzero("--key", "", 32)?,
            memory: parsed.usize_nonzero("--memory", "", 512)?,
            chunk: parsed.usize_nonzero("--chunk", "", 64)?,
            lr: parsed.f32("--lr", "", 1e-3)?,
            accumulate: parsed.usize_nonzero("--accumulate", "", 8)?,
            warmup_steps: parsed.required_usize("--warmup-steps", "", 0)?,
            seed: parsed.string("--seed", "").map_or(Ok(42), |x| {
                x.parse()
                    .map_err(|_| "--seed must be an unsigned integer".to_string())
            })?,
            max_tokens: parsed
                .string("--max-tokens", "")
                .map(|x| {
                    x.parse::<usize>()
                        .map_err(|_| "--max-tokens must be a positive integer".to_string())
                })
                .transpose()?,
            tokenizer,
            vocab_size: parsed.usize_nonzero("--vocab-size", "", 2048)?,
            resume: parsed.string("--resume", "").map(str::to_string),
        };
        if !(x.lr > 0.0) {
            return Err("--lr must be positive".into());
        }
        if x.max_tokens == Some(0) {
            return Err("--max-tokens must be positive".into());
        }
        if x.tokenizer == TokenizerKind::Bpe && x.vocab_size < 257 {
            return Err("--vocab-size must be at least 257 for byte-level BPE".into());
        }
        Ok(x)
    }

    fn documents(
        raw: &str,
        tokenizer: &Tokenizer,
        limit: Option<usize>,
    ) -> Result<Vec<Vec<usize>>, String> {
        let mut docs = Vec::new();
        let mut remaining = limit.unwrap_or(usize::MAX);
        for line in raw.lines() {
            if remaining == 0 {
                break;
            }
            let mut ids = tokenizer.try_encode(line, true)?;
            ids.truncate(remaining);
            remaining -= ids.len();
            if !ids.is_empty() {
                if ids.len() < 2 {
                    return Err(
                        "each nonempty training document must contain at least two tokens".into(),
                    );
                }
                docs.push(ids);
            }
        }
        if docs.is_empty() {
            Err("dataset has no token transitions".into())
        } else {
            Ok(docs)
        }
    }
    fn finite(model: &PSSALayerV2) -> bool { model.all_parameters_finite() }

    pub fn train_corpus(
        raw: &str,
        options: &TrainingOptions,
    ) -> Result<(PSSALayerV2, Tokenizer), String> {
        let mut observer = NoopTrainingObserver;
        Self::train_corpus_with_observer(raw, options, &mut observer)
    }

    fn train_corpus_with_observer(
        raw: &str,
        options: &TrainingOptions,
        observer: &mut dyn TrainingObserver,
    ) -> Result<(PSSALayerV2, Tokenizer), String> {
                let (mut model, tokenizer) = match options.resume.as_deref() {
            Some(path) => {
                let loaded = checkpoint::load_checkpoint(path)
                    .map_err(|e| format!("cannot resume from '{path}': {e}"))?;
                let model = loaded.model;
                if model.vocabulary.is_empty() {
                    return Err("resume checkpoint lacks vocabulary provenance".into());
                }
                let tokenizer = match &model.tokenizer_json {
                    Some(json) => Tokenizer::from_serialized(json)?,
                    None => Tokenizer::from_vocabulary(&model.vocabulary)?,
                };
                if tokenizer.vocab_size != model.cfg.d_vocab
                    || tokenizer.ordered_vocabulary()? != model.vocabulary
                {
                    return Err("resume checkpoint tokenizer/vocabulary mismatch".into());
                }
                println!(
                    "resumed_from={path} vocab={} d_latent={} prior_steps={}",
                    model.cfg.d_vocab, model.cfg.d_latent, model.step_counter
                );
                (model, tokenizer)
            }
            None => {
                let tokenizer = match options.tokenizer {
                    TokenizerKind::Word => Tokenizer::from_corpus(raw, true),
                    TokenizerKind::Bpe => Tokenizer::from_corpus_bpe(raw, options.vocab_size)?,
                };
                let cfg = PSSAConfigV2 {
                    d_vocab: tokenizer.vocab_size,
                    d_latent: options.latent,
                    d_state: options.state,
                    d_mem_key: options.key,
                    mem_capacity: options.memory,
                    chunk_len: options.chunk,
                    lr: options.lr,
                    ..Default::default()
                };
                cfg.validate();
                let mut model = PSSALayerV2::new(cfg, options.seed);
                model.vocabulary = tokenizer.ordered_vocabulary()?;
                model.tokenizer_json = tokenizer.serialized_metadata();
                (model, tokenizer)
            }
        };
        let docs = Self::documents(raw, &tokenizer, options.max_tokens)?;
        let mut plan = Vec::<(usize, usize, usize)>::new();
        for (doc_id, doc) in docs.iter().enumerate() {
            let mut start = 0;
            while start + 1 < doc.len() {
                let len = options.chunk.min(doc.len() - 1 - start);
                plan.push((doc_id, start, len));
                start += len;
            }
        }
        let groups_per_epoch = plan.len().div_ceil(options.accumulate);
        let total_updates = groups_per_epoch
            .checked_mul(options.epochs)
            .ok_or("training update count overflow")?;
        if total_updates == 0 {
            return Err("dataset has no training chunks".into());
        }
        if options.warmup_steps >= total_updates && options.warmup_steps != 0 {
            return Err(format!(
                "--warmup-steps ({}) must be less than total optimizer updates ({total_updates})",
                options.warmup_steps
            ));
        }
        let started = Instant::now();
        observer.initialized(&model, &tokenizer, options)?;
        let mut update = 0;
        let mut last_lr = 0.0f32;
        for epoch in 0..options.epochs {
            model.reset_recurrent_state();
            let mut loss_sum = 0.0f64;
            let mut token_sum = 0usize;
            for group in plan.chunks(options.accumulate) {
                let total_tokens: usize = group.iter().map(|x| x.2).sum();
                if total_tokens == 0 {
                    continue;
                }
                model.zero_gradients();
                for &(doc_id, start, len) in group {
                    if start == 0 {
                        model.reset_recurrent_state();
                    }
                    let input = &docs[doc_id][start..start + len];
                    let target = &docs[doc_id][start + 1..start + 1 + len];
                    let loss = model.forward_train_chunk(input, target);
                    if !loss.is_finite() {
                        return Err("non-finite loss; training aborted; any completed epoch artifacts remain available".into());
                    }
                    model.backward_chunk(len, len as f32 / total_tokens as f32);
                    model.insert_training_memory(loss, len);
                    loss_sum += loss as f64 * len as f64;
                    token_sum += len;
                }
                update += 1;
                last_lr = learning_rate_for_update(
                    options.lr,
                    update,
                    total_updates,
                    options.warmup_steps,
                )?;
                model.apply_adamw(last_lr);
                if !Self::finite(&model) {
                    return Err("non-finite parameters; training aborted; any completed epoch artifacts remain available".into());
                }
            }
            model.ema_consolidate_plasticity();
            let cross_entropy = loss_sum / token_sum.max(1) as f64;
            println!(
                "epoch {}/{} loss={:.6} tokens={} updates={}",
                epoch + 1,
                options.epochs,
                cross_entropy,
                token_sum,
                update
            );
            observer.epoch(
                &model,
                &tokenizer,
                options,
                epoch + 1,
                cross_entropy,
                token_sum,
                last_lr,
                started.elapsed().as_secs_f32(),
            )?;
        }
        println!(
            "training_seconds={:.3} optimizer_updates={}",
            started.elapsed().as_secs_f32(),
            update
        );
        Ok((model, tokenizer))
    }

    pub fn run_training(data: &str, options: &TrainingOptions, out: &str) -> Result<(), String> {
        let raw = DatasetManager::try_load_dataset(Some(data))?;
        let (model, _) = Self::train_corpus(&raw, options)?;
        Self::save_model_v2(&model, out)
            .map_err(|e| format!("cannot save checkpoint '{out}': {e}"))?;
        println!("saved_checkpoint={out}");
        Ok(())
    }

    fn run_training_with_artifacts(
        data: &str,
        options: &TrainingOptions,
        out: &str,
        run_dir: &str,
    ) -> Result<(), String> {
        let run_path = Path::new(run_dir);
        if run_path.exists() {
            return Err(format!(
                "run directory '{}' already exists; --run-dir refuses reuse",
                run_path.display()
            ));
        }
        fs::create_dir(run_path).map_err(|e| {
            format!(
                "cannot create new run directory '{}': {e}",
                run_path.display()
            )
        })?;
        let output = Path::new(out);
        let output_parent = output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let output_name = output.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let reserved = matches!(output_name, "run.json" | "metrics.json")
            || (output_name.starts_with("epoch-") && output_name.ends_with(".pssa"));
        if reserved && output_parent.canonicalize().ok() == run_path.canonicalize().ok() {
            return Err(format!(
                "final output '{out}' conflicts with a reserved run artifact"
            ));
        }
        let raw = DatasetManager::try_load_dataset(Some(data))?;
        let mut observer = ArtifactTrainingObserver::new(run_path, data, out);
        let (model, _) = Self::train_corpus_with_observer(&raw, options, &mut observer)?;
        Self::save_model_v2(&model, out)
            .map_err(|e| format!("cannot save checkpoint '{out}': {e}"))?;
        println!("saved_checkpoint={out}");
        Ok(())
    }

    fn word_tokenizer_for_model(
        model: &PSSALayerV2,
        data: Option<&str>,
        label: &str,
    ) -> Result<Tokenizer, String> {
        if model.vocabulary.is_empty() {
            return Err(format!("{label} checkpoint lacks vocabulary provenance"));
        }
        let tokenizer = Tokenizer::from_vocabulary(&model.vocabulary)?;
        if let Some(source) = data {
            let raw = DatasetManager::try_load_dataset(Some(source))?;
            let external = Tokenizer::from_corpus(&raw, true);
            if external.ordered_vocabulary()? != model.vocabulary {
                return Err(format!(
                    "--data tokenizer/order does not match {label} checkpoint"
                ));
            }
        }
        Ok(tokenizer)
    }
    fn load_for_inference(
        model_path: &str,
        data: Option<&str>,
    ) -> Result<(PSSALayerV2, Tokenizer), String> {
        let loaded = checkpoint::load_checkpoint(model_path)
            .map_err(|e| format!("cannot load model '{model_path}': {e}"))?;
        let model = loaded.model;
        match loaded.format {
            CheckpointFormat::V8 | CheckpointFormat::V7 => match &model.tokenizer_json {
                Some(json) => {
                    if data.is_some() {
                        return Err("--data is legacy word provenance only; V7 BPE checkpoints restore their embedded tokenizer and never retrain it".into());
                    }
                    let tokenizer = Tokenizer::from_serialized(json)?;
                    if tokenizer.ordered_vocabulary()? != model.vocabulary
                        || tokenizer.vocab_size != model.cfg.d_vocab
                    {
                        return Err("V7 tokenizer metadata/model vocabulary mismatch".into());
                    }
                    Ok((model, tokenizer))
                }
                None => {
                    let tokenizer = Self::word_tokenizer_for_model(&model, data, "V7 word")?;
                    Ok((model, tokenizer))
                }
            },
            CheckpointFormat::V6 => {
                let tokenizer = Self::word_tokenizer_for_model(&model, data, "V6")?;
                Ok((model, tokenizer))
            }
            CheckpointFormat::LegacyV5InferenceOnly => {
                let source = data.ok_or("legacy V5 checkpoint requires explicit --data; tokenizer provenance is unavailable")?;
                eprintln!(
                    "warning: legacy V5 checkpoint is inference-only; optimizer and tokenizer provenance are unavailable"
                );
                let raw = DatasetManager::try_load_dataset(Some(source))?;
                let tokenizer = Tokenizer::from_corpus(&raw, true);
                if tokenizer.vocab_size != model.cfg.d_vocab {
                    return Err(format!(
                        "legacy corpus vocabulary size {} does not match checkpoint {}",
                        tokenizer.vocab_size, model.cfg.d_vocab
                    ));
                }
                Ok((model, tokenizer))
            }
        }
    }

    pub fn run_generate(
        prompt: &str,
        model_path: &str,
        data: Option<&str>,
        temperature: f32,
        max_new: usize,
    ) -> Result<String, String> {
        let (mut model, tokenizer) = Self::load_for_inference(model_path, data)?;
        let cfg = InferenceConfig {
            temperature,
            max_new_tokens: max_new,
            top_k: if temperature == 0.0 { 1 } else { 24 },
            repetition_penalty: if temperature == 0.0 { 1.0 } else { 1.25 },
            ..Default::default()
        };
        PSSAInferenceEngine::try_new(&mut model, &tokenizer)?.try_generate_chat_turn(
            prompt,
            &cfg,
            |_| {},
        )
    }

    pub fn evaluate_corpus(
        model: &mut PSSALayerV2,
        tokenizer: &Tokenizer,
        raw: &str,
    ) -> Result<(f64, usize, usize, usize), String> {
        let docs = Self::documents(raw, tokenizer, None)?;
        let mut loss = 0.0f64;
        let mut tokens = 0usize;
        let mut correct = 0usize;
        let mut oov = 0usize;
        for doc in &docs {
            oov += doc.iter().filter(|&&x| x == 0).count();
            model.reset_recurrent_state();
            let mut start = 0;
            while start + 1 < doc.len() {
                let len = model.cfg.chunk_len.min(doc.len() - 1 - start);
                let l = model.forward_train_chunk(
                    &doc[start..start + len],
                    &doc[start + 1..start + 1 + len],
                );
                if !l.is_finite() {
                    return Err("non-finite evaluation loss".into());
                }
                loss += l as f64 * len as f64;
                for t in 0..len {
                    let logits =
                        &model.tape.logits[t * model.cfg.d_vocab..(t + 1) * model.cfg.d_vocab];
                    let guess = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                        .map(|x| x.0)
                        .unwrap_or(0);
                    correct += usize::from(guess == doc[start + 1 + t]);
                }
                tokens += len;
                start += len;
            }
        }
        Ok((loss / tokens.max(1) as f64, tokens, correct, oov))
    }
    fn run_evaluate(model_path: &str, data: &str) -> Result<(), String> {
        let (mut model, tokenizer) = Self::load_for_inference(model_path, None)?;
        let raw = DatasetManager::try_load_dataset(Some(data))?;
        let (ce, tokens, correct, oov) = Self::evaluate_corpus(&mut model, &tokenizer, &raw)?;
        let encoded = docs_token_count(&raw, &tokenizer)?;
        let ppl = ce.exp();
        let (ppl_json, overflow) = if ppl.is_finite() {
            (format!("{ppl:.8}"), false)
        } else {
            ("null".into(), true)
        };
        println!(
            "{{\"cross_entropy\":{ce:.8},\"perplexity\":{ppl_json},\"perplexity_overflow\":{overflow},\"oov_rate\":{:.8},\"token_count\":{},\"next_token_accuracy\":{:.8}}}",
            oov as f64 / encoded.max(1) as f64,
            tokens,
            correct as f64 / tokens.max(1) as f64
        );
        Ok(())
    }
    fn run_chat(model_path: &str, data: Option<&str>, temp: f32) -> Result<(), String> {
        let (mut model, tokenizer) = Self::load_for_inference(model_path, data)?;
        println!("interactive: /exit");
        loop {
            print!("user> ");
            io::stdout().flush().map_err(|e| e.to_string())?;
            let mut line = String::new();
            if io::stdin()
                .read_line(&mut line)
                .map_err(|e| e.to_string())?
                == 0
            {
                break;
            }
            let p = line.trim();
            if p == "/exit" || p == "quit" {
                break;
            }
            if p.is_empty() {
                continue;
            }
            let cfg = InferenceConfig {
                temperature: temp,
                ..Default::default()
            };
            println!(
                "{}",
                PSSAInferenceEngine::try_new(&mut model, &tokenizer)?.try_generate_chat_turn(
                    p,
                    &cfg,
                    |_| {}
                )?
            );
        }
        Ok(())
    }
    pub fn run_benchmark() -> Result<(), String> {
        let raw = "the patient scientist observes the bright moon .\nthe patient scientist observes the bright moon .\nthe patient scientist observes the bright moon .\n";
        let opts = TrainingOptions {
            epochs: 120,
            latent: 16,
            state: 4,
            key: 8,
            memory: 8,
            chunk: 8,
            lr: 0.02,
            accumulate: 1,
            warmup_steps: 4,
            seed: 7,
            max_tokens: None,
            tokenizer: TokenizerKind::Word,
            vocab_size: 2048,
            resume: None,
        };
        let (mut m, tok) = Self::train_corpus(raw, &opts)?;
        let (ce, _, _, _) = Self::evaluate_corpus(&mut m, &tok, raw)?;
        if !(ce < 0.8) {
            return Err(format!(
                "benchmark learned-model fidelity insufficient: CE={ce}"
            ));
        }
        let out = PSSAInferenceEngine::try_new(&mut m, &tok)?.try_generate_chat_turn(
            "the patient scientist",
            &InferenceConfig {
                temperature: 0.0,
                max_new_tokens: 5,
                ..Default::default()
            },
            |_| {},
        )?;
        if !out.contains("observes") {
            return Err(format!(
                "benchmark completion lacks learned reference token: {out}"
            ));
        }
        println!("benchmark_pass ce={ce:.6} completion={out}");
        Ok(())
    }
    pub fn print_help() {
        println!(
            "Usage: oxide <command> [options]\nCommands: train, generate, evaluate, chat, download, benchmark\ntrain [source] [-d|--data source] [-o|--out path] [--run-dir path] [--tokenizer bpe|word --vocab-size 2048] [-e|--epochs n] [--latent n --state n --key n --memory n --chunk n --lr f --accumulate n --warmup-steps n --seed n --max-tokens n] [--resume path]\ngenerate <prompt> [-p|--prompt text] [-m|--model path] [-d|--data source] [-t|--temp f] [--max-new-tokens n]\nevaluate -m|--model path -d|--data source\nDefault training is byte-level BPE (2048 maximum vocabulary). --data is only a legacy-word provenance check for generation; V7 BPE restores embedded tokenizer metadata. --max-tokens caps the encoded training prefix reused per epoch, not tokenizer fitting or the total across epochs. --run-dir must name a new directory and writes epoch checkpoints plus run.json and metrics.json; it does not enable CLI resume; use --resume path to continue optimizer state from a checkpoint."
        );
    }
    pub fn parse_and_execute(args: Vec<String>) -> Result<(), String> {
        if args.len() < 2 {
            Self::print_help();
            return Ok(());
        }
        match args[1].as_str() {
            "help" | "--help" | "-h" => {
                Self::print_help();
                Ok(())
            }
            "train" => {
                let p = Parsed::parse(
                    &args[2..],
                    &[
                        "--data",
                        "-d",
                        "--out",
                        "-o",
                        "--run-dir",
                        "--epochs",
                        "-e",
                        "--latent",
                        "--state",
                        "--key",
                        "--memory",
                        "--chunk",
                        "--lr",
                        "--accumulate",
                        "--warmup-steps",
                        "--seed",
                        "--max-tokens",
                        "--tokenizer",
                        "--vocab-size",
                        "--resume",
                    ],
                )?;
                if p.positional.len() > 1 {
                    return Err("train accepts at most one positional source".into());
                }
                let data = p
                    .string("--data", "-d")
                    .map(str::to_string)
                    .or_else(|| p.positional.first().cloned())
                    .unwrap_or_else(Self::default_data);
                let out = p.string("--out", "-o").unwrap_or("data/model.pssa");
                let options = Self::options(&p)?;
                if let Some(run_dir) = p.string("--run-dir", "") {
                    Self::run_training_with_artifacts(&data, &options, out, run_dir)
                } else {
                    Self::run_training(&data, &options, out)
                }
            }
            "generate" => {
                let p = Parsed::parse(
                    &args[2..],
                    &[
                        "--prompt",
                        "-p",
                        "--model",
                        "-m",
                        "--data",
                        "-d",
                        "--temp",
                        "-t",
                        "--max-new-tokens",
                    ],
                )?;
                if p.positional.len() > 1 {
                    return Err("generate accepts one positional prompt".into());
                }
                let prompt = p
                    .string("--prompt", "-p")
                    .or_else(|| p.positional.first().map(String::as_str))
                    .ok_or("generate requires a prompt")?;
                let temp = p.f32("--temp", "-t", 0.70)?;
                if temp < 0.0 {
                    return Err("--temp must be >= 0".into());
                }
                let max = p.usize_nonzero("--max-new-tokens", "", 64)?;
                println!(
                    "{}",
                    Self::run_generate(
                        prompt,
                        p.string("--model", "-m").unwrap_or("data/model.pssa"),
                        p.string("--data", "-d"),
                        temp,
                        max
                    )?
                );
                Ok(())
            }
            "evaluate" => {
                let p = Parsed::parse(&args[2..], &["--model", "-m", "--data", "-d"])?;
                if !p.positional.is_empty() {
                    return Err("evaluate does not accept positional arguments".into());
                }
                Self::run_evaluate(
                    p.string("--model", "-m").unwrap_or("data/model.pssa"),
                    p.string("--data", "-d").ok_or("evaluate requires --data")?,
                )
            }
            "chat" | "repl" => {
                let p = Parsed::parse(
                    &args[2..],
                    &["--model", "-m", "--data", "-d", "--temp", "-t"],
                )?;
                let t = p.f32("--temp", "-t", 0.70)?;
                if t < 0.0 {
                    return Err("--temp must be >= 0".into());
                }
                Self::run_chat(
                    p.string("--model", "-m").unwrap_or("data/model.pssa"),
                    p.string("--data", "-d"),
                    t,
                )
            }
            "download" => {
                let p = Parsed::parse(&args[2..], &["--out", "-o"])?;
                if p.positional.len() != 1 {
                    return Err("download requires one Hugging Face repository".into());
                }
                let text = DatasetManager::download_huggingface_dataset(&p.positional[0])?;
                std::fs::write(
                    p.string("--out", "-o").unwrap_or("data/downloaded.txt"),
                    text,
                )
                .map_err(|e| e.to_string())
            }
            "gpu-probe" => {
                run_gpu_probe();
                Ok(())
            }
            "benchmark" => {
                if args.len() != 2 {
                    return Err("benchmark takes no options".into());
                }
                Self::run_benchmark()
            }
            _ => Err(format!("unknown command '{}'; run oxide help", args[1])),
        }
    }
}
fn docs_token_count(raw: &str, tokenizer: &Tokenizer) -> Result<usize, String> {
    raw.lines().try_fold(0usize, |n, line| {
        tokenizer.try_encode(line, true).map(|ids| n + ids.len())
    })
}

/// Bring up the WebGPU compute device, run the embedded tiled GEMM kernel on it,
/// and check the result against the CPU reference implementation.
pub fn run_gpu_probe() {
    println!("=== oxide gpu-probe ===");
    let device = match Device::try_gpu() {
        Ok(d) => {
            println!("adapter: WebGPU compute device acquired");
            d
        }
        Err(e) => {
            println!("adapter: unavailable ({})", e);
            println!("result: no GPU on this machine; training stays on CPU");
            return;
        }
    };

    let ctx = match &device {
        Device::Gpu(ctx) => ctx.clone(),
        Device::Cpu => {
            println!("result: CPU device returned; nothing to probe");
            return;
        }
    };

    let (batch, m, n, k) = (2usize, 64usize, 96usize, 128usize);
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / 8_388_608.0) - 1.0
    };
    let x: Vec<f32> = (0..batch * m * k).map(|_| next()).collect();
    let w: Vec<f32> = (0..n * k).map(|_| next()).collect();

    let t_gpu = Instant::now();
    let y_gpu = ctx.dispatch_gemm(&x, &w, m, n, k, batch);
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;

    let t_cpu = Instant::now();
    let y_cpu = gemm_cpu_reference(&x, &w, m, n, k, batch);
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

    let mut max_abs = 0.0f32;
    for (a, b) in y_gpu.iter().zip(y_cpu.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }

    println!("shape: batch={} M={} N={} K={}", batch, m, n, k);
    println!("gpu:   {:.3} ms", gpu_ms);
    println!("cpu:   {:.3} ms", cpu_ms);
    println!("max_abs_diff: {:.3e}", max_abs);
    if max_abs < 1e-3 {
        println!("result: PASS, GPU kernel matches CPU reference");
    } else {
        println!("result: FAIL, GPU kernel diverges from CPU reference");
    }
    println!("note: layer math is still CPU-dispatched; this proves the device path only");
}
