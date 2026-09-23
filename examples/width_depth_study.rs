//! Resumable, provenance-guarded controlled width/depth study runner.
//!
//! Usage: `cargo run --release --example width_depth_study -- MANIFEST.json`.
//! The manifest is deliberately the sole command-line argument: it records the
//! frozen corpus/tokenizer plan rather than accepting mutable CLI training flags.

use oxide_ai_pssa::checkpoint::{self, CheckpointFormat};
use oxide_ai_pssa::cli::{learning_rate_for_update, CLIHandler};
use oxide_ai_pssa::dataset::Tokenizer;
use oxide_ai_pssa::inference::{InferenceConfig, PSSAInferenceEngine};
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use serde_json::{json, Map, Value};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const SCHEMA: u64 = 1;
const RESUME_KEEP: usize = 2;
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
struct Manifest {
    run_dir: PathBuf,
    initial_checkpoint: PathBuf,
    train_path: PathBuf,
    validation_path: PathBuf,
    prompts_path: PathBuf,
    width: usize,
    depth: usize,
    epochs: usize,
    seed: u64,
    chunk: usize,
    accumulate: usize,
    warmup_steps: usize,
    base_lr: f32,
    state: usize,
    key: usize,
    memory: usize,
    checkpoint_every_updates: usize,
    diagnostics_every_updates: usize,
    expected_transitions_per_epoch: usize,
    max_updates_this_invocation: Option<usize>,
    preflight_only: bool,
}

#[derive(Clone)]
struct Chunk {
    document: usize,
    start: usize,
    len: usize,
}

struct Prepared {
    manifest: Manifest,
    tokenizer: Tokenizer,
    docs: Vec<Vec<usize>>,
    plan: Vec<Chunk>,
    groups_per_epoch: usize,
    total_updates: usize,
    cfg: PSSAConfigV2,
    guard: Value,
    manifest_value: Value,
}

#[derive(Clone)]
struct State {
    epoch: usize,
    next_group: usize,
    checkpoint: String,
    global_updates: usize,
    train_loss_sum: f64,
    scored_transitions: usize,
    epoch_loss_sum: f64,
    epoch_transitions: usize,
    compute_seconds: f64,
    epoch_compute_seconds: f64,
    wall_elapsed_seconds: f64,
    epoch_metrics: Vec<Value>,
}

fn err<T>(message: impl Into<String>) -> Result<T, String> { Err(message.into()) }

fn required<'a>(object: &'a Map<String, Value>, key: &str) -> Result<&'a Value, String> {
    object.get(key).ok_or_else(|| format!("manifest missing '{key}'"))
}
fn required_string(object: &Map<String, Value>, key: &str) -> Result<PathBuf, String> {
    let s = required(object, key)?.as_str().ok_or_else(|| format!("'{key}' must be a string"))?;
    if s.is_empty() { return err(format!("'{key}' must not be empty")); }
    Ok(PathBuf::from(s))
}
fn required_usize(object: &Map<String, Value>, key: &str, nonzero: bool) -> Result<usize, String> {
    let n = required(object, key)?.as_u64().ok_or_else(|| format!("'{key}' must be an unsigned integer"))?;
    let n = usize::try_from(n).map_err(|_| format!("'{key}' does not fit usize"))?;
    if nonzero && n == 0 { return err(format!("'{key}' must be positive")); }
    Ok(n)
}
fn required_f32(object: &Map<String, Value>, key: &str) -> Result<f32, String> {
    let n = required(object, key)?.as_f64().ok_or_else(|| format!("'{key}' must be a number"))? as f32;
    if !(n.is_finite() && n > 0.0) { return err(format!("'{key}' must be finite and positive")); }
    Ok(n)
}

impl Manifest {
    fn parse(value: Value) -> Result<Self, String> {
        let o = value.as_object().ok_or("manifest must be a JSON object")?;
        let max_updates_this_invocation = match o.get("max_updates_this_invocation") {
            None => None,
            Some(v) => {
                let n = v.as_u64().ok_or("'max_updates_this_invocation' must be an unsigned integer")?;
                let n = usize::try_from(n).map_err(|_| "'max_updates_this_invocation' does not fit usize")?;
                if n == 0 { return err("'max_updates_this_invocation' must be positive"); }
                Some(n)
            }
        };
        let preflight_only = o.get("preflight_only").map_or(Ok(false), |v| {
            v.as_bool().ok_or_else(|| "'preflight_only' must be boolean".to_string())
        })?;
        Ok(Self {
            run_dir: required_string(o, "run_dir")?,
            initial_checkpoint: required_string(o, "initial_checkpoint")?,
            train_path: required_string(o, "train_path")?,
            validation_path: required_string(o, "validation_path")?,
            prompts_path: required_string(o, "prompts_path")?,
            width: required_usize(o, "width", true)?,
            depth: required_usize(o, "depth", true)?,
            epochs: required_usize(o, "epochs", true)?,
            seed: required(o, "seed")?.as_u64().ok_or("'seed' must be an unsigned integer")?,
            chunk: required_usize(o, "chunk", true)?,
            accumulate: required_usize(o, "accumulate", true)?,
            warmup_steps: required_usize(o, "warmup_steps", false)?,
            base_lr: required_f32(o, "base_lr")?,
            state: required_usize(o, "state", true)?,
            key: required_usize(o, "key", true)?,
            memory: required_usize(o, "memory", true)?,
            checkpoint_every_updates: required_usize(o, "checkpoint_every_updates", true)?,
            diagnostics_every_updates: required_usize(o, "diagnostics_every_updates", true)?,
            expected_transitions_per_epoch: required_usize(o, "expected_transitions_per_epoch", true)?,
            max_updates_this_invocation,
            preflight_only,
        })
    }
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes { h = (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3); }
    h
}
fn file_fingerprint(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(json!({"path": path.to_string_lossy(), "bytes": bytes.len(), "fnv1a64": format!("{:016x}", fnv64(&bytes))}))
}
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    let name = path.file_name().and_then(|x| x.to_str()).unwrap_or("artifact");
    let temp = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), TEMP_NONCE.fetch_add(1, Ordering::Relaxed)));
    let result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temp)
            .map_err(|e| format!("cannot create {}: {e}", temp.display()))?;
        file.write_all(bytes).map_err(|e| format!("cannot write {}: {e}", temp.display()))?;
        file.flush().map_err(|e| format!("cannot flush {}: {e}", temp.display()))?;
        file.sync_all().map_err(|e| format!("cannot sync {}: {e}", temp.display()))?;
        fs::rename(&temp, path).map_err(|e| format!("cannot rename {}: {e}", path.display()))?;
        Ok(())
    })();
    if result.is_err() { let _ = fs::remove_file(&temp); }
    result
}
fn atomic_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut text = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    text.push(b'\n');
    atomic_write(path, &text)
}
fn read_json(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("invalid JSON {}: {e}", path.display()))
}

fn documents(raw: &str, tokenizer: &Tokenizer) -> Result<Vec<Vec<usize>>, String> {
    let mut docs = Vec::new();
    for line in raw.lines() {
        let ids = tokenizer.try_encode(line, true)?;
        if !ids.is_empty() {
            if ids.len() < 2 { return err("each nonempty training document must contain at least two tokens"); }
            docs.push(ids);
        }
    }
    if docs.is_empty() { err("dataset has no token transitions") } else { Ok(docs) }
}
fn make_plan(docs: &[Vec<usize>], chunk: usize) -> Vec<Chunk> {
    let mut out = Vec::new();
    for (document, doc) in docs.iter().enumerate() {
        let mut start = 0;
        while start + 1 < doc.len() {
            let len = chunk.min(doc.len() - 1 - start);
            out.push(Chunk { document, start, len });
            start += len;
        }
    }
    out
}
fn require_prompt_plan(raw: &str) -> Result<Value, String> {
    let plan: Value = serde_json::from_str(raw).map_err(|e| format!("invalid prompts JSON: {e}"))?;
    let sampling = plan.get("sampling").and_then(Value::as_object).ok_or("prompts JSON missing sampling")?;
    if sampling.get("seed").and_then(Value::as_u64) != Some(1337) { return err("prompts sampling.seed must be 1337"); }
    if sampling.get("max_new_tokens").and_then(Value::as_u64).filter(|x| *x > 0).is_none() { return err("prompts sampling.max_new_tokens must be positive"); }
    let temperatures = sampling.get("temperatures").and_then(Value::as_array).ok_or("prompts sampling.temperatures missing")?;
    if temperatures.is_empty() || temperatures.iter().any(|x| x.as_f64().is_none_or(|t| !t.is_finite() || t < 0.0)) { return err("prompts temperatures must be nonempty finite nonnegative numbers"); }
    let prompts = plan.get("prompts").and_then(Value::as_array).ok_or("prompts JSON missing prompts")?;
    if prompts.is_empty() || prompts.iter().any(|p| p.get("text").and_then(Value::as_str).is_none_or(str::is_empty)) { return err("every prompt must have nonempty text"); }
    Ok(plan)
}

fn prepare(manifest_path: &Path) -> Result<Prepared, String> {
    let manifest_value = read_json(manifest_path)?;
    let manifest = Manifest::parse(manifest_value.clone())?;
    if manifest.depth > PSSALayerV2::MAX_DEPTH { return err(format!("depth must be at most {}", PSSALayerV2::MAX_DEPTH)); }
    let train_raw = fs::read_to_string(&manifest.train_path).map_err(|e| format!("cannot read train_path: {e}"))?;
    let validation_raw = fs::read_to_string(&manifest.validation_path).map_err(|e| format!("cannot read validation_path: {e}"))?;
    if validation_raw.is_empty() { return err("validation_path is empty"); }
    let prompts_raw = fs::read_to_string(&manifest.prompts_path).map_err(|e| format!("cannot read prompts_path: {e}"))?;
    let prompt_plan = require_prompt_plan(&prompts_raw)?;
    let loaded = checkpoint::load_checkpoint(&manifest.initial_checkpoint).map_err(|e| format!("cannot load initial checkpoint: {e}"))?;
    if loaded.format != CheckpointFormat::V7 || loaded.model.depth() != 1 {
        return err("initial_checkpoint must be the depth-one V7 control checkpoint");
    }
    let tokenizer_json = loaded.model.tokenizer_json.as_deref().ok_or("initial checkpoint lacks BPE tokenizer metadata")?;
    let tokenizer = Tokenizer::from_serialized(tokenizer_json)?;
    if tokenizer.ordered_vocabulary()? != loaded.model.vocabulary || tokenizer.vocab_size != loaded.model.cfg.d_vocab {
        return err("initial checkpoint tokenizer metadata/vocabulary/model do not agree");
    }
    let docs = documents(&train_raw, &tokenizer)?;
    let plan = make_plan(&docs, manifest.chunk);
    let transitions: usize = plan.iter().map(|x| x.len).sum();
    if transitions != manifest.expected_transitions_per_epoch {
        return err(format!("training plan has {transitions} scored transitions, expected {}", manifest.expected_transitions_per_epoch));
    }
    let groups_per_epoch = plan.len().div_ceil(manifest.accumulate);
    let total_updates = groups_per_epoch.checked_mul(manifest.epochs).ok_or("total update count overflow")?;
    if total_updates == 0 { return err("training plan has no optimizer updates"); }
    if manifest.warmup_steps >= total_updates && manifest.warmup_steps != 0 {
        return err("warmup_steps must be less than total optimizer updates");
    }
    let cfg = PSSAConfigV2 {
        d_vocab: tokenizer.vocab_size,
        d_latent: manifest.width,
        d_state: manifest.state,
        d_mem_key: manifest.key,
        mem_capacity: manifest.memory,
        chunk_len: manifest.chunk,
        lr: manifest.base_lr,
        ..Default::default()
    };
    cfg.validate();
    let guard = json!({
        "schema_version": SCHEMA,
        "files": {
            "initial_checkpoint": file_fingerprint(&manifest.initial_checkpoint)?,
            "train": file_fingerprint(&manifest.train_path)?,
            "validation": file_fingerprint(&manifest.validation_path)?,
            "prompts": file_fingerprint(&manifest.prompts_path)?,
            "tokenizer_metadata": {"bytes": tokenizer_json.len(), "fnv1a64": format!("{:016x}", fnv64(tokenizer_json.as_bytes()))}
        },
        "model": {"d_vocab": cfg.d_vocab, "width": cfg.d_latent, "depth": manifest.depth, "state": cfg.d_state,
                  "key": cfg.d_mem_key, "memory": cfg.mem_capacity, "chunk": cfg.chunk_len, "lr": cfg.lr,
                  "beta1": cfg.beta1, "beta2": cfg.beta2, "weight_decay": cfg.weight_decay, "eps": cfg.eps,
                  "tau_mem": cfg.tau_mem, "ema_alpha": cfg.ema_alpha, "parameter_count": PSSALayerV2::new_with_depth(cfg.clone(), manifest.seed, manifest.depth).parameter_count()},
        "training_plan": {"documents": docs.len(), "chunks": plan.len(), "groups_per_epoch": groups_per_epoch,
                          "transitions_per_epoch": transitions, "epochs": manifest.epochs, "total_updates": total_updates,
                          "accumulate": manifest.accumulate, "warmup_steps": manifest.warmup_steps,
                          "expected_transitions_per_epoch": manifest.expected_transitions_per_epoch},
        "seed": manifest.seed,
        "checkpoint_every_updates": manifest.checkpoint_every_updates,
        "diagnostics_every_updates": manifest.diagnostics_every_updates,
        "prompt_plan_fnv1a64": format!("{:016x}", fnv64(prompts_raw.as_bytes())),
        "prompt_sampling": prompt_plan.get("sampling")
    });
    Ok(Prepared { manifest, tokenizer, docs, plan, groups_per_epoch, total_updates, cfg, guard, manifest_value })
}

fn recognized_entries(root: &Path) -> Result<bool, String> {
    if !root.exists() { return Ok(true); }
    for entry in fs::read_dir(root).map_err(|e| format!("cannot read run_dir: {e}"))? {
        let name = entry.map_err(|e| e.to_string())?.file_name();
        let name = name.to_string_lossy();
        if !matches!(name.as_ref(), "preflight.json" | "run.json" | "state.json" | "experiment.json" | "metrics.json" | "generations.json" | "initial.pssa" | "final.pssa" | "epochs" | "resume" | "diagnostics") {
            return Ok(false);
        }
    }
    Ok(true)
}
fn ensure_run_root(prepared: &Prepared) -> Result<(), String> {
    let root = &prepared.manifest.run_dir;
    if !root.exists() { fs::create_dir_all(root).map_err(|e| format!("cannot create run_dir: {e}"))?; }
    if !recognized_entries(root)? { return err("run_dir is nonempty and is not a recognized width_depth_study directory"); }
    let run = root.join("run.json");
    if run.exists() {
        let prior = read_json(&run)?;
        if prior.get("frozen") != Some(&prepared.guard) { return err("existing run.json frozen files/options/tokenizer/plan mismatch"); }
    }
    Ok(())
}
fn write_run_json(prepared: &Prepared) -> Result<(), String> {
    let path = prepared.manifest.run_dir.join("run.json");
    if path.exists() { return Ok(()); }
    atomic_json(&path, &json!({
        "schema_version": SCHEMA,
        "runner": "width_depth_study",
        "frozen": prepared.guard,
        "initial_manifest": prepared.manifest_value,
        "resume_policy": "state.json is authoritative and committed only after its referenced checkpoint is durable; an orphan checkpoint is ignored"
    }))
}
fn new_model(prepared: &Prepared) -> Result<PSSALayerV2, String> {
    let mut model = PSSALayerV2::new_with_depth(prepared.cfg.clone(), prepared.manifest.seed, prepared.manifest.depth);
    model.vocabulary = prepared.tokenizer.ordered_vocabulary()?;
    model.tokenizer_json = prepared.tokenizer.serialized_metadata();
    Ok(model)
}
fn same_config(a: &PSSAConfigV2, b: &PSSAConfigV2) -> bool {
    a.d_vocab == b.d_vocab && a.d_latent == b.d_latent && a.d_state == b.d_state
        && a.d_mem_key == b.d_mem_key && a.mem_capacity == b.mem_capacity
        && a.chunk_len == b.chunk_len && a.lr.to_bits() == b.lr.to_bits()
        && a.beta1.to_bits() == b.beta1.to_bits() && a.beta2.to_bits() == b.beta2.to_bits()
        && a.weight_decay.to_bits() == b.weight_decay.to_bits() && a.eps.to_bits() == b.eps.to_bits()
        && a.tau_mem.to_bits() == b.tau_mem.to_bits() && a.ema_alpha.to_bits() == b.ema_alpha.to_bits()
}
fn state_json(state: &State, prepared: &Prepared, wall: f64) -> Value {
    json!({
        "schema_version": SCHEMA,
        "frozen_fnv1a64": format!("{:016x}", fnv64(serde_json::to_string(&prepared.guard).unwrap_or_default().as_bytes())),
        "checkpoint": state.checkpoint,
        "cursor": {"epoch": state.epoch, "next_group": state.next_group,
                   "boundary": if state.next_group == prepared.groups_per_epoch { "pending_consolidation" } else { "training" }},
        "global_updates": state.global_updates,
        "train_loss_sum": state.train_loss_sum,
        "scored_transitions": state.scored_transitions,
        "epoch_loss_sum": state.epoch_loss_sum,
        "epoch_scored_transitions": state.epoch_transitions,
        "compute_seconds": state.compute_seconds,
        "epoch_compute_seconds": state.epoch_compute_seconds,
        "wall_elapsed_seconds": wall,
        "epoch_metrics": state.epoch_metrics
    })
}
fn parse_state(value: Value, prepared: &Prepared) -> Result<State, String> {
    if value.get("schema_version").and_then(Value::as_u64) != Some(SCHEMA) { return err("unsupported state.json schema"); }
    let frozen = value.get("frozen_fnv1a64").and_then(Value::as_str).ok_or("state missing frozen fingerprint")?;
    let expected = format!("{:016x}", fnv64(serde_json::to_string(&prepared.guard).unwrap_or_default().as_bytes()));
    if frozen != expected { return err("state.json does not match current frozen run configuration"); }
    let cursor = value.get("cursor").and_then(Value::as_object).ok_or("state missing cursor")?;
    let get_usize = |v: Option<&Value>, label: &str| -> Result<usize, String> { usize::try_from(v.and_then(Value::as_u64).ok_or_else(|| format!("state missing {label}"))?).map_err(|_| format!("state {label} too large")) };
    let get_f64 = |v: Option<&Value>, label: &str| -> Result<f64, String> { v.and_then(Value::as_f64).filter(|x| x.is_finite() && *x >= 0.0).ok_or_else(|| format!("state invalid {label}")) };
    let epoch = get_usize(cursor.get("epoch"), "cursor.epoch")?;
    let next_group = get_usize(cursor.get("next_group"), "cursor.next_group")?;
    if epoch > prepared.manifest.epochs || next_group > prepared.groups_per_epoch || (epoch == prepared.manifest.epochs && next_group != 0) { return err("state cursor outside frozen plan"); }
    let checkpoint = value.get("checkpoint").and_then(Value::as_str).filter(|x| !x.is_empty()).ok_or("state missing checkpoint")?.to_string();
    let epoch_metrics = value.get("epoch_metrics").and_then(Value::as_array).ok_or("state missing epoch_metrics")?.clone();
    if epoch_metrics.len() != epoch { return err("state epoch_metrics length does not match cursor"); }
    Ok(State {
        epoch, next_group, checkpoint,
        global_updates: get_usize(value.get("global_updates"), "global_updates")?,
        train_loss_sum: get_f64(value.get("train_loss_sum"), "train_loss_sum")?,
        scored_transitions: get_usize(value.get("scored_transitions"), "scored_transitions")?,
        epoch_loss_sum: get_f64(value.get("epoch_loss_sum"), "epoch_loss_sum")?,
        epoch_transitions: get_usize(value.get("epoch_scored_transitions"), "epoch_scored_transitions")?,
        compute_seconds: get_f64(value.get("compute_seconds"), "compute_seconds")?,
        epoch_compute_seconds: get_f64(value.get("epoch_compute_seconds"), "epoch_compute_seconds")?,
        wall_elapsed_seconds: get_f64(value.get("wall_elapsed_seconds"), "wall_elapsed_seconds")?,
        epoch_metrics,
    })
}
fn commit_state(root: &Path, state: &State, prepared: &Prepared, invocation: &Instant) -> Result<(), String> {
    atomic_json(&root.join("state.json"), &state_json(state, prepared, state.wall_elapsed_seconds + invocation.elapsed().as_secs_f64()))
}
fn load_or_initialize(prepared: &Prepared, invocation: &Instant) -> Result<(PSSALayerV2, State), String> {
    let root = &prepared.manifest.run_dir;
    let sidecar = root.join("state.json");
    if sidecar.exists() {
        let state = parse_state(read_json(&sidecar)?, prepared)?;
        let checkpoint_path = root.join(&state.checkpoint);
        let model = checkpoint::load_checkpoint(&checkpoint_path).map_err(|e| format!("cannot load state checkpoint {}: {e}", checkpoint_path.display()))?.model;
        if model.depth() != prepared.manifest.depth || model.cfg.d_latent != prepared.manifest.width || !same_config(&model.cfg, &prepared.cfg) || model.step_counter != state.global_updates || model.vocabulary != prepared.tokenizer.ordered_vocabulary()? || model.tokenizer_json != prepared.tokenizer.serialized_metadata() {
            return err("state checkpoint does not match frozen model/tokenizer/cursor configuration");
        }
        return Ok((model, state));
    }
    let state = State { epoch: 0, next_group: 0, checkpoint: "initial.pssa".into(), global_updates: 0,
        train_loss_sum: 0.0, scored_transitions: 0, epoch_loss_sum: 0.0, epoch_transitions: 0,
        compute_seconds: 0.0, epoch_compute_seconds: 0.0, wall_elapsed_seconds: 0.0, epoch_metrics: Vec::new() };
    let model = new_model(prepared)?;
    checkpoint::save_model(&model, root.join(&state.checkpoint)).map_err(|e| format!("cannot save initial checkpoint: {e}"))?;
    commit_state(root, &state, prepared, invocation)?;
    // Keep the pre-invocation base in memory; each sidecar serializes base plus
    // current invocation elapsed time, avoiding double-counting later commits.
    Ok((model, state))
}

fn boundary_stats(values: &[f32]) -> Value {
    let finite: Vec<f64> = values.iter().filter(|x| x.is_finite()).map(|x| f64::from(*x)).collect();
    let n = finite.len();
    let nonfinite = values.len() - n;
    if n == 0 { return json!({"count": values.len(), "finite_count": 0, "nonfinite_count": nonfinite, "mean": Value::Null, "population_variance": Value::Null, "rms": Value::Null, "min": Value::Null, "max": Value::Null}); }
    let mean = finite.iter().sum::<f64>() / n as f64;
    let variance = finite.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let rms = (finite.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    json!({"count": values.len(), "finite_count": n, "nonfinite_count": nonfinite, "mean": mean, "population_variance": variance, "rms": rms, "min": min, "max": max})
}
fn write_diagnostic(root: &Path, model: &PSSALayerV2, seq_len: usize, update: usize) -> Result<(), String> {
    let n = seq_len.checked_mul(model.cfg.d_latent).ok_or("diagnostic length overflow")?;
    if model.boundary_adjoints.len() != model.depth() + 1 || model.boundary_adjoints.iter().any(|x| x.len() < n) { return err("model boundary adjoint buffers do not satisfy depth diagnostic contract"); }
    let mut boundaries = Vec::new();
    let mut prior_rms: Option<f64> = None;
    for (i, boundary) in model.boundary_adjoints.iter().enumerate() {
        let mut item = boundary_stats(&boundary[..n]);
        let rms = item.get("rms").and_then(Value::as_f64);
        let ratio = match (rms, prior_rms) { (Some(a), Some(b)) if b != 0.0 => json!(a / b), _ => Value::Null };
        item.as_object_mut().expect("object").insert("successive_rms_ratio_to_previous_boundary".into(), ratio);
        item.as_object_mut().expect("object").insert("boundary".into(), Value::String(if i == 0 { "raw_embeddings".into() } else if i == 1 { "first_block_output".into() } else { format!("residual_block_{}_output", i - 1) }));
        prior_rms = rms;
        boundaries.push(item);
    }
    let path = root.join("diagnostics").join(format!("update-{update:09}.json"));
    // Re-execution after a crash before state commit reaches the same deterministic
    // update; a single filename prevents a duplicate record.
    if !path.exists() { atomic_json(&path, &json!({"global_update": update, "timing": "after backward before AdamW", "variance": "population variance across coordinates within one boundary", "boundaries": boundaries}))?; }
    Ok(())
}
fn save_resume_checkpoint(root: &Path, model: &PSSALayerV2, state: &mut State, prepared: &Prepared, invocation: &Instant) -> Result<(), String> {
    let relative = format!("resume/update-{:09}.pssa", state.global_updates);
    checkpoint::save_model(model, root.join(&relative)).map_err(|e| format!("cannot save resume checkpoint: {e}"))?;
    state.checkpoint = relative;
    commit_state(root, state, prepared, invocation)?; // commit-last: this makes checkpoint authoritative.
    prune_resume(root, &state.checkpoint)?;
    Ok(())
}
fn prune_resume(root: &Path, authoritative: &str) -> Result<(), String> {
    let dir = root.join("resume");
    if !dir.exists() { return Ok(()); }
    let mut entries = fs::read_dir(&dir).map_err(|e| e.to_string())?.filter_map(Result::ok)
        .filter(|e| e.file_type().map(|x| x.is_file()).unwrap_or(false)).collect::<Vec<_>>();
    entries.sort_by_key(|e| e.file_name());
    while entries.len() > RESUME_KEEP {
        let old = entries.remove(0);
        let rel = format!("resume/{}", old.file_name().to_string_lossy());
        if rel != authoritative { fs::remove_file(old.path()).map_err(|e| format!("cannot prune stale resume checkpoint: {e}"))?; }
    }
    Ok(())
}

fn evaluate_epoch(_root: &Path, epoch_path: &Path, prepared: &Prepared, state: &State) -> Result<Value, String> {
    let mut eval_model = checkpoint::load_checkpoint(epoch_path).map_err(|e| format!("cannot reload epoch checkpoint for validation: {e}"))?.model;
    let raw = fs::read_to_string(&prepared.manifest.validation_path).map_err(|e| format!("cannot read validation_path: {e}"))?;
    let (nll, transitions, correct, oov) = CLIHandler::evaluate_corpus(&mut eval_model, &prepared.tokenizer, &raw)?;
    let perplexity = nll.exp();
    Ok(json!({"epoch": state.epoch + 1, "train_nll": state.epoch_loss_sum / state.epoch_transitions.max(1) as f64,
        "train_scored_transitions": state.epoch_transitions, "validation_nll": nll,
        "validation_perplexity": if perplexity.is_finite() { json!(perplexity) } else { Value::Null },
        "validation_perplexity_overflow": !perplexity.is_finite(), "validation_scored_transitions": transitions,
        "validation_accuracy": correct as f64 / transitions.max(1) as f64, "validation_correct": correct,
        "validation_oov_count": oov, "compute_seconds_epoch": state.epoch_compute_seconds,
        "optimizer_updates": state.global_updates,
        "evaluation": "separately reloaded checkpoint; teacher-forced; recurrent reset per document"}))
}
fn generations(root: &Path, final_checkpoint: &Path, prepared: &Prepared) -> Result<(), String> {
    let prompt_value = require_prompt_plan(&fs::read_to_string(&prepared.manifest.prompts_path).map_err(|e| e.to_string())?)?;
    let sampling = prompt_value["sampling"].as_object().ok_or("prompts sampling missing")?;
    let max_new = usize::try_from(sampling["max_new_tokens"].as_u64().ok_or("prompts max_new_tokens missing")?).map_err(|_| "max_new_tokens too large")?;
    let temps = sampling["temperatures"].as_array().ok_or("prompts temperatures missing")?;
    let prompts = prompt_value["prompts"].as_array().ok_or("prompts missing")?;
    let mut model = checkpoint::load_checkpoint(final_checkpoint).map_err(|e| format!("cannot reload final checkpoint for generation: {e}"))?.model;
    let mut samples = Vec::new();
    for prompt in prompts {
        let text = prompt["text"].as_str().ok_or("prompt text missing")?;
        for temperature in temps {
            let temperature = temperature.as_f64().ok_or("invalid temperature")? as f32;
            let config = InferenceConfig { temperature, top_p: 1.0, top_k: prepared.tokenizer.vocab_size, repetition_penalty: 1.0, max_new_tokens: max_new };
            // A newly constructed engine supplies seed 1337 for each sample.
            let completion = PSSAInferenceEngine::try_new(&mut model, &prepared.tokenizer)?.try_generate_chat_turn(text, &config, |_| {})?;
            samples.push(json!({"prompt": text, "temperature": temperature, "completion": completion}));
        }
    }
    atomic_json(&root.join("generations.json"), &json!({"sampling": {"seed_per_sample": 1337, "top_p": 1.0, "top_k": prepared.tokenizer.vocab_size, "repetition_penalty": 1.0, "max_new_tokens": max_new, "token_policy": "existing inference engine excludes token 0 (<unk>); no other candidate truncation"}, "selection": "all prompts and temperatures retained in input order, no rerolls", "samples": samples}))
}
fn publish(root: &Path, prepared: &Prepared, state: &State, status: &str, detail: Option<&str>, invocation: &Instant) -> Result<(), String> {
    let wall = state.wall_elapsed_seconds + invocation.elapsed().as_secs_f64();
    atomic_json(&root.join("metrics.json"), &json!({"epoch_metrics": state.epoch_metrics, "training": {"scored_transitions": state.scored_transitions, "global_updates": state.global_updates, "compute_seconds": state.compute_seconds, "transitions_per_compute_second": state.scored_transitions as f64 / state.compute_seconds.max(f64::MIN_POSITIVE), "wall_elapsed_seconds": wall}}))?;
    atomic_json(&root.join("experiment.json"), &json!({"schema_version": SCHEMA, "status": status, "detail": detail, "frozen": prepared.guard, "metrics": {"scored_transitions": state.scored_transitions, "global_updates": state.global_updates, "compute_seconds": state.compute_seconds, "wall_elapsed_seconds": wall}, "cursor": {"epoch": state.epoch, "next_group": state.next_group}, "checkpoint": state.checkpoint}))
}

fn finish_epoch(root: &Path, model: &mut PSSALayerV2, state: &mut State, prepared: &Prepared, invocation: &Instant) -> Result<(), String> {
    if state.next_group != prepared.groups_per_epoch { return err("internal epoch boundary called before all groups"); }
    let timer = Instant::now();
    model.ema_consolidate_plasticity();
    let finite = model.all_parameters_finite();
    let elapsed = timer.elapsed().as_secs_f64();
    state.compute_seconds += elapsed;
    state.epoch_compute_seconds += elapsed;
    if !finite { return err("numerical-failure: non-finite parameters after epoch consolidation"); }
    let epoch_path = root.join("epochs").join(format!("epoch-{:04}.pssa", state.epoch + 1));
    checkpoint::save_model(model, &epoch_path).map_err(|e| format!("cannot save epoch checkpoint: {e}"))?;
    let metric = evaluate_epoch(root, &epoch_path, prepared, state)?;
    state.epoch_metrics.push(metric);
    state.epoch += 1;
    state.next_group = 0;
    state.epoch_loss_sum = 0.0;
    state.epoch_transitions = 0;
    state.epoch_compute_seconds = 0.0;
    state.checkpoint = format!("epochs/epoch-{:04}.pssa", state.epoch);
    if state.epoch == prepared.manifest.epochs {
        let final_path = root.join("final.pssa");
        checkpoint::save_model(model, &final_path).map_err(|e| format!("cannot save final checkpoint: {e}"))?;
        generations(root, &final_path, prepared)?;
    }
    commit_state(root, state, prepared, invocation)?;
    Ok(())
}

fn run_prepared(prepared: &Prepared, invocation: &Instant) -> Result<String, String> {
    ensure_run_root(prepared)?;
    let root = &prepared.manifest.run_dir;
    if prepared.manifest.preflight_only {
        let model = new_model(prepared)?;
        atomic_json(&root.join("preflight.json"), &json!({"schema_version": SCHEMA, "status": "preflight-ok", "frozen": prepared.guard, "actual_transitions_per_epoch": prepared.manifest.expected_transitions_per_epoch, "chunks": prepared.plan.len(), "groups_per_epoch": prepared.groups_per_epoch, "total_updates": prepared.total_updates, "parameter_count": model.parameter_count()}))?;
        return Ok("preflight-ok".into());
    }
    write_run_json(prepared)?;
    let (mut model, mut state) = load_or_initialize(prepared, invocation)?;
    if state.epoch == prepared.manifest.epochs {
        publish(root, prepared, &state, "complete", None, invocation)?;
        return Ok("complete".into());
    }
    let mut updates_this_invocation = 0usize;
    loop {
        if state.epoch == prepared.manifest.epochs { break; }
        if state.next_group == prepared.groups_per_epoch {
            if let Err(e) = finish_epoch(root, &mut model, &mut state, prepared, invocation) {
                let status = if e.contains("numerical-failure") { "numerical-failure" } else { "error" };
                let _ = publish(root, prepared, &state, status, Some(&e), invocation);
                return Err(e);
            }
            continue;
        }
        if prepared.manifest.max_updates_this_invocation.is_some_and(|limit| updates_this_invocation >= limit) {
            save_resume_checkpoint(root, &model, &mut state, prepared, invocation)?;
            publish(root, prepared, &state, "paused-budget", None, invocation)?;
            return Ok("paused-budget".into());
        }
        let group_end = (state.next_group + prepared.manifest.accumulate).min(prepared.plan.len());
        let group = &prepared.plan[state.next_group * prepared.manifest.accumulate..group_end];
        let total_tokens: usize = group.iter().map(|x| x.len).sum();
        if total_tokens == 0 { return err("frozen plan contains an empty update group"); }
        let timer = Instant::now();
        model.zero_gradients();
        let elapsed = timer.elapsed().as_secs_f64();
        state.compute_seconds += elapsed;
        state.epoch_compute_seconds += elapsed;
        let update_number = state.global_updates + 1;
        for (index, chunk) in group.iter().enumerate() {
            // Resets match the CLI exactly, but are intentionally outside the
            // compute interval defined by the study protocol.
            if chunk.start == 0 { model.reset_recurrent_state(); }
            let doc = &prepared.docs[chunk.document];
            let input = &doc[chunk.start..chunk.start + chunk.len];
            let target = &doc[chunk.start + 1..chunk.start + 1 + chunk.len];
            let timer = Instant::now();
            let loss = model.forward_train_chunk(input, target);
            if loss.is_finite() { model.backward_chunk(chunk.len, chunk.len as f32 / total_tokens as f32); }
            let elapsed = timer.elapsed().as_secs_f64();
            state.compute_seconds += elapsed;
            state.epoch_compute_seconds += elapsed;
            if !loss.is_finite() {
                let message = "numerical-failure: non-finite training loss; last committed checkpoint remains authoritative";
                let _ = publish(root, prepared, &state, "numerical-failure", Some(message), invocation);
                return err(message);
            }
            // Diagnostics are the loss-normalized adjoints from the last chunk,
            // after backward and before AdamW. Formatting/I/O is not timed.
            if index + 1 == group.len() && update_number % prepared.manifest.diagnostics_every_updates == 0 {
                write_diagnostic(root, &model, chunk.len, update_number)?;
            }
            let timer = Instant::now();
            model.insert_training_memory(loss, chunk.len);
            let elapsed = timer.elapsed().as_secs_f64();
            state.compute_seconds += elapsed;
            state.epoch_compute_seconds += elapsed;
            state.train_loss_sum += f64::from(loss) * chunk.len as f64;
            state.epoch_loss_sum += f64::from(loss) * chunk.len as f64;
            state.scored_transitions += chunk.len;
            state.epoch_transitions += chunk.len;
        }
        let lr = learning_rate_for_update(prepared.manifest.base_lr, update_number, prepared.total_updates, prepared.manifest.warmup_steps)?;
        let timer = Instant::now();
        model.apply_adamw(lr);
        let finite = model.all_parameters_finite();
        let elapsed = timer.elapsed().as_secs_f64();
        state.compute_seconds += elapsed;
        state.epoch_compute_seconds += elapsed;
        if !finite {
            let message = "numerical-failure: non-finite parameters after AdamW; last committed checkpoint remains authoritative";
            let _ = publish(root, prepared, &state, "numerical-failure", Some(message), invocation);
            return err(message);
        }
        state.global_updates = update_number;
        state.next_group += 1;
        updates_this_invocation += 1;
        if state.global_updates % prepared.manifest.checkpoint_every_updates == 0 {
            save_resume_checkpoint(root, &model, &mut state, prepared, invocation)?;
        }
    }
    publish(root, prepared, &state, "complete", None, invocation)?;
    Ok("complete".into())
}

fn record_error(prepared: &Prepared, error: &str, invocation: &Instant) {
    let root = &prepared.manifest.run_dir;
    if !root.exists() { return; }
    let status = if error.contains("numerical-failure") { "numerical-failure" } else { "error" };
    if let Ok(state_value) = read_json(&root.join("state.json")) {
        if let Ok(state) = parse_state(state_value, prepared) {
            let _ = publish(root, prepared, &state, status, Some(error), invocation);
            return;
        }
    }
    let _ = atomic_json(&root.join("experiment.json"), &json!({"schema_version": SCHEMA, "status": status, "detail": error, "frozen": prepared.guard, "last_committed_state": Value::Null}));
}
fn run_manifest(path: &Path) -> Result<String, String> {
    let invocation = Instant::now();
    let prepared = prepare(path)?;
    match run_prepared(&prepared, &invocation) {
        Ok(status) => Ok(status),
        Err(error) => { record_error(&prepared, &error, &invocation); Err(error) }
    }
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 { eprintln!("error: usage: width_depth_study MANIFEST.json"); std::process::exit(2); }
    match run_manifest(Path::new(&args[1])) {
        Ok(status) => println!("{}", json!({"status": status})),
        Err(error) => { eprintln!("error: {error}"); std::process::exit(1); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, bytes: &[u8]) { atomic_write(path, bytes).unwrap(); }
    fn tiny_manifest(root: &Path, initial: &Path, train: &Path, validation: &Path, prompts: &Path, budget: Option<usize>) -> PathBuf {
        let tokenizer = {
            let loaded = checkpoint::load_checkpoint(initial).unwrap();
            Tokenizer::from_serialized(loaded.model.tokenizer_json.as_deref().unwrap()).unwrap()
        };
        let transitions: usize = documents(&fs::read_to_string(train).unwrap(), &tokenizer).unwrap().iter().map(|x| x.len() - 1).sum();
        let mut value = json!({"run_dir": root.to_string_lossy().to_string(), "initial_checkpoint": initial.to_string_lossy().to_string(), "train_path": train.to_string_lossy().to_string(), "validation_path": validation.to_string_lossy().to_string(), "prompts_path": prompts.to_string_lossy().to_string(),
            "width": 8, "depth": 1, "epochs": 2, "seed": 42, "chunk": 3, "accumulate": 2, "warmup_steps": 0,
            "base_lr": 0.001, "state": 2, "key": 2, "memory": 2, "checkpoint_every_updates": 256,
            "diagnostics_every_updates": 1, "expected_transitions_per_epoch": transitions});
        if let Some(n) = budget { value.as_object_mut().unwrap().insert("max_updates_this_invocation".into(), json!(n)); }
        let path = root.with_extension("manifest.json"); atomic_json(&path, &value).unwrap(); path
    }
    fn strip_timing(value: &mut Value) {
        match value {
            Value::Object(o) => { o.remove("compute_seconds"); o.remove("compute_seconds_epoch"); o.remove("wall_elapsed_seconds"); o.remove("transitions_per_compute_second"); for x in o.values_mut() { strip_timing(x); } }
            Value::Array(a) => for x in a { strip_timing(x); },
            _ => {}
        }
    }
    #[test]
    fn uninterrupted_and_segmented_resume_match_and_reject_mismatch() {
        let base = std::env::temp_dir().join(format!("width-depth-study-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base); fs::create_dir_all(&base).unwrap();
        let train = base.join("train.txt"); let validation = base.join("validation.txt"); let prompts = base.join("prompts.json");
        write(&train, b"alpha beta gamma delta epsilon\none two three four five\n"); write(&validation, b"alpha beta gamma\none two three\n");
        atomic_json(&prompts, &json!({"sampling":{"temperatures":[0.0],"max_new_tokens":1,"seed":1337},"prompts":[{"text":"alpha"}]})).unwrap();
        let tokenizer = Tokenizer::from_corpus_bpe(&fs::read_to_string(&train).unwrap(), 257).unwrap();
        let mut initial = PSSALayerV2::new(PSSAConfigV2 { d_vocab: tokenizer.vocab_size, d_latent: 8, d_state: 2, d_mem_key: 2, mem_capacity: 2, chunk_len: 3, ..Default::default() }, 42);
        initial.vocabulary = tokenizer.ordered_vocabulary().unwrap(); initial.tokenizer_json = tokenizer.serialized_metadata();
        let initial_path = base.join("initial.pssa"); checkpoint::save_model(&initial, &initial_path).unwrap();
        let all_root = base.join("all"); let all_manifest = tiny_manifest(&all_root, &initial_path, &train, &validation, &prompts, None);
        assert_eq!(run_manifest(&all_manifest).unwrap(), "complete");
        let segmented_root = base.join("segmented"); let segmented_manifest = tiny_manifest(&segmented_root, &initial_path, &train, &validation, &prompts, Some(1));
        for _ in 0..32 { if run_manifest(&segmented_manifest).unwrap() == "complete" { break; } }
        assert_eq!(fs::read(all_root.join("final.pssa")).unwrap(), fs::read(segmented_root.join("final.pssa")).unwrap());
        let mut all_metrics = read_json(&all_root.join("metrics.json")).unwrap(); let mut segmented_metrics = read_json(&segmented_root.join("metrics.json")).unwrap();
        strip_timing(&mut all_metrics); strip_timing(&mut segmented_metrics); assert_eq!(all_metrics, segmented_metrics);
        write(&train, b"changed corpus bytes\n");
        assert!(run_manifest(&segmented_manifest).is_err(), "changed corpus must be rejected by frozen provenance guard");
        write(&train, b"alpha beta gamma delta epsilon\none two three four five\n");
        let mut changed_manifest = read_json(&segmented_manifest).unwrap();
        changed_manifest.as_object_mut().unwrap().insert("base_lr".into(), json!(0.002));
        atomic_json(&segmented_manifest, &changed_manifest).unwrap();
        assert!(run_manifest(&segmented_manifest).is_err(), "changed training options must be rejected by frozen provenance guard");
        let _ = fs::remove_dir_all(&base);
    }
}
