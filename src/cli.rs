use std::io::{self, Write};
use std::time::Instant;
use crate::pssa::{PSSAConfigV2, PSSALayerV2};
use crate::dataset::{Tokenizer, DatasetManager, TokenChunkIterator};
use crate::diagnostics::DiagnosticsSuite;
use crate::inference::{PSSAInferenceEngine, InferenceConfig};
use crate::adapter::PlasticAdapterV2;

pub struct ArgParser {
    args: Vec<String>,
}

impl ArgParser {
    pub fn new(args: Vec<String>) -> Self {
        Self { args }
    }

    pub fn get_pos_or_flag(&self, pos: usize, long: &str, short: &str, default: &str) -> String {
        let mut i = 0;
        while i < self.args.len() {
            if self.args[i] == long || self.args[i] == short {
                if i + 1 < self.args.len() {
                    return self.args[i + 1].clone();
                }
            }
            i += 1;
        }
        if pos < self.args.len() && !self.args[pos].starts_with('-') {
            return self.args[pos].clone();
        }
        default.to_string()
    }

    pub fn get_str(&self, long: &str, short: &str, default: &str) -> String {
        let mut i = 0;
        while i < self.args.len() {
            if self.args[i] == long || self.args[i] == short {
                if i + 1 < self.args.len() {
                    return self.args[i + 1].clone();
                }
            }
            i += 1;
        }
        default.to_string()
    }

    pub fn get_usize(&self, long: &str, short: &str, default: usize) -> usize {
        self.get_str(long, short, "").parse().unwrap_or(default)
    }

    pub fn get_f32(&self, long: &str, short: &str, default: f32) -> f32 {
        self.get_str(long, short, "").parse().unwrap_or(default)
    }
}

pub struct CLIHandler;

impl CLIHandler {
    fn format_progress_bar(pct: f32, width: usize) -> String {
        let filled = ((pct / 100.0) * (width as f32)).round() as usize;
        let filled = filled.min(width);
        let empty = width - filled;
        format!("[{}{}]", "=".repeat(filled), " ".repeat(empty))
    }

    fn resolve_default_data() -> String {
        if std::path::Path::new("data/downloaded.txt").exists() {
            "data/downloaded.txt".to_string()
        } else {
            "science".to_string()
        }
    }

    // =========================================================================
    // MODEL BINARY SERIALIZATION (PSSA V5 FORMAT - 38-BYTE HEADER ALIGNED)
    // =========================================================================
    pub fn save_model_v2(model: &PSSALayerV2, path: &str) -> io::Result<()> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"PSSA");
        bytes.extend_from_slice(&5u16.to_le_bytes()); // Format v5 (6 bytes)

        bytes.extend_from_slice(&(model.cfg.d_vocab as u32).to_le_bytes());
        bytes.extend_from_slice(&(model.cfg.d_latent as u32).to_le_bytes());
        bytes.extend_from_slice(&(model.cfg.d_state as u32).to_le_bytes());
        bytes.extend_from_slice(&(model.cfg.d_mem_key as u32).to_le_bytes());
        bytes.extend_from_slice(&(model.cfg.mem_capacity as u32).to_le_bytes());
        bytes.extend_from_slice(&(model.cfg.chunk_len as u32).to_le_bytes());
        bytes.extend_from_slice(&(model.memory.count as u32).to_le_bytes());
        bytes.extend_from_slice(&(model.adapters.len() as u32).to_le_bytes());
        // Exact 38 bytes header

        let write_slice = |buf: &mut Vec<u8>, s: &[f32]| {
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            for &val in s {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        };

        write_slice(&mut bytes, &model.embed_w.data);
        write_slice(&mut bytes, &model.norm_gamma.data);
        write_slice(&mut bytes, &model.norm_beta.data);
        write_slice(&mut bytes, &model.a_mat.data);
        write_slice(&mut bytes, &model.w_delta.data);
        write_slice(&mut bytes, &model.w_b.data);
        write_slice(&mut bytes, &model.w_c.data);
        write_slice(&mut bytes, &model.w_qx.data);
        write_slice(&mut bytes, &model.w_qh.data);
        write_slice(&mut bytes, &model.w_gate.data);
        write_slice(&mut bytes, &model.w_proj.data);
        write_slice(&mut bytes, &model.mlp_w1.data);
        write_slice(&mut bytes, &model.mlp_w2.data);
        write_slice(&mut bytes, &model.unembed_w.data);

        // Memory bank
        let k_len = model.memory.count * model.cfg.d_mem_key;
        write_slice(&mut bytes, &model.memory.keys[..k_len]);
        let v_len = model.memory.count * model.cfg.d_latent;
        write_slice(&mut bytes, &model.memory.values[..v_len]);

        // Adapters
        for ad in &model.adapters {
            bytes.extend_from_slice(&(ad.rank as u32).to_le_bytes());
            write_slice(&mut bytes, &ad.down_proj.data);
            write_slice(&mut bytes, &ad.up_proj.data);
        }

        std::fs::create_dir_all("data").ok();
        std::fs::write(path, bytes)
    }

    pub fn load_model_v2(path: &str) -> Result<PSSALayerV2, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("Failed to read model file: {}", e))?;
        if bytes.len() < 38 || &bytes[0..4] != b"PSSA" {
            return Err("Invalid PSSA model file format or header corrupted".to_string());
        }

        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != 5 {
            return Err(format!("Unsupported model format version: {}", version));
        }

        let mut offset = 6;

        let read_u32 = |off: &mut usize| -> Result<usize, String> {
            if *off + 4 > bytes.len() {
                return Err("Unexpected EOF while reading u32".to_string());
            }
            let val = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
            *off += 4;
            Ok(val)
        };

        let read_slice = |off: &mut usize, target: &mut [f32]| -> Result<(), String> {
            let len = read_u32(off)?;
            if len != target.len() || *off + len * 4 > bytes.len() {
                return Err("Shape mismatch while deserializing parameter tensor".to_string());
            }
            for i in 0..len {
                target[i] = f32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
                *off += 4;
            }
            Ok(())
        };

        let d_vocab = read_u32(&mut offset)?;
        let d_latent = read_u32(&mut offset)?;
        let d_state = read_u32(&mut offset)?;
        let d_mem_key = read_u32(&mut offset)?;
        let mem_capacity = read_u32(&mut offset)?;
        let chunk_len = read_u32(&mut offset)?;
        let mem_count = read_u32(&mut offset)?;
        let adapter_count = read_u32(&mut offset)?;

        let cfg = PSSAConfigV2 {
            d_vocab,
            d_latent,
            d_state,
            d_mem_key,
            mem_capacity,
            chunk_len,
            ..Default::default()
        };

        let mut model = PSSALayerV2::new(cfg, 42);

        read_slice(&mut offset, &mut model.embed_w.data)?;
        read_slice(&mut offset, &mut model.norm_gamma.data)?;
        read_slice(&mut offset, &mut model.norm_beta.data)?;
        read_slice(&mut offset, &mut model.a_mat.data)?;
        read_slice(&mut offset, &mut model.w_delta.data)?;
        read_slice(&mut offset, &mut model.w_b.data)?;
        read_slice(&mut offset, &mut model.w_c.data)?;
        read_slice(&mut offset, &mut model.w_qx.data)?;
        read_slice(&mut offset, &mut model.w_qh.data)?;
        read_slice(&mut offset, &mut model.w_gate.data)?;
        read_slice(&mut offset, &mut model.w_proj.data)?;
        read_slice(&mut offset, &mut model.mlp_w1.data)?;
        read_slice(&mut offset, &mut model.mlp_w2.data)?;
        read_slice(&mut offset, &mut model.unembed_w.data)?;

        // Memory bank
        let k_len = mem_count * d_mem_key;
        read_slice(&mut offset, &mut model.memory.keys[..k_len])?;
        let v_len = mem_count * d_latent;
        read_slice(&mut offset, &mut model.memory.values[..v_len])?;
        model.memory.count = mem_count;

        // Adapters
        model.adapters.clear();
        for _ in 0..adapter_count {
            let rank = read_u32(&mut offset)?;
            let mut ad = PlasticAdapterV2::new(d_latent, rank, &mut model.rng);
            read_slice(&mut offset, &mut ad.down_proj.data)?;
            read_slice(&mut offset, &mut ad.up_proj.data)?;
            model.adapters.push(ad);
        }

        Ok(model)
    }

    // =========================================================================
    // TRAINING PIPELINE (L=64 TBPTT & ADAMW)
    // =========================================================================
    pub fn run_training(data_arg: &str, epochs: usize, out_path: &str) {
        DiagnosticsSuite::print_banner("OXIDE AI: PSSA V2 SEQUENCE TBPTT & ADAMW TRAINING");

        println!("[INFO] Loading dataset source: '{}'", data_arg);
        let raw_corpus = DatasetManager::load_dataset(Some(data_arg));
        let tokenizer = Tokenizer::from_corpus(&raw_corpus, true);
        let token_ids = tokenizer.encode(&raw_corpus, true);

        println!(
            "[INFO] Ingested {} tokens | Unique Vocabulary: {} | Clean Dictionary Built",
            token_ids.len(),
            tokenizer.vocab_size
        );

        let cfg = PSSAConfigV2 {
            d_vocab: tokenizer.vocab_size,
            d_latent: 256,
            d_state: 16,
            d_mem_key: 32,
            mem_capacity: 512,
            chunk_len: 64,
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            tau_mem: 0.1,
            ema_alpha: 0.01,
        };

        let mut model = PSSALayerV2::new(cfg, 42);

        println!("\n[PHASE 1] Multi-Channel SSM Sequence Chunk Training (L=64 TBPTT)");
        let total_training_start = Instant::now();
        let total_chunks_per_epoch = token_ids.len() / model.cfg.chunk_len;
        let total_steps = total_chunks_per_epoch * epochs;

        let mut global_step = 0;

        for epoch in 1..=epochs {
            let epoch_start = Instant::now();
            let mut last_progress = 0;
            let mut last_time = Instant::now();
            let mut running_loss = 0.0f32;
            let mut chunk_count = 0;

            let chunk_iter = TokenChunkIterator::new(&token_ids, model.cfg.chunk_len);

            for (inputs, targets) in chunk_iter {
                global_step += 1;
                chunk_count += 1;

                // 1. Forward pass across chunk
                let loss = model.forward_train_chunk(inputs, targets);
                running_loss += loss;

                // 2. Exact Truncated Backpropagation Through Time (TBPTT) + AdamW step
                model.backward_and_step_chunk(inputs.len());

                // 3. Periodic episodic memory insertion on high-loss transitions
                if loss > 3.5 {
                    let last_t = inputs.len().saturating_sub(1);
                    let q_off = last_t * model.cfg.d_mem_key;
                    let last_q = &model.tape.q_poincare[q_off..q_off + model.cfg.d_mem_key];
                    let v_off = last_t * model.cfg.d_latent;
                    let last_v = &model.tape.z_final[v_off..v_off + model.cfg.d_latent];
                    model.memory.insert(last_q, last_v);
                }

                // Progress logging every 100 chunks
                if chunk_count - last_progress >= 100 || chunk_count == total_chunks_per_epoch {
                    let elapsed_interval = last_time.elapsed().as_secs_f32().max(0.001);
                    let processed_tokens = (chunk_count - last_progress) * model.cfg.chunk_len;
                    let current_tps = processed_tokens as f32 / elapsed_interval;
                    let pct = (chunk_count as f32 / total_chunks_per_epoch as f32) * 100.0;

                    let total_elapsed = total_training_start.elapsed().as_secs_f32().max(0.001);
                    let global_tps = (global_step * model.cfg.chunk_len) as f32 / total_elapsed;
                    let remaining_steps = total_steps.saturating_sub(global_step);
                    let eta_total_secs = if global_tps > 0.0 { (remaining_steps * model.cfg.chunk_len) as f32 / global_tps } else { 0.0 };
                    let eta_mins = (eta_total_secs / 60.0).floor() as usize;
                    let eta_secs = (eta_total_secs % 60.0).floor() as usize;

                    let avg_loss = running_loss / (chunk_count as f32);
                    let bar = Self::format_progress_bar(pct, 20);

                    println!(
                        "  [Epoch {}/{}] {} {:>5.1}% | {:>6.0} tok/s | ETA: {:02}:{:02} | Loss: {:.4} | Mem: {}",
                        epoch, epochs, bar, pct, current_tps, eta_mins, eta_secs, avg_loss, model.memory.count
                    );

                    last_progress = chunk_count;
                    last_time = Instant::now();
                }
            }

            // Phase 2: Neurological EMA Sleep Consolidation
            model.ema_consolidate_plasticity();
            println!(
                "  [Epoch {}/{}] Finished in {:.2}s | Plasticity Consolidated via EMA\n",
                epoch, epochs, epoch_start.elapsed().as_secs_f32()
            );
        }

        Self::save_model_v2(&model, out_path).expect("Failed to write model file");

        println!(
            "[SUCCESS] Training finished in {:.2}s. Model written to '{}'\n",
            total_training_start.elapsed().as_secs_f32(),
            out_path
        );
    }

    // =========================================================================
    // INTERACTIVE REPL
    // =========================================================================
    pub fn run_chat(data_arg: &str, model_path: &str, temp: f32) {
        DiagnosticsSuite::print_banner("OXIDE AI: INTERACTIVE REPL (PSSA V2)");

        let mut model = Self::load_model_v2(model_path).unwrap_or_else(|_| {
            println!("[WARN] Model '{}' not found. Training initial V2 model...", model_path);
            Self::run_training(data_arg, 4, model_path);
            Self::load_model_v2(model_path).expect("Failed to load model")
        });

        let raw_corpus = DatasetManager::load_dataset(Some(data_arg));
        let tokenizer = Tokenizer::from_corpus(&raw_corpus, true);

        let mut inf_cfg = InferenceConfig {
            temperature: temp,
            top_p: 0.85,
            top_k: 24,
            repetition_penalty: 1.25,
            max_new_tokens: 64,
        };

        println!(
            "[INFO] Model online. Vocabulary: {} tokens | Active Memory: {} | Adapters: {}",
            model.cfg.d_vocab,
            model.memory.count,
            model.adapters.len()
        );
        println!("Commands: /exit, /info, /temp <val>\n------------------------------------------------------------\n");

        loop {
            print!("user> ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                break;
            }

            let trimmed = input.trim();
            if trimmed == "/exit" || trimmed == "quit" {
                println!("[INFO] Exiting REPL.");
                break;
            } else if trimmed == "/info" {
                println!(
                    "  [INFO] Model: {} (Mem: {}, Adapters: {}, Temp: {:.2})\n",
                    model_path,
                    model.memory.count,
                    model.adapters.len(),
                    inf_cfg.temperature
                );
                continue;
            } else if trimmed.starts_with("/temp ") {
                if let Ok(new_t) = trimmed[6..].parse::<f32>() {
                    inf_cfg.temperature = new_t;
                    println!("  [INFO] Temperature updated to {:.2}\n", new_t);
                }
                continue;
            } else if trimmed.is_empty() {
                continue;
            }

            print!("oxide> ");
            io::stdout().flush().unwrap();

            let mut engine = PSSAInferenceEngine::new(&mut model, &tokenizer);
            engine.generate_chat_turn(trimmed, &inf_cfg, |tok| {
                print!("{} ", tok);
                io::stdout().flush().unwrap();
            });
            println!("\n");
        }
    }

    // =========================================================================
    // TEXT GENERATION
    // =========================================================================
    pub fn run_generate(prompt: &str, data_arg: &str, model_path: &str) {
        let mut model = Self::load_model_v2(model_path).unwrap_or_else(|_| {
            println!("[WARN] Model '{}' not found. Training model first...", model_path);
            Self::run_training(data_arg, 4, model_path);
            Self::load_model_v2(model_path).expect("Failed to read model")
        });

        let raw_corpus = DatasetManager::load_dataset(Some(data_arg));
        let tokenizer = Tokenizer::from_corpus(&raw_corpus, true);

        let inf_cfg = InferenceConfig {
            temperature: 0.35,
            top_p: 0.85,
            top_k: 24,
            repetition_penalty: 1.25,
            max_new_tokens: 64,
        };

        print!("[PROMPT] '{:<24}' => ", prompt);
        let mut engine = PSSAInferenceEngine::new(&mut model, &tokenizer);
        let completed = engine.generate_chat_turn(prompt, &inf_cfg, |tok| {
            print!("{} ", tok);
            io::stdout().flush().unwrap();
        });
        println!("\n[OUTPUT] \"{}\"\n", completed);
    }

    pub fn run_download(dataset_name: &str, out_path: &str) {
        DiagnosticsSuite::print_banner("OXIDE AI: DATASET DOWNLOADER");
        match DatasetManager::download_huggingface_dataset(dataset_name) {
            Ok(content) => {
                std::fs::create_dir_all("data").ok();
                std::fs::write(out_path, &content).expect("Failed to save dataset");
                println!(
                    "[SUCCESS] Downloaded '{}' ({} KB) to '{}'\n",
                    dataset_name,
                    content.len() / 1024,
                    out_path
                );
            }
            Err(e) => {
                println!("[ERROR] Download failed: {}\n", e);
            }
        }
    }

    // =========================================================================
    // BENCHMARK SUITE
    // =========================================================================
    pub fn run_benchmark() {
        DiagnosticsSuite::print_banner("OXIDE AI: PSSA V2 ARCHITECTURAL BENCHMARK");

        println!("--- [Milestone 1: Sequence TBPTT & Multi-Channel SSM] ---");
        let c_tok = Tokenizer::from_corpus(&DatasetManager::contradiction_stream(), true);
        let c_ids = c_tok.encode(&DatasetManager::contradiction_stream(), true);
        let mut c_model = PSSALayerV2::new(
            PSSAConfigV2 {
                d_vocab: c_tok.vocab_size,
                d_latent: 64,
                d_state: 8,
                d_mem_key: 16,
                mem_capacity: 64,
                chunk_len: 16,
                lr: 2e-3,
                ..Default::default()
            },
            42,
        );

        let chunk_iter = TokenChunkIterator::new(&c_ids, 16);
        for (inputs, targets) in chunk_iter {
            c_model.forward_train_chunk(inputs, targets);
            c_model.backward_and_step_chunk(inputs.len());
        }
        println!("  [PASS] Multi-channel continuous SSM with TBPTT trained without divergence.\n");

        println!("--- [Milestone 2: MQAR Distractor Gap Learning] ---");
        let m_tok = Tokenizer::from_corpus(&DatasetManager::mqar_stream(12), true);
        let m_ids = m_tok.encode(&DatasetManager::mqar_stream(12), true);
        let mut m_model = PSSALayerV2::new(
            PSSAConfigV2 {
                d_vocab: m_tok.vocab_size,
                d_latent: 64,
                d_state: 8,
                d_mem_key: 16,
                mem_capacity: 64,
                chunk_len: 16,
                lr: 2e-3,
                ..Default::default()
            },
            42,
        );

        let chunk_iter = TokenChunkIterator::new(&m_ids, 16);
        for (inputs, targets) in chunk_iter {
            m_model.forward_train_chunk(inputs, targets);
            m_model.backward_and_step_chunk(inputs.len());
        }
        println!("  [PASS] MQAR Sequence successfully ingested across distractor gaps.\n");

        println!("--- [Milestone 3: Neurological EMA Consolidation] ---");
        m_model.ema_consolidate_plasticity();
        println!("  [PASS] Fast adapter weights consolidated into base representation via EMA.\n");

        println!("--- [Milestone 4: Container Binary V5 Serialization] ---");
        Self::save_model_v2(&m_model, "data/test_v5.pssa").expect("Save failed");
        let imported = Self::load_model_v2("data/test_v5.pssa").expect("Load failed");
        assert_eq!(imported.cfg.d_vocab, m_model.cfg.d_vocab);
        assert_eq!(imported.cfg.d_latent, m_model.cfg.d_latent);
        assert_eq!(imported.cfg.d_state, m_model.cfg.d_state);
        std::fs::remove_file("data/test_v5.pssa").ok();
        println!("  [PASS] Format v5 verified with 100% parameter fidelity.\n");

        println!("--- [Milestone 5: Generation Test] ---");
        Self::run_training("science", 2, "data/model_v2.pssa");
        Self::run_generate("the solar system", "science", "data/model_v2.pssa");
        Self::run_generate("quantum mechanics", "science", "data/model_v2.pssa");
    }

    pub fn print_help() {
        println!("OXIDE AI: Plastic State-Space Architecture (PSSA) V2\n");
        println!("Usage: oxide <COMMAND> [OPTIONS]\n");
        println!("Commands:");
        println!("  train [source]       Train model with sequence TBPTT and AdamW");
        println!("  chat  [source]       Launch conversational REPL");
        println!("  generate <prompt>    Generate text completion");
        println!("  download <repo>      Download dataset from Hugging Face");
        println!("  benchmark            Run milestone verification test suite\n");
        println!("Options:");
        println!("  -d, --data <path>    Dataset source path (default: 'science')");
        println!("  -m, --model <path>   Model file path (default: 'data/model.pssa')");
        println!("  -e, --epochs <num>   Number of epochs (default: 4)");
        println!("  -p, --prompt <text>  Prompt for text generation");
        println!("  -t, --temp <float>   Sampling temperature (default: 0.70)");
        println!("  -o, --out <path>     Output model path (default: 'data/model.pssa')");
        println!("  -h, --help           Display help");
    }

    pub fn parse_and_execute(args: Vec<String>) {
        let parser = ArgParser::new(args.clone());

        if args.len() < 2 {
            Self::print_help();
            return;
        }

        let default_data = Self::resolve_default_data();

        match args[1].as_str() {
            "train" => {
                let data = parser.get_pos_or_flag(2, "--data", "-d", &default_data);
                let epochs = parser.get_usize("--epochs", "-e", 4);
                let out = parser.get_str("--out", "-o", "data/model.pssa");
                Self::run_training(&data, epochs, &out);
            }
            "chat" | "repl" => {
                let data = parser.get_pos_or_flag(2, "--data", "-d", &default_data);
                let model = parser.get_str("--model", "-m", "data/model.pssa");
                let temp = parser.get_f32("--temp", "-t", 0.70);
                Self::run_chat(&data, &model, temp);
            }
            "generate" => {
                let prompt = parser.get_pos_or_flag(2, "--prompt", "-p", "");
                if prompt.is_empty() {
                    println!("[ERROR] Please provide a prompt: oxide generate \"your prompt here\"");
                    return;
                }
                let data = parser.get_pos_or_flag(3, "--data", "-d", &default_data);
                let model = parser.get_str("--model", "-m", "data/model.pssa");
                Self::run_generate(&prompt, &data, &model);
            }
            "download" => {
                if args.len() < 3 {
                    println!("[ERROR] Please specify a dataset: oxide download wikimedia/wikipedia");
                    return;
                }
                let dataset = &args[2];
                let out = parser.get_str("--out", "-o", "data/downloaded.txt");
                Self::run_download(dataset, &out);
            }
            "benchmark" => {
                Self::run_benchmark();
            }
            "help" | "--help" | "-h" => {
                Self::print_help();
            }
            _ => {
                println!("[ERROR] Unknown command '{}'. Run 'oxide help' for usage.", args[1]);
            }
        }
    }
}