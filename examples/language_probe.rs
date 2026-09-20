//! Reproducible BPE diagnostics; metrics and samples are not proof of coherence.
use oxide_ai_pssa::checkpoint;
use oxide_ai_pssa::cli::CLIHandler;
use oxide_ai_pssa::dataset::Tokenizer;
use oxide_ai_pssa::inference::{InferenceConfig, PSSAInferenceEngine};
use serde_json::{Value, json};
use std::fs;

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 || !matches!(args[1].as_str(), "evaluate" | "generate") {
        return Err(
            "usage: language_probe evaluate MODEL DATA | generate MODEL PROMPTS_JSON".into(),
        );
    }
    let mut model = checkpoint::load_checkpoint(&args[2])
        .map_err(|e| e.to_string())?
        .model;
    let tokenizer = Tokenizer::from_serialized(
        model
            .tokenizer_json
            .as_deref()
            .ok_or("language_probe requires a self-contained BPE checkpoint")?,
    )?;
    let input = fs::read_to_string(&args[3]).map_err(|e| e.to_string())?;
    let result = if args[1] == "evaluate" {
        let (ce, transitions, correct, oov) =
            CLIHandler::evaluate_corpus(&mut model, &tokenizer, &input)?;
        let mut encoded = 0usize;
        let mut documents = 0usize;
        for line in input.lines() {
            let ids = tokenizer.try_encode(line, true)?;
            encoded += ids.len();
            documents += usize::from(!ids.is_empty());
        }
        json!({
            "cross_entropy_nats_per_bpe_transition": ce,
            "perplexity_per_bpe_transition": if ce.exp().is_finite() { Some(ce.exp()) } else { None },
            "perplexity_overflow": !ce.exp().is_finite(),
            "scored_transitions": transitions,
            "encoded_input_tokens": encoded,
            "documents": documents,
            "oov_count": oov,
            "oov_rate_encoded_input": oov as f64 / encoded.max(1) as f64,
            "next_token_accuracy": correct as f64 / transitions.max(1) as f64,
            "vocabulary_size": tokenizer.vocab_size,
            "optimizer_updates": model.step_counter,
            "evaluation": "teacher-forced; recurrent reset per document; frozen training memory retained; no optimizer updates"
        })
    } else {
        let plan: Value = serde_json::from_str(&input).map_err(|e| e.to_string())?;
        if plan["sampling"]["seed"].as_u64() != Some(1337) {
            return Err(
                "the current inference engine uses fixed seed 1337; prompt plan must match".into(),
            );
        }
        let max_new_tokens = plan["sampling"]["max_new_tokens"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("missing max_new_tokens")?;
        let prompts = plan["prompts"].as_array().ok_or("missing prompts")?;
        let temperatures = plan["sampling"]["temperatures"]
            .as_array()
            .ok_or("missing temperatures")?;
        let mut samples = Vec::new();
        for prompt in prompts {
            let text = prompt["text"].as_str().ok_or("missing prompt text")?;
            for temp in temperatures {
                let temperature = temp.as_f64().ok_or("invalid temperature")? as f32;
                let config = InferenceConfig {
                    temperature,
                    top_p: 1.0,
                    top_k: tokenizer.vocab_size,
                    repetition_penalty: 1.0,
                    max_new_tokens,
                };
                // A fresh engine resets the sampling seed to 1337 for every sample.
                // Generation resets recurrent state and does not modify the memory bank.
                let completion = PSSAInferenceEngine::try_new(&mut model, &tokenizer)?
                    .try_generate_chat_turn(text, &config, |_| {})?;
                samples
                    .push(json!({"prompt": text, "temperature": temp, "completion": completion}));
            }
        }
        json!({
            "sampling": {"seed_per_sample": 1337, "top_p": 1.0, "top_k": tokenizer.vocab_size,
                "repetition_penalty": 1.0, "max_new_tokens": max_new_tokens,
                "token_policy": "existing inference engine excludes token 0 (<unk>); no other candidate truncation"},
            "selection": "all prompts and temperatures retained in input order, no rerolls",
            "decoding_limit": "existing streaming decoder can omit an incomplete final UTF-8 suffix at the token cutoff",
            "samples": samples
        })
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
