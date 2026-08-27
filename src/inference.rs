use crate::pssa::PSSALayerV2;
use crate::dataset::Tokenizer;
use crate::linalg::SimpleRng;
use std::f32;

pub struct InferenceConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub max_new_tokens: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.70,
            top_p: 0.85,
            top_k: 24,
            repetition_penalty: 1.25,
            max_new_tokens: 64,
        }
    }
}

pub struct PSSAInferenceEngine<'a> {
    model: &'a mut PSSALayerV2,
    tokenizer: &'a Tokenizer,
    rng: SimpleRng,
}

impl<'a> PSSAInferenceEngine<'a> {
    pub fn new(model: &'a mut PSSALayerV2, tokenizer: &'a Tokenizer) -> Self {
        Self {
            model,
            tokenizer,
            rng: SimpleRng::new(1337),
        }
    }

    pub fn generate_chat_turn<F>(&mut self, prompt: &str, cfg: &InferenceConfig, mut callback: F) -> String
    where
        F: FnMut(&str),
    {
        let ids = self.tokenizer.encode(prompt, true);
        if ids.is_empty() {
            return String::new();
        }

        let d_v = self.model.cfg.d_vocab;
        self.model.reset_recurrent_state();

        let mut logits_buf = vec![0.0f32; d_v];
        let mut probs_buf = vec![0.0f32; d_v];

        // 1. Ingest Prompt Context Token-by-Token into Multi-Channel Recurrent State
        for &in_id in &ids {
            let clamped_id = in_id % d_v;
            self.model.forward_inference(clamped_id, &mut logits_buf);
        }

        let mut generated_ids = ids.clone();
        let mut out_text = String::new();
        let mut sentence_count = 0;

        // 2. Autoregressive Generation Loop
        for step_i in 0..cfg.max_new_tokens {
            if step_i > 0 {
                let last_id = *generated_ids.last().unwrap_or(&0) % d_v;
                self.model.forward_inference(last_id, &mut logits_buf);
            }

            // Hard ban <unk> token (ID 0)
            if !logits_buf.is_empty() {
                logits_buf[0] = -1e4;
            }

            // Finiteness validation
            for val in logits_buf.iter_mut() {
                if !val.is_finite() {
                    *val = -1e4;
                }
            }

            // Apply Windowed Repetition Penalty
            let penalty = cfg.repetition_penalty.max(1.0);
            let window_start = generated_ids.len().saturating_sub(64);
            for &prev_id in &generated_ids[window_start..] {
                if prev_id < logits_buf.len() {
                    if logits_buf[prev_id] > 0.0 {
                        logits_buf[prev_id] /= penalty;
                    } else {
                        logits_buf[prev_id] *= penalty;
                    }
                }
            }

            // Suppress direct immediate self-transition loops
            if let Some(&last_id) = generated_ids.last() {
                if last_id < logits_buf.len() {
                    logits_buf[last_id] -= 2.0;
                }
            }

            // Temperature Scaling
            let temp = cfg.temperature.max(0.01);
            for val in logits_buf.iter_mut() {
                *val /= temp;
            }

            // Numerically Stable Softmax
            let mut max_l = f32::NEG_INFINITY;
            for &l in &logits_buf {
                if l > max_l {
                    max_l = l;
                }
            }

            let mut sum_exp = 0.0f32;
            for i in 0..d_v {
                let exp_val = (logits_buf[i] - max_l).exp();
                probs_buf[i] = exp_val;
                sum_exp += exp_val;
            }
            let inv_sum = 1.0 / sum_exp.max(1e-8);
            for i in 0..d_v {
                probs_buf[i] *= inv_sum;
            }

            // Top-K + Top-P (Nucleus) Filtering
            let mut candidates: Vec<(usize, f32)> = (0..d_v)
                .filter(|&i| i != 0 && probs_buf[i].is_finite())
                .map(|i| (i, probs_buf[i]))
                .collect();

            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_k_len = candidates.len().min(cfg.top_k.max(1));
            let top_candidates = &candidates[..top_k_len];

            let mut cum_sum = 0.0;
            let mut cutoff_idx = top_candidates.len();
            let top_p = cfg.top_p.clamp(0.01, 1.0);

            for (idx, &(_, p)) in top_candidates.iter().enumerate() {
                cum_sum += p;
                if cum_sum >= top_p {
                    cutoff_idx = idx + 1;
                    break;
                }
            }

            let filtered = &top_candidates[..cutoff_idx.max(1)];
            let total_filtered_prob: f32 = filtered.iter().map(|(_, p)| p).sum();
            let rand_val = self.rng.gen_range_f32(0.0, total_filtered_prob.max(1e-6));

            let mut running = 0.0;
            let mut selected_id = filtered[0].0;
            for &(id, p) in filtered {
                running += p;
                if running >= rand_val {
                    selected_id = id;
                    break;
                }
            }

            generated_ids.push(selected_id);

            let token_str = if let Some(word) = self.tokenizer.id_to_token.get(&selected_id) {
                word.clone()
            } else {
                format!("[#{}]", selected_id)
            };

            callback(&token_str);
            out_text.push_str(&token_str);
            out_text.push(' ');

            if token_str == "." {
                sentence_count += 1;
                if sentence_count >= 2 {
                    break;
                }
            }
        }

        out_text.trim().to_string()
    }
}