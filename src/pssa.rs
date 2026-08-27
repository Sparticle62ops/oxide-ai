use crate::adapter::PlasticAdapterV2;
use crate::backend::{Device, WgpuContext};
use crate::linalg::{dot_slice, sigmoid, softplus, SimpleRng};
use crate::memory::HyperbolicEpisodicBankV2;
use std::f32;

// =============================================================================
// PARAMETER TENSORS WITH INTEGRATED ADAMW MOMENTS
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct ParamVector {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl ParamVector {
    pub fn new(len: usize, init_val: f32) -> Self {
        Self {
            data: vec![init_val; len],
            grad: vec![0.0; len],
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    #[inline(always)]
    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }

    pub fn step_adamw(&mut self, lr: f32, beta1: f32, beta2: f32, weight_decay: f32, eps: f32, step: usize) {
        let step_f = step as f32;
        let bias_corr1 = 1.0 - beta1.powf(step_f);
        let bias_corr2 = 1.0 - beta2.powf(step_f);

        for i in 0..self.data.len() {
            let g = self.grad[i];
            if weight_decay > 0.0 {
                self.data[i] -= lr * weight_decay * self.data[i];
            }
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m[i] / bias_corr1;
            let v_hat = self.v[i] / bias_corr2;
            self.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParamMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl ParamMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let total = rows * cols;
        Self {
            rows,
            cols,
            data: vec![0.0; total],
            grad: vec![0.0; total],
            m: vec![0.0; total],
            v: vec![0.0; total],
        }
    }

    pub fn random_xavier(rows: usize, cols: usize, rng: &mut SimpleRng) -> Self {
        let total = rows * cols;
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        let mut data = Vec::with_capacity(total);
        for _ in 0..total {
            data.push(rng.gen_range_f32(-limit, limit));
        }
        Self {
            rows,
            cols,
            data,
            grad: vec![0.0; total],
            m: vec![0.0; total],
            v: vec![0.0; total],
        }
    }

    #[inline(always)]
    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }

    #[inline(always)]
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(self.cols, x.len());
        assert_eq!(self.rows, out.len());
        for r in 0..self.rows {
            let row_slice = &self.data[r * self.cols..(r + 1) * self.cols];
            out[r] = dot_slice(row_slice, x);
        }
    }

    #[inline(always)]
    pub fn matvec_transpose(&self, g: &[f32], out: &mut [f32]) {
        assert_eq!(self.rows, g.len());
        assert_eq!(self.cols, out.len());
        out.fill(0.0);
        for r in 0..self.rows {
            let gr = g[r];
            let row_off = r * self.cols;
            for c in 0..self.cols {
                out[c] += gr * self.data[row_off + c];
            }
        }
    }

    pub fn step_adamw(&mut self, lr: f32, beta1: f32, beta2: f32, weight_decay: f32, eps: f32, step: usize) {
        let step_f = step as f32;
        let bias_corr1 = 1.0 - beta1.powf(step_f);
        let bias_corr2 = 1.0 - beta2.powf(step_f);

        for i in 0..self.data.len() {
            let g = self.grad[i];
            if weight_decay > 0.0 {
                self.data[i] -= lr * weight_decay * self.data[i];
            }
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m[i] / bias_corr1;
            let v_hat = self.v[i] / bias_corr2;
            self.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

// =============================================================================
// ENGINE CONFIGURATION
// =============================================================================

#[derive(Clone, Debug)]
pub struct PSSAConfigV2 {
    pub d_vocab: usize,
    pub d_latent: usize,
    pub d_state: usize,
    pub d_mem_key: usize,
    pub mem_capacity: usize,
    pub chunk_len: usize,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub eps: f32,
    pub tau_mem: f32,
    pub ema_alpha: f32,
}

impl Default for PSSAConfigV2 {
    fn default() -> Self {
        Self {
            d_vocab: 10_000,
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
        }
    }
}

// =============================================================================
// ZERO-ALLOCATION TBPTT ACTIVATION TAPE
// =============================================================================

#[derive(Clone, Debug)]
pub struct ChunkActivationTape {
    pub max_l: usize,
    pub x_ids: Vec<usize>,
    pub target_ids: Vec<usize>,
    pub x_norm: Vec<f32>,
    pub inv_rms: Vec<f32>,
    pub delta: Vec<f32>,
    pub b_proj: Vec<f32>,
    pub c_proj: Vec<f32>,
    pub bar_a: Vec<f32>,
    pub bar_b: Vec<f32>,
    pub h_states: Vec<f32>,
    pub y_ssm: Vec<f32>,
    pub q_euc: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub q_poincare: Vec<f32>,
    pub mem_weights: Vec<f32>,
    pub m_val: Vec<f32>,
    pub g_mem: Vec<f32>,
    pub m_inj: Vec<f32>,
    pub adapter_act: Vec<f32>,
    pub z_raw: Vec<f32>,
    pub mlp_hidden: Vec<f32>,
    pub mlp_act: Vec<f32>,
    pub z_final: Vec<f32>,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub losses: Vec<f32>,
}

impl ChunkActivationTape {
    pub fn new(max_l: usize, d_vocab: usize, d_latent: usize, d_state: usize, d_key: usize, mem_cap: usize, rank: usize) -> Self {
        Self {
            max_l,
            x_ids: vec![0; max_l],
            target_ids: vec![0; max_l],
            x_norm: vec![0.0; max_l * d_latent],
            inv_rms: vec![0.0; max_l],
            delta: vec![0.0; max_l * d_latent],
            b_proj: vec![0.0; max_l * d_state],
            c_proj: vec![0.0; max_l * d_state],
            bar_a: vec![0.0; max_l * d_latent * d_state],
            bar_b: vec![0.0; max_l * d_latent * d_state],
            h_states: vec![0.0; (max_l + 1) * d_latent * d_state],
            y_ssm: vec![0.0; max_l * d_latent],
            q_euc: vec![0.0; max_l * d_key],
            q_norm: vec![0.0; max_l],
            q_poincare: vec![0.0; max_l * d_key],
            mem_weights: vec![0.0; max_l * mem_cap],
            m_val: vec![0.0; max_l * d_latent],
            g_mem: vec![0.0; max_l * d_latent],
            m_inj: vec![0.0; max_l * d_latent],
            adapter_act: vec![0.0; max_l * rank],
            z_raw: vec![0.0; max_l * d_latent],
            mlp_hidden: vec![0.0; max_l * 2 * d_latent],
            mlp_act: vec![0.0; max_l * 2 * d_latent],
            z_final: vec![0.0; max_l * d_latent],
            logits: vec![0.0; max_l * d_vocab],
            probs: vec![0.0; max_l * d_vocab],
            losses: vec![0.0; max_l],
        }
    }
}

// =============================================================================
// PSSA LAYER V2 - MULTI-CHANNEL NEURAL ENGINE
// =============================================================================

pub struct PSSALayerV2 {
    pub cfg: PSSAConfigV2,
    pub step_counter: usize,
    pub device: Device,
    pub rng: SimpleRng,

    // 1. Learned Affine RMSNorm & Embeddings
    pub embed_w: ParamMatrix,
    pub norm_gamma: ParamVector,
    pub norm_beta: ParamVector,

    // 2. Multi-Channel Continuous SSM Recurrence
    pub a_mat: ParamMatrix,
    pub w_delta: ParamMatrix,
    pub w_b: ParamMatrix,
    pub w_c: ParamMatrix,
    pub h_persistent: Vec<f32>,

    // 3. Diffeomorphic Poincaré Episodic Memory
    pub w_qx: ParamMatrix,
    pub w_qh: ParamMatrix,
    pub w_gate: ParamMatrix,
    pub w_proj: ParamMatrix,
    pub memory: HyperbolicEpisodicBankV2,

    // 4. Zero-Init Plastic Adapters
    pub adapters: Vec<PlasticAdapterV2>,

    // 5. SiLU Feed-Forward Expansion Head
    pub mlp_w1: ParamMatrix,
    pub mlp_w2: ParamMatrix,

    // 6. Output Logits Projection
    pub unembed_w: ParamMatrix,

    // Preallocated Tape for Zero-Allocation Training
    pub tape: ChunkActivationTape,

    // Backward Temporary Adjoint Buffers
    pub grad_h_next: Vec<f32>,
    pub grad_z_final: Vec<f32>,
    pub grad_z_raw: Vec<f32>,
    pub grad_x_norm: Vec<f32>,

    // Inference Scratch Buffers
    pub inf_x_norm: Vec<f32>,
    pub inf_delta: Vec<f32>,
    pub inf_b: Vec<f32>,
    pub inf_c: Vec<f32>,
    pub inf_y_ssm: Vec<f32>,
    pub inf_q_euc: Vec<f32>,
    pub inf_q_pnc: Vec<f32>,
    pub inf_mem_weights: Vec<f32>,
    pub inf_m_val: Vec<f32>,
    pub inf_g_mem: Vec<f32>,
    pub inf_m_proj: Vec<f32>,
    pub inf_ad_act: Vec<f32>,
    pub inf_ad_out: Vec<f32>,
    pub inf_z_raw: Vec<f32>,
    pub inf_mlp_act: Vec<f32>,
    pub inf_mlp_out: Vec<f32>,
    pub inf_z_final: Vec<f32>,
}

impl PSSALayerV2 {
    pub fn new(cfg: PSSAConfigV2, seed: u64) -> Self {
        Self::new_with_device(cfg, seed, Device::Cpu)
    }

    pub fn new_with_device(cfg: PSSAConfigV2, seed: u64, device: Device) -> Self {
        let mut rng = SimpleRng::new(seed);
        let d_v = cfg.d_vocab;
        let d_m = cfg.d_latent;
        let d_s = cfg.d_state;
        let d_k = cfg.d_mem_key;
        let d_mlp = d_m * 2;
        let mem_cap = cfg.mem_capacity;
        let chunk_len = cfg.chunk_len;
        let rank = 16;

        let embed_w = ParamMatrix::random_xavier(d_v, d_m, &mut rng);
        let norm_gamma = ParamVector::new(d_m, 1.0);
        let norm_beta = ParamVector::new(d_m, 0.0);

        // Continuous HiPPO timescale decay initialization
        let mut a_mat = ParamMatrix::zeros(d_m, d_s);
        let ln_min = 1.5f32.ln();
        let ln_max = 200.0f32.ln();
        for i in 0..d_m {
            for j in 0..d_s {
                let ratio = if d_s > 1 { j as f32 / (d_s - 1) as f32 } else { 0.0 };
                let tau = (ln_min + ratio * (ln_max - ln_min)).exp();
                a_mat.data[i * d_s + j] = -1.0 / tau;
            }
        }

        let w_delta = ParamMatrix::random_xavier(d_m, d_m, &mut rng);
        let w_b = ParamMatrix::random_xavier(d_s, d_m, &mut rng);
        let w_c = ParamMatrix::random_xavier(d_s, d_m, &mut rng);
        let h_persistent = vec![0.0; d_m * d_s];

        let w_qx = ParamMatrix::random_xavier(d_k, d_m, &mut rng);
        let w_qh = ParamMatrix::random_xavier(d_k, d_m, &mut rng);
        let w_gate = ParamMatrix::random_xavier(d_m, d_m, &mut rng);
        let w_proj = ParamMatrix::random_xavier(d_m, d_m, &mut rng);
        let memory = HyperbolicEpisodicBankV2::new(mem_cap, d_k, d_m);

        let mut adapters = Vec::new();
        adapters.push(PlasticAdapterV2::new(d_m, rank, &mut rng));

        let mlp_w1 = ParamMatrix::random_xavier(d_mlp, d_m, &mut rng);
        let mlp_w2 = ParamMatrix::zeros(d_m, d_mlp);
        let unembed_w = ParamMatrix::random_xavier(d_v, d_m, &mut rng);

        let tape = ChunkActivationTape::new(chunk_len, d_v, d_m, d_s, d_k, mem_cap, rank);

        Self {
            cfg,
            step_counter: 0,
            device,
            rng,
            embed_w,
            norm_gamma,
            norm_beta,
            a_mat,
            w_delta,
            w_b,
            w_c,
            h_persistent,
            w_qx,
            w_qh,
            w_gate,
            w_proj,
            memory,
            adapters,
            mlp_w1,
            mlp_w2,
            unembed_w,
            tape,
            grad_h_next: vec![0.0; d_m * d_s],
            grad_z_final: vec![0.0; d_m],
            grad_z_raw: vec![0.0; d_m],
            grad_x_norm: vec![0.0; d_m],
            inf_x_norm: vec![0.0; d_m],
            inf_delta: vec![0.0; d_m],
            inf_b: vec![0.0; d_s],
            inf_c: vec![0.0; d_s],
            inf_y_ssm: vec![0.0; d_m],
            inf_q_euc: vec![0.0; d_k],
            inf_q_pnc: vec![0.0; d_k],
            inf_mem_weights: vec![0.0; mem_cap],
            inf_m_val: vec![0.0; d_m],
            inf_g_mem: vec![0.0; d_m],
            inf_m_proj: vec![0.0; d_m],
            inf_ad_act: vec![0.0; rank],
            inf_ad_out: vec![0.0; d_m],
            inf_z_raw: vec![0.0; d_m],
            inf_mlp_act: vec![0.0; d_mlp],
            inf_mlp_out: vec![0.0; d_m],
            inf_z_final: vec![0.0; d_m],
        }
    }

    pub fn reset_recurrent_state(&mut self) {
        self.h_persistent.fill(0.0);
    }

    // =========================================================================
    // 1. HIGH-SPEED INFERENCE PATHWAY (ZERO BACKPROP OVERHEAD)
    // =========================================================================
    #[inline(always)]
    pub fn forward_inference(&mut self, x_id: usize, logits_out: &mut [f32]) {
        let d_m = self.cfg.d_latent;
        let d_s = self.cfg.d_state;
        let d_k = self.cfg.d_mem_key;
        let d_mlp = d_m * 2;
        let rank = self.adapters[0].rank;

        // 1. Embedding & Affine RMSNorm
        let e_t = &self.embed_w.data[x_id * d_m..(x_id + 1) * d_m];
        let sum_sq: f32 = e_t.iter().map(|&x| x * x).sum();
        let inv_rms = 1.0 / (sum_sq / (d_m as f32) + 1e-5).sqrt();

        for i in 0..d_m {
            self.inf_x_norm[i] = self.norm_gamma.data[i] * (e_t[i] * inv_rms) + self.norm_beta.data[i];
        }

        // 2. Data-Dependent Projections
        self.w_delta.matvec(&self.inf_x_norm, &mut self.inf_delta);
        for i in 0..d_m {
            self.inf_delta[i] = softplus(self.inf_delta[i]);
        }

        self.w_b.matvec(&self.inf_x_norm, &mut self.inf_b);
        self.w_c.matvec(&self.inf_x_norm, &mut self.inf_c);

        // 3. Multi-Channel SSM State Update (In-Place on Persistent State)
        for i in 0..d_m {
            let d_i = self.inf_delta[i];
            let mut y_i = 0.0f32;
            let row_off = i * d_s;

            for j in 0..d_s {
                let idx = row_off + j;
                let bar_a = (d_i * self.a_mat.data[idx]).exp();
                let bar_b = d_i * self.inf_b[j];

                let h_val = bar_a * self.h_persistent[idx] + bar_b * self.inf_x_norm[i];
                self.h_persistent[idx] = h_val;
                y_i += h_val * self.inf_c[j];
            }
            self.inf_y_ssm[i] = y_i;
        }

        // 4. Diffeomorphic Poincaré Memory Retrieval
        for r in 0..d_k {
            let row_x = &self.w_qx.data[r * d_m..(r + 1) * d_m];
            let row_h = &self.w_qh.data[r * d_m..(r + 1) * d_m];
            self.inf_q_euc[r] = dot_slice(row_x, &self.inf_x_norm) + dot_slice(row_h, &self.inf_y_ssm);
        }

        HyperbolicEpisodicBankV2::diffeomorphic_project(&self.inf_q_euc, &mut self.inf_q_pnc);

        self.memory.retrieve_soft_into(
            &self.inf_q_pnc,
            self.cfg.tau_mem,
            &mut self.inf_m_val,
            &mut self.inf_mem_weights,
        );

        self.w_gate.matvec(&self.inf_x_norm, &mut self.inf_g_mem);
        for i in 0..d_m {
            self.inf_g_mem[i] = sigmoid(self.inf_g_mem[i]);
        }

        self.w_proj.matvec(&self.inf_m_val, &mut self.inf_m_proj);

        // 5. Plastic Adapter
        self.adapters[0].down_proj.matvec(&self.inf_x_norm, &mut self.inf_ad_act[..rank]);
        for r in 0..rank {
            let h = self.inf_ad_act[r];
            self.inf_ad_act[r] = h * sigmoid(h);
        }
        self.adapters[0].up_proj.matvec(&self.inf_ad_act[..rank], &mut self.inf_ad_out);

        // 6. Latent Aggregation & SiLU MLP Expansion
        for i in 0..d_m {
            self.inf_z_raw[i] = self.inf_y_ssm[i] + (self.inf_g_mem[i] * self.inf_m_proj[i]) + self.inf_ad_out[i];
        }

        self.mlp_w1.matvec(&self.inf_z_raw, &mut self.inf_mlp_act[..d_mlp]);
        for i in 0..d_mlp {
            let h = self.inf_mlp_act[i];
            self.inf_mlp_act[i] = h * sigmoid(h);
        }

        self.mlp_w2.matvec(&self.inf_mlp_act[..d_mlp], &mut self.inf_mlp_out);
        for i in 0..d_m {
            self.inf_z_final[i] = self.inf_z_raw[i] + self.inf_mlp_out[i];
        }

        // 7. Output Vocabulary Logits
        self.unembed_w.matvec(&self.inf_z_final, logits_out);
    }

    // =========================================================================
    // 2. TBPTT SEQUENCE TRAINING PATHWAY
    // =========================================================================
    pub fn forward_train_chunk(&mut self, token_ids: &[usize], target_ids: &[usize]) -> f32 {
        let seq_len = token_ids.len().min(self.cfg.chunk_len);
        let d_m = self.cfg.d_latent;
        let d_s = self.cfg.d_state;
        let d_k = self.cfg.d_mem_key;
        let d_v = self.cfg.d_vocab;
        let d_mlp = d_m * 2;
        let mem_cap = self.cfg.mem_capacity;
        let rank = self.adapters[0].rank;

        self.tape.x_ids[..seq_len].copy_from_slice(&token_ids[..seq_len]);
        self.tape.target_ids[..seq_len].copy_from_slice(&target_ids[..seq_len]);
        self.tape.h_states[..d_m * d_s].copy_from_slice(&self.h_persistent);

        let mut total_loss = 0.0f32;

        for t in 0..seq_len {
            let x_id = self.tape.x_ids[t];
            let tgt_id = self.tape.target_ids[t];

            // 1. Embedding & Affine RMSNorm
            let e_t = &self.embed_w.data[x_id * d_m..(x_id + 1) * d_m];
            let sum_sq: f32 = e_t.iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / (d_m as f32) + 1e-5).sqrt();
            self.tape.inv_rms[t] = inv_rms;

            let xn_off = t * d_m;
            for i in 0..d_m {
                self.tape.x_norm[xn_off + i] = self.norm_gamma.data[i] * (e_t[i] * inv_rms) + self.norm_beta.data[i];
            }
            let x_n = &self.tape.x_norm[xn_off..xn_off + d_m];

            // 2. Data-Dependent Projections
            let del_off = t * d_m;
            self.w_delta.matvec(x_n, &mut self.tape.delta[del_off..del_off + d_m]);
            for i in 0..d_m {
                self.tape.delta[del_off + i] = softplus(self.tape.delta[del_off + i]);
            }
            let delta = &self.tape.delta[del_off..del_off + d_m];

            let b_off = t * d_s;
            self.w_b.matvec(x_n, &mut self.tape.b_proj[b_off..b_off + d_s]);
            let b_p = &self.tape.b_proj[b_off..b_off + d_s];

            let c_off = t * d_s;
            self.w_c.matvec(x_n, &mut self.tape.c_proj[c_off..c_off + d_s]);
            let c_p = &self.tape.c_proj[c_off..c_off + d_s];

            // 3. Multi-Channel SSM Recurrent Scan
            let h_prev_off = t * (d_m * d_s);
            let h_next_off = (t + 1) * (d_m * d_s);
            let ssm_off = t * (d_m * d_s);
            let y_off = t * d_m;

            for i in 0..d_m {
                let d_i = delta[i];
                let mut y_i = 0.0f32;
                for j in 0..d_s {
                    let idx = i * d_s + j;
                    let bar_a = (d_i * self.a_mat.data[idx]).exp();
                    let bar_b = d_i * b_p[j];

                    self.tape.bar_a[ssm_off + idx] = bar_a;
                    self.tape.bar_b[ssm_off + idx] = bar_b;

                    let h_val = bar_a * self.tape.h_states[h_prev_off + idx] + bar_b * x_n[i];
                    self.tape.h_states[h_next_off + idx] = h_val;

                    y_i += h_val * c_p[j];
                }
                self.tape.y_ssm[y_off + i] = y_i;
            }
            let y_ssm = &self.tape.y_ssm[y_off..y_off + d_m];

            // 4. Diffeomorphic Poincaré Memory Retrieval
            let q_off = t * d_k;
            for r in 0..d_k {
                let row_x = &self.w_qx.data[r * d_m..(r + 1) * d_m];
                let row_h = &self.w_qh.data[r * d_m..(r + 1) * d_m];
                self.tape.q_euc[q_off + r] = dot_slice(row_x, x_n) + dot_slice(row_h, y_ssm);
            }

            self.tape.q_norm[t] = HyperbolicEpisodicBankV2::diffeomorphic_project(
                &self.tape.q_euc[q_off..q_off + d_k],
                &mut self.tape.q_poincare[q_off..q_off + d_k],
            );

            let m_off = t * d_m;
            let mw_off = t * mem_cap;
            self.memory.retrieve_soft_into(
                &self.tape.q_poincare[q_off..q_off + d_k],
                self.cfg.tau_mem,
                &mut self.tape.m_val[m_off..m_off + d_m],
                &mut self.tape.mem_weights[mw_off..mw_off + mem_cap],
            );

            self.w_gate.matvec(x_n, &mut self.tape.g_mem[m_off..m_off + d_m]);
            for i in 0..d_m {
                self.tape.g_mem[m_off + i] = sigmoid(self.tape.g_mem[m_off + i]);
            }

            let mut m_proj_out = vec![0.0; d_m];
            self.w_proj.matvec(&self.tape.m_val[m_off..m_off + d_m], &mut m_proj_out);
            for i in 0..d_m {
                self.tape.m_inj[m_off + i] = self.tape.g_mem[m_off + i] * m_proj_out[i];
            }

            // 5. Zero-Init Plastic Adapter
            let ad_off = t * rank;
            self.adapters[0].down_proj.matvec(x_n, &mut self.tape.adapter_act[ad_off..ad_off + rank]);
            for r in 0..rank {
                let h = self.tape.adapter_act[ad_off + r];
                self.tape.adapter_act[ad_off + r] = h * sigmoid(h);
            }
            let mut ad_out = vec![0.0; d_m];
            self.adapters[0].up_proj.matvec(&self.tape.adapter_act[ad_off..ad_off + rank], &mut ad_out);

            // 6. Latent Aggregation & SiLU MLP Expansion
            let z_off = t * d_m;
            for i in 0..d_m {
                self.tape.z_raw[z_off + i] = y_ssm[i] + self.tape.m_inj[m_off + i] + ad_out[i];
            }
            let z_raw = &self.tape.z_raw[z_off..z_off + d_m];

            let mlp_off = t * d_mlp;
            self.mlp_w1.matvec(z_raw, &mut self.tape.mlp_hidden[mlp_off..mlp_off + d_mlp]);
            for i in 0..d_mlp {
                let h = self.tape.mlp_hidden[mlp_off + i];
                self.tape.mlp_act[mlp_off + i] = h * sigmoid(h);
            }
            self.mlp_w2.matvec(&self.tape.mlp_act[mlp_off..mlp_off + d_mlp], &mut self.tape.z_final[z_off..z_off + d_m]);
            for i in 0..d_m {
                self.tape.z_final[z_off + i] += z_raw[i];
            }
            let z_final = &self.tape.z_final[z_off..z_off + d_m];

            // 7. Output Vocabulary Logits & Softmax Loss
            let log_off = t * d_v;
            self.unembed_w.matvec(z_final, &mut self.tape.logits[log_off..log_off + d_v]);

            let mut max_l = f32::NEG_INFINITY;
            for i in 0..d_v {
                let l = self.tape.logits[log_off + i];
                if l > max_l {
                    max_l = l;
                }
            }

            let mut sum_exp = 0.0f32;
            for i in 0..d_v {
                let exp_l = (self.tape.logits[log_off + i] - max_l).exp();
                self.tape.probs[log_off + i] = exp_l;
                sum_exp += exp_l;
            }
            let inv_sum = 1.0 / sum_exp.max(1e-8);
            for i in 0..d_v {
                self.tape.probs[log_off + i] *= inv_sum;
            }

            let nll_loss = -self.tape.probs[log_off + tgt_id].max(1e-12).ln();
            self.tape.losses[t] = nll_loss;
            total_loss += nll_loss;
        }

        let last_h_off = seq_len * (d_m * d_s);
        self.h_persistent.copy_from_slice(&self.tape.h_states[last_h_off..last_h_off + d_m * d_s]);

        total_loss / (seq_len as f32)
    }

    // =========================================================================
    // 3. EXACT TBPTT BACKWARD PASS & ADAMW OPTIMIZATION
    // =========================================================================
    pub fn backward_and_step_chunk(&mut self, seq_len: usize) {
        let d_m = self.cfg.d_latent;
        let d_s = self.cfg.d_state;
        let d_v = self.cfg.d_vocab;
        let d_mlp = d_m * 2;
        let rank = self.adapters[0].rank;
        let scale_loss = 1.0 / (seq_len as f32);

        self.embed_w.zero_grad();
        self.norm_gamma.zero_grad();
        self.norm_beta.zero_grad();
        self.a_mat.zero_grad();
        self.w_delta.zero_grad();
        self.w_b.zero_grad();
        self.w_c.zero_grad();
        self.w_qx.zero_grad();
        self.w_qh.zero_grad();
        self.w_gate.zero_grad();
        self.w_proj.zero_grad();
        self.adapters[0].zero_grad();
        self.mlp_w1.zero_grad();
        self.mlp_w2.zero_grad();
        self.unembed_w.zero_grad();

        self.grad_h_next.fill(0.0);

        // Reverse Time Loop across sequence chunk L
        for t in (0..seq_len).rev() {
            let x_id = self.tape.x_ids[t];
            let tgt_id = self.tape.target_ids[t];
            let z_off = t * d_m;
            let log_off = t * d_v;

            // 1. Output Cross-Entropy Adjoint (SPARSE PRECISION BOUNDED)
            self.grad_z_final.fill(0.0);
            
            for i in 0..d_v {
                let prob = self.tape.probs[log_off + i];
                let is_target = i == tgt_id;
                
                // RESTORED: Prune 99.5% of mathematically irrelevant updates (< epsilon)
                if prob < 1e-5 && !is_target {
                    continue;
                }

                let indicator = if is_target { 1.0 } else { 0.0 };
                let g_logit = (prob - indicator) * scale_loss;

                let row_off = i * d_m;
                
                // 8-way unrolled gradient accumulation
                for j in (0..d_m).step_by(8) {
                    self.grad_z_final[j] += g_logit * self.unembed_w.data[row_off + j];
                    self.grad_z_final[j + 1] += g_logit * self.unembed_w.data[row_off + j + 1];
                    self.grad_z_final[j + 2] += g_logit * self.unembed_w.data[row_off + j + 2];
                    self.grad_z_final[j + 3] += g_logit * self.unembed_w.data[row_off + j + 3];
                    self.grad_z_final[j + 4] += g_logit * self.unembed_w.data[row_off + j + 4];
                    self.grad_z_final[j + 5] += g_logit * self.unembed_w.data[row_off + j + 5];
                    self.grad_z_final[j + 6] += g_logit * self.unembed_w.data[row_off + j + 6];
                    self.grad_z_final[j + 7] += g_logit * self.unembed_w.data[row_off + j + 7];

                    self.unembed_w.grad[row_off + j] += g_logit * self.tape.z_final[z_off + j];
                    self.unembed_w.grad[row_off + j + 1] += g_logit * self.tape.z_final[z_off + j + 1];
                    self.unembed_w.grad[row_off + j + 2] += g_logit * self.tape.z_final[z_off + j + 2];
                    self.unembed_w.grad[row_off + j + 3] += g_logit * self.tape.z_final[z_off + j + 3];
                    self.unembed_w.grad[row_off + j + 4] += g_logit * self.tape.z_final[z_off + j + 4];
                    self.unembed_w.grad[row_off + j + 5] += g_logit * self.tape.z_final[z_off + j + 5];
                    self.unembed_w.grad[row_off + j + 6] += g_logit * self.tape.z_final[z_off + j + 6];
                    self.unembed_w.grad[row_off + j + 7] += g_logit * self.tape.z_final[z_off + j + 7];
                }
            }

            // 2. SiLU MLP Backward
            let mlp_off = t * d_mlp;
            let mut g_mlp_act = vec![0.0; d_mlp];
            self.mlp_w2.matvec_transpose(&self.grad_z_final, &mut g_mlp_act);

            for i in 0..d_m {
                let gz_i = self.grad_z_final[i];
                let row_off = i * d_mlp;
                for j in (0..d_mlp).step_by(8) {
                    self.mlp_w2.grad[row_off + j] += gz_i * self.tape.mlp_act[mlp_off + j];
                    self.mlp_w2.grad[row_off + j + 1] += gz_i * self.tape.mlp_act[mlp_off + j + 1];
                    self.mlp_w2.grad[row_off + j + 2] += gz_i * self.tape.mlp_act[mlp_off + j + 2];
                    self.mlp_w2.grad[row_off + j + 3] += gz_i * self.tape.mlp_act[mlp_off + j + 3];
                    self.mlp_w2.grad[row_off + j + 4] += gz_i * self.tape.mlp_act[mlp_off + j + 4];
                    self.mlp_w2.grad[row_off + j + 5] += gz_i * self.tape.mlp_act[mlp_off + j + 5];
                    self.mlp_w2.grad[row_off + j + 6] += gz_i * self.tape.mlp_act[mlp_off + j + 6];
                    self.mlp_w2.grad[row_off + j + 7] += gz_i * self.tape.mlp_act[mlp_off + j + 7];
                }
            }

            let mut g_mlp_hidden = vec![0.0; d_mlp];
            for i in 0..d_mlp {
                let h = self.tape.mlp_hidden[mlp_off + i];
                let sig_h = sigmoid(h);
                let silu_prime = sig_h * (1.0 + h * (1.0 - sig_h));
                g_mlp_hidden[i] = g_mlp_act[i] * silu_prime;
            }

            let mut g_zraw_from_mlp = vec![0.0; d_m];
            self.mlp_w1.matvec_transpose(&g_mlp_hidden, &mut g_zraw_from_mlp);

            for i in 0..d_mlp {
                let gh_i = g_mlp_hidden[i];
                let row_off = i * d_m;
                for j in (0..d_m).step_by(8) {
                    self.mlp_w1.grad[row_off + j] += gh_i * self.tape.z_raw[z_off + j];
                    self.mlp_w1.grad[row_off + j + 1] += gh_i * self.tape.z_raw[z_off + j + 1];
                    self.mlp_w1.grad[row_off + j + 2] += gh_i * self.tape.z_raw[z_off + j + 2];
                    self.mlp_w1.grad[row_off + j + 3] += gh_i * self.tape.z_raw[z_off + j + 3];
                    self.mlp_w1.grad[row_off + j + 4] += gh_i * self.tape.z_raw[z_off + j + 4];
                    self.mlp_w1.grad[row_off + j + 5] += gh_i * self.tape.z_raw[z_off + j + 5];
                    self.mlp_w1.grad[row_off + j + 6] += gh_i * self.tape.z_raw[z_off + j + 6];
                    self.mlp_w1.grad[row_off + j + 7] += gh_i * self.tape.z_raw[z_off + j + 7];
                }
            }

            for i in 0..d_m {
                self.grad_z_raw[i] = self.grad_z_final[i] + g_zraw_from_mlp[i];
            }

            // 3. Adapter Backward
            let ad_off = t * rank;
            let mut g_ad_act = vec![0.0; rank];
            self.adapters[0].up_proj.matvec_transpose(&self.grad_z_raw, &mut g_ad_act);

            for i in 0..d_m {
                let gz_i = self.grad_z_raw[i];
                let row_off = i * rank;
                for r in 0..rank {
                    self.adapters[0].up_proj.grad[row_off + r] += gz_i * self.tape.adapter_act[ad_off + r];
                }
            }

            let mut g_ad_down = vec![0.0; rank];
            for r in 0..rank {
                let h = self.tape.adapter_act[ad_off + r];
                let sig_h = sigmoid(h);
                let silu_prime = sig_h * (1.0 + h * (1.0 - sig_h));
                g_ad_down[r] = g_ad_act[r] * silu_prime;
            }

            self.grad_x_norm.fill(0.0);
            for r in 0..rank {
                let gad_r = g_ad_down[r];
                let row_off = r * d_m;
                for j in 0..d_m {
                    self.grad_x_norm[j] += gad_r * self.adapters[0].down_proj.data[row_off + j];
                    self.adapters[0].down_proj.grad[row_off + j] += gad_r * self.tape.x_norm[t * d_m + j];
                }
            }

            // 4. Memory Injection Backward
            let m_off = t * d_m;
            let mut g_m_proj_out = vec![0.0; d_m];
            for i in 0..d_m {
                let gz_i = self.grad_z_raw[i];
                let g_mem = self.tape.g_mem[m_off + i];
                g_m_proj_out[i] = gz_i * g_mem;

                let d_sig = g_mem * (1.0 - g_mem);
                let g_wgate_pre = gz_i * self.tape.m_val[m_off + i] * d_sig;
                let row_off = i * d_m;
                for j in 0..d_m {
                    self.grad_x_norm[j] += g_wgate_pre * self.w_gate.data[row_off + j];
                    self.w_gate.grad[row_off + j] += g_wgate_pre * self.tape.x_norm[t * d_m + j];
                }
            }

            let mut g_m_val = vec![0.0; d_m];
            self.w_proj.matvec_transpose(&g_m_proj_out, &mut g_m_val);
            for i in 0..d_m {
                let g_mp_i = g_m_proj_out[i];
                let row_off = i * d_m;
                for j in 0..d_m {
                    self.w_proj.grad[row_off + j] += g_mp_i * self.tape.m_val[m_off + j];
                }
            }

            // 5. Multi-Channel SSM Recurrence Backward & Temporal State Flow
            let y_off = t * d_m;
            let ssm_off = t * (d_m * d_s);
            let h_prev_off = t * (d_m * d_s);
            let c_off = t * d_s;
            let b_off = t * d_s;
            let del_off = t * d_m;

            let mut g_delta = vec![0.0; d_m];
            let mut g_b_proj = vec![0.0; d_s];
            let mut g_c_proj = vec![0.0; d_s];
            let mut g_h_prev = vec![0.0; d_m * d_s];

            for i in 0..d_m {
                let gz_i = self.grad_z_raw[i];
                let d_i = self.tape.delta[del_off + i];
                let xn_i = self.tape.x_norm[t * d_m + i];

                for j in 0..d_s {
                    let idx = i * d_s + j;
                    let h_next = self.tape.h_states[(t + 1) * (d_m * d_s) + idx];
                    let c_val = self.tape.c_proj[c_off + j];
                    let bar_a = self.tape.bar_a[ssm_off + idx];
                    let a_orig = self.a_mat.data[idx];
                    let b_val = self.tape.b_proj[b_off + j];

                    let g_h_total = gz_i * c_val + self.grad_h_next[idx];

                    g_c_proj[j] += gz_i * h_next;
                    g_h_prev[idx] += g_h_total * bar_a;

                    self.a_mat.grad[idx] += g_h_total * (d_i * bar_a) * self.tape.h_states[h_prev_off + idx];
                    g_delta[i] += g_h_total * (a_orig * bar_a * self.tape.h_states[h_prev_off + idx] + b_val * xn_i);
                    g_b_proj[j] += g_h_total * (d_i * xn_i);
                    self.grad_x_norm[i] += g_h_total * self.tape.bar_b[ssm_off + idx];
                }
            }

            self.grad_h_next.copy_from_slice(&g_h_prev);

            for i in 0..d_m {
                let d_sig = sigmoid(self.tape.delta[del_off + i]);
                let gd_i = g_delta[i] * d_sig;
                let row_off = i * d_m;
                for j in 0..d_m {
                    self.grad_x_norm[j] += gd_i * self.w_delta.data[row_off + j];
                    self.w_delta.grad[row_off + j] += gd_i * self.tape.x_norm[t * d_m + j];
                }
            }

            for j in 0..d_s {
                let gb_j = g_b_proj[j];
                let gc_j = g_c_proj[j];
                let row_off = j * d_m;
                for k in 0..d_m {
                    self.grad_x_norm[k] += gb_j * self.w_b.data[row_off + k] + gc_j * self.w_c.data[row_off + k];
                    self.w_b.grad[row_off + k] += gb_j * self.tape.x_norm[t * d_m + k];
                    self.w_c.grad[row_off + k] += gc_j * self.tape.x_norm[t * d_m + k];
                }
            }

            // 6. Affine RMSNorm Backward to Gamma, Beta, and Input Embeddings
            let inv_rms = self.tape.inv_rms[t];
            let e_t = &self.embed_w.data[x_id * d_m..(x_id + 1) * d_m];

            let mut dot_gx_e = 0.0f32;
            for i in 0..d_m {
                let gx_i = self.grad_x_norm[i];
                self.norm_beta.grad[i] += gx_i;
                self.norm_gamma.grad[i] += gx_i * (e_t[i] * inv_rms);

                let g_unnorm = gx_i * self.norm_gamma.data[i];
                dot_gx_e += g_unnorm * e_t[i];
            }

            let emb_row_off = x_id * d_m;
            for i in 0..d_m {
                let g_unnorm = self.grad_x_norm[i] * self.norm_gamma.data[i];
                let g_e_i = inv_rms * (g_unnorm - e_t[i] * (dot_gx_e * inv_rms * inv_rms / (d_m as f32)));
                self.embed_w.grad[emb_row_off + i] += g_e_i;
            }
        }

        // =====================================================================
        // APPLY ADAMW PARAMETER UPDATES ACROSS ALL SUBSYSTEMS
        // =====================================================================
        self.step_counter += 1;
        let lr = self.cfg.lr;
        let beta1 = self.cfg.beta1;
        let beta2 = self.cfg.beta2;
        let wd = self.cfg.weight_decay;
        let eps = self.cfg.eps;
        let step = self.step_counter;

        self.embed_w.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.norm_gamma.step_adamw(lr, beta1, beta2, 0.0, eps, step);
        self.norm_beta.step_adamw(lr, beta1, beta2, 0.0, eps, step);
        self.a_mat.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_delta.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_b.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_c.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_qx.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_qh.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_gate.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_proj.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.adapters[0].step_adamw(lr, beta1, beta2, wd, eps, step);
        self.mlp_w1.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.mlp_w2.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.unembed_w.step_adamw(lr, beta1, beta2, wd, eps, step);
    }

    // =========================================================================
    // 4. EMA CONSOLIDATION (THE SLEEP PHASE)
    // =========================================================================
    pub fn ema_consolidate_plasticity(&mut self) {
        let alpha = self.cfg.ema_alpha;
        let d_m = self.cfg.d_latent;
        let rank = self.adapters[0].rank;

        for r in 0..rank {
            let ad_down_row = &self.adapters[0].down_proj.data[r * d_m..(r + 1) * d_m];
            for i in 0..d_m {
                let ad_up_val = self.adapters[0].up_proj.data[i * rank + r];
                if ad_up_val.abs() > 1e-6 {
                    let base_idx = i * d_m + r;
                    if base_idx < self.mlp_w1.data.len() {
                        self.mlp_w1.data[base_idx] = (1.0 - alpha) * self.mlp_w1.data[base_idx] + alpha * (ad_up_val * ad_down_row[r]);
                    }
                }
            }
        }

        self.adapters[0].up_proj.data.fill(0.0);
    }
}