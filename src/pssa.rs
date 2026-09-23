use crate::adapter::PlasticAdapterV2;
use crate::backend::Device;
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

    pub fn step_adamw(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        eps: f32,
        step: usize,
    ) {
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

    pub fn step_adamw_row(
        &mut self,
        row: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        eps: f32,
        step: usize,
    ) {
        assert!(row < self.rows);
        let step_f = step as f32;
        let bias_corr1 = 1.0 - beta1.powf(step_f);
        let bias_corr2 = 1.0 - beta2.powf(step_f);
        let row_start = row * self.cols;
        let row_end = row_start + self.cols;

        for i in row_start..row_end {
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

    pub fn step_adamw(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        eps: f32,
        step: usize,
    ) {
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

impl PSSAConfigV2 {
    pub fn validate(&self) {
        assert!(
            self.d_vocab > 0
                && self.d_latent > 0
                && self.d_state > 0
                && self.d_mem_key > 0
                && self.mem_capacity > 0
                && self.chunk_len > 0,
            "all model dimensions, memory capacity, and chunk length must be positive"
        );
        assert!(
            self.lr.is_finite()
                && self.lr > 0.0
                && self.eps.is_finite()
                && self.eps > 0.0
                && self.tau_mem.is_finite()
                && self.tau_mem > 0.0,
            "lr, eps, and tau_mem must be finite and positive"
        );
        assert!(
            self.beta1.is_finite()
                && self.beta1 >= 0.0
                && self.beta1 < 1.0
                && self.beta2.is_finite()
                && self.beta2 >= 0.0
                && self.beta2 < 1.0,
            "Adam betas must be in [0, 1)"
        );
        assert!(
            self.weight_decay.is_finite()
                && self.weight_decay >= 0.0
                && self.ema_alpha.is_finite()
                && (0.0..=1.0).contains(&self.ema_alpha),
            "weight decay must be nonnegative and ema_alpha must be in [0, 1]"
        );
    }
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
// OWNED CONTINUOUS BLOCK AND MODEL-ENDPOINT TAPES
// =============================================================================

#[derive(Clone, Debug)]
pub struct PSSAContinuousConfigV2 {
    pub d_latent: usize,
    pub d_state: usize,
    pub d_mem_key: usize,
    pub mem_capacity: usize,
    pub chunk_len: usize,
    pub tau_mem: f32,
    pub ema_alpha: f32,
}

impl From<&PSSAConfigV2> for PSSAContinuousConfigV2 {
    fn from(c: &PSSAConfigV2) -> Self {
        Self {
            d_latent: c.d_latent,
            d_state: c.d_state,
            d_mem_key: c.d_mem_key,
            mem_capacity: c.mem_capacity,
            chunk_len: c.chunk_len,
            tau_mem: c.tau_mem,
            ema_alpha: c.ema_alpha,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PSSAContinuousTapeV2 {
    pub max_l: usize,
    /// Raw caller-owned continuous features, cached for affine RMS backward.
    pub x_raw: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub inv_rms: Vec<f32>,
    pub delta_raw: Vec<f32>,
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
    pub m_proj: Vec<f32>,
    pub adapter_hidden: Vec<f32>,
    pub adapter_act: Vec<f32>,
    pub z_raw: Vec<f32>,
    pub mlp_hidden: Vec<f32>,
    pub mlp_act: Vec<f32>,
    pub z_final: Vec<f32>,
}

impl PSSAContinuousTapeV2 {
    fn new(max_l: usize, d_m: usize, d_s: usize, d_k: usize, cap: usize, rank: usize) -> Self {
        Self {
            max_l,
            x_raw: vec![0.0; max_l * d_m],
            x_norm: vec![0.0; max_l * d_m],
            inv_rms: vec![0.0; max_l],
            delta_raw: vec![0.0; max_l * d_m],
            delta: vec![0.0; max_l * d_m],
            b_proj: vec![0.0; max_l * d_s],
            c_proj: vec![0.0; max_l * d_s],
            bar_a: vec![0.0; max_l * d_m * d_s],
            bar_b: vec![0.0; max_l * d_m * d_s],
            h_states: vec![0.0; (max_l + 1) * d_m * d_s],
            y_ssm: vec![0.0; max_l * d_m],
            q_euc: vec![0.0; max_l * d_k],
            q_norm: vec![0.0; max_l],
            q_poincare: vec![0.0; max_l * d_k],
            mem_weights: vec![0.0; max_l * cap],
            m_val: vec![0.0; max_l * d_m],
            g_mem: vec![0.0; max_l * d_m],
            m_inj: vec![0.0; max_l * d_m],
            m_proj: vec![0.0; max_l * d_m],
            adapter_hidden: vec![0.0; max_l * rank],
            adapter_act: vec![0.0; max_l * rank],
            z_raw: vec![0.0; max_l * d_m],
            mlp_hidden: vec![0.0; max_l * 2 * d_m],
            mlp_act: vec![0.0; max_l * 2 * d_m],
            z_final: vec![0.0; max_l * d_m],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelEndpointTapeV2 {
    pub max_l: usize,
    pub x_ids: Vec<usize>,
    pub target_ids: Vec<usize>,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub losses: Vec<f32>,
}

impl ModelEndpointTapeV2 {
    fn new(max_l: usize, d_vocab: usize) -> Self {
        Self {
            max_l,
            x_ids: vec![0; max_l],
            target_ids: vec![0; max_l],
            logits: vec![0.0; max_l * d_vocab],
            probs: vec![0.0; max_l * d_vocab],
            losses: vec![0.0; max_l],
        }
    }
}

/// Independently owned, vocabulary-free affine-RMS/SSM/memory/adapter/MLP block.
/// It accepts and differentiates raw continuous features; detached memory contents
/// and recurrent carry are state, not graph edges across chunks.
pub struct PSSAContinuousBlockV2 {
    pub cfg: PSSAContinuousConfigV2,
    pub norm_gamma: ParamVector,
    pub norm_beta: ParamVector,
    pub a_mat: ParamMatrix,
    pub w_delta: ParamMatrix,
    pub w_b: ParamMatrix,
    pub w_c: ParamMatrix,
    pub h_persistent: Vec<f32>,
    pub w_qx: ParamMatrix,
    pub w_qh: ParamMatrix,
    pub w_gate: ParamMatrix,
    pub w_proj: ParamMatrix,
    pub memory: HyperbolicEpisodicBankV2,
    pub adapters: Vec<PlasticAdapterV2>,
    pub mlp_w1: ParamMatrix,
    pub mlp_w2: ParamMatrix,
    pub tape: PSSAContinuousTapeV2,
    grad_h_next: Vec<f32>,
    grad_z_final: Vec<f32>,
    grad_z_raw: Vec<f32>,
    grad_x_norm: Vec<f32>,
    buf_m_proj: Vec<f32>,
    buf_ad_out: Vec<f32>,
    buf_g_mlp_act: Vec<f32>,
    buf_g_mlp_hidden: Vec<f32>,
    buf_g_zraw_mlp: Vec<f32>,
    buf_g_ad_act: Vec<f32>,
    buf_g_ad_down: Vec<f32>,
    buf_g_m_proj_out: Vec<f32>,
    buf_g_m_val: Vec<f32>,
    g_query_pnc: Vec<f32>,
    g_query_euc: Vec<f32>,
    g_y_ssm: Vec<f32>,
    buf_g_delta: Vec<f32>,
    buf_g_b_proj: Vec<f32>,
    buf_g_c_proj: Vec<f32>,
    buf_g_h_prev: Vec<f32>,
    inf_x_norm: Vec<f32>,
    inf_delta: Vec<f32>,
    inf_b: Vec<f32>,
    inf_c: Vec<f32>,
    inf_y_ssm: Vec<f32>,
    inf_q_euc: Vec<f32>,
    inf_q_pnc: Vec<f32>,
    inf_mem_weights: Vec<f32>,
    inf_m_val: Vec<f32>,
    inf_g_mem: Vec<f32>,
    inf_m_proj: Vec<f32>,
    inf_ad_act: Vec<f32>,
    inf_ad_out: Vec<f32>,
    inf_z_raw: Vec<f32>,
    inf_mlp_act: Vec<f32>,
    inf_mlp_out: Vec<f32>,
}

/// Token endpoint with a first continuous block and zero or more residual blocks.
/// `block` is retained as layer zero for depth-one compatibility; all further
/// blocks own their recurrent state, bank, tape, and optimizer moments.
pub struct PSSALayerV2 {
    pub cfg: PSSAConfigV2,
    pub step_counter: usize,
    pub device: Device,
    pub rng: SimpleRng,
    pub vocabulary: Vec<String>,
    pub tokenizer_json: Option<String>,
    pub embed_w: ParamMatrix,
    pub block: PSSAContinuousBlockV2,
    pub extra_blocks: Vec<PSSAContinuousBlockV2>,
    /// Fixed, non-trainable residual branch scales, one for each extra block.
    pub residual_scales: Vec<f32>,
    pub unembed_w: ParamMatrix,
    pub tape: ModelEndpointTapeV2,
    pub embed_row_marks: Vec<usize>,
    /// Loss-normalized adjoints at raw embeddings, layer-zero output, and all
    /// residual outputs.  Every vector is allocated at construction time.
    pub boundary_adjoints: Vec<Vec<f32>>,
    continuous_inputs: Vec<f32>,
    output_adjoints: Vec<f32>,
    input_adjoints: Vec<f32>,
    residual_block_adjoints: Vec<f32>,
    residual_input_adjoints: Vec<f32>,
    /// Outputs of extra blocks after their residual connection.
    layer_activations: Vec<Vec<f32>>,
    inf_features: Vec<f32>,
    inf_z_final: Vec<f32>,
    inf_block_out: Vec<f32>,
}
impl PSSAContinuousBlockV2 {
    fn new_with_rng(cfg: PSSAContinuousConfigV2, rng: &mut SimpleRng) -> Self {
        let d_m = cfg.d_latent;
        let d_s = cfg.d_state;
        let d_k = cfg.d_mem_key;
        let d_mlp = d_m * 2;
        let cap = cfg.mem_capacity;
        let chunk_len = cfg.chunk_len;
        let rank = 16;
        let norm_gamma = ParamVector::new(d_m, 1.0);
        let norm_beta = ParamVector::new(d_m, 0.0);
        let mut a_mat = ParamMatrix::zeros(d_m, d_s);
        let ln_min = 1.5f32.ln();
        let ln_max = 200.0f32.ln();
        for i in 0..d_m {
            for j in 0..d_s {
                let ratio = if d_s > 1 { j as f32 / (d_s - 1) as f32 } else { 0.0 };
                let tau = (ln_min + ratio * (ln_max - ln_min)).exp();
                let rate = 1.0 / tau;
                a_mat.data[i * d_s + j] = rate.exp_m1().ln();
            }
        }
        let w_delta = ParamMatrix::random_xavier(d_m, d_m, rng);
        let w_b = ParamMatrix::random_xavier(d_s, d_m, rng);
        let w_c = ParamMatrix::random_xavier(d_s, d_m, rng);
        let w_qx = ParamMatrix::random_xavier(d_k, d_m, rng);
        let w_qh = ParamMatrix::random_xavier(d_k, d_m, rng);
        let w_gate = ParamMatrix::random_xavier(d_m, d_m, rng);
        let w_proj = ParamMatrix::random_xavier(d_m, d_m, rng);
        let adapters = vec![PlasticAdapterV2::new(d_m, rank, rng)];
        let mlp_w1 = ParamMatrix::random_xavier(d_mlp, d_m, rng);
        let mlp_w2 = ParamMatrix::zeros(d_m, d_mlp);
        Self {
            cfg,
            norm_gamma, norm_beta, a_mat, w_delta, w_b, w_c,
            h_persistent: vec![0.0; d_m * d_s],
            w_qx, w_qh, w_gate, w_proj,
            memory: HyperbolicEpisodicBankV2::new(cap, d_k, d_m),
            adapters, mlp_w1, mlp_w2,
            tape: PSSAContinuousTapeV2::new(chunk_len, d_m, d_s, d_k, cap, rank),
            grad_h_next: vec![0.0; d_m*d_s], grad_z_final: vec![0.0; d_m],
            grad_z_raw: vec![0.0; d_m], grad_x_norm: vec![0.0; d_m],
            buf_m_proj: vec![0.0; d_m], buf_ad_out: vec![0.0; d_m],
            buf_g_mlp_act: vec![0.0; d_mlp], buf_g_mlp_hidden: vec![0.0; d_mlp],
            buf_g_zraw_mlp: vec![0.0; d_m], buf_g_ad_act: vec![0.0; rank],
            buf_g_ad_down: vec![0.0; rank], buf_g_m_proj_out: vec![0.0; d_m],
            buf_g_m_val: vec![0.0; d_m], g_query_pnc: vec![0.0; d_k],
            g_query_euc: vec![0.0; d_k], g_y_ssm: vec![0.0; d_m],
            buf_g_delta: vec![0.0; d_m], buf_g_b_proj: vec![0.0; d_s],
            buf_g_c_proj: vec![0.0; d_s], buf_g_h_prev: vec![0.0; d_m*d_s],
            inf_x_norm: vec![0.0; d_m], inf_delta: vec![0.0; d_m],
            inf_b: vec![0.0; d_s], inf_c: vec![0.0; d_s], inf_y_ssm: vec![0.0; d_m],
            inf_q_euc: vec![0.0; d_k], inf_q_pnc: vec![0.0; d_k],
            inf_mem_weights: vec![0.0; cap], inf_m_val: vec![0.0; d_m],
            inf_g_mem: vec![0.0; d_m], inf_m_proj: vec![0.0; d_m],
            inf_ad_act: vec![0.0; rank], inf_ad_out: vec![0.0; d_m],
            inf_z_raw: vec![0.0; d_m], inf_mlp_act: vec![0.0; d_mlp],
            inf_mlp_out: vec![0.0; d_m],
        }
    }

    pub fn reset_recurrent_state(&mut self) { self.h_persistent.fill(0.0); }

    /// Single-token raw-continuous inference. Performs this block's affine RMS norm.
    #[inline(always)]
    pub fn forward_continuous_inference(&mut self, x_features: &[f32], z_out: &mut [f32]) {
        assert_eq!(x_features.len(), self.cfg.d_latent);
        assert_eq!(z_out.len(), self.cfg.d_latent);
        let d_m = self.cfg.d_latent;
        let sum_sq: f32 = x_features.iter().map(|&x| x * x).sum();
        let inv_rms = 1.0 / (sum_sq / d_m as f32 + 1e-5).sqrt();
        for i in 0..d_m {
            self.inf_x_norm[i] = self.norm_gamma.data[i] * (x_features[i] * inv_rms) + self.norm_beta.data[i];
        }
        let x_norm = &self.inf_x_norm;
        let d_s = self.cfg.d_state;
        let d_k = self.cfg.d_mem_key;
        let d_mlp = d_m * 2;
        let rank = self.adapters[0].rank;
        let ssm_scale = 1.0 / (d_s as f32).sqrt();
        // 2. Data-Dependent Projections
        self.w_delta.matvec(x_norm, &mut self.inf_delta);
        for i in 0..d_m {
            self.inf_delta[i] = softplus(self.inf_delta[i]);
        }

        self.w_b.matvec(x_norm, &mut self.inf_b);
        self.w_c.matvec(x_norm, &mut self.inf_c);

        // 3. Multi-Channel SSM State Update (In-Place on Persistent State)
        for i in 0..d_m {
            let d_i = self.inf_delta[i];
            let mut y_i = 0.0f32;
            let row_off = i * d_s;

            for j in 0..d_s {
                let idx = row_off + j;
                let bar_a = (d_i * -softplus(self.a_mat.data[idx])).exp();
                let bar_b = d_i * self.inf_b[j];

                let h_val = bar_a * self.h_persistent[idx] + bar_b * x_norm[i];
                self.h_persistent[idx] = h_val;
                y_i += h_val * self.inf_c[j];
            }
            self.inf_y_ssm[i] = y_i;
        }

        // 4. Diffeomorphic Poincaré Memory Retrieval
        for r in 0..d_k {
            let row_x = &self.w_qx.data[r * d_m..(r + 1) * d_m];
            let row_h = &self.w_qh.data[r * d_m..(r + 1) * d_m];
            self.inf_q_euc[r] =
                dot_slice(row_x, x_norm) + dot_slice(row_h, &self.inf_y_ssm);
        }

        HyperbolicEpisodicBankV2::diffeomorphic_project(&self.inf_q_euc, &mut self.inf_q_pnc);

        self.memory.retrieve_soft_into(
            &self.inf_q_pnc,
            self.cfg.tau_mem,
            &mut self.inf_m_val,
            &mut self.inf_mem_weights,
        );

        self.w_gate.matvec(x_norm, &mut self.inf_g_mem);
        for i in 0..d_m {
            self.inf_g_mem[i] = sigmoid(self.inf_g_mem[i]);
        }

        self.w_proj.matvec(&self.inf_m_val, &mut self.inf_m_proj);

        // 5. Plastic Adapter
        self.adapters[0]
            .down_proj
            .matvec(x_norm, &mut self.inf_ad_act[..rank]);
        for r in 0..rank {
            let h = self.inf_ad_act[r];
            self.inf_ad_act[r] = h * sigmoid(h);
        }
        self.adapters[0].total_up_matvec(&self.inf_ad_act[..rank], &mut self.inf_ad_out);

        // 6. Latent Aggregation & SiLU MLP Expansion
        for i in 0..d_m {
            self.inf_z_raw[i] = (self.inf_y_ssm[i] * ssm_scale)
                + (self.inf_g_mem[i] * self.inf_m_proj[i])
                + self.inf_ad_out[i];
        }

        self.mlp_w1
            .matvec(&self.inf_z_raw, &mut self.inf_mlp_act[..d_mlp]);
        for i in 0..d_mlp {
            let h = self.inf_mlp_act[i];
            self.inf_mlp_act[i] = h * sigmoid(h);
        }

        self.mlp_w2
            .matvec(&self.inf_mlp_act[..d_mlp], &mut self.inf_mlp_out);
        for i in 0..d_m {
            z_out[i] = self.inf_z_raw[i] + self.inf_mlp_out[i];
        }

    }

    /// Chunk forward from raw continuous rows. Outputs remain in `tape.z_final`.
    pub fn forward_train_chunk(&mut self, raw_features: &[f32], seq_len: usize) -> &[f32] {
        assert!(seq_len > 0 && seq_len <= self.cfg.chunk_len);
        let d_m = self.cfg.d_latent;
        assert_eq!(raw_features.len(), seq_len * d_m);
        let d_s = self.cfg.d_state;
        let d_k = self.cfg.d_mem_key;
        let d_mlp = d_m * 2;
        let mem_cap = self.cfg.mem_capacity;
        let rank = self.adapters[0].rank;
        let ssm_scale = 1.0 / (d_s as f32).sqrt();
        self.tape.x_raw[..seq_len*d_m].copy_from_slice(raw_features);
        self.tape.h_states[..d_m*d_s].copy_from_slice(&self.h_persistent);
        for t in 0..seq_len {
            let raw_off = t*d_m;
            let raw = &self.tape.x_raw[raw_off..raw_off+d_m];
            let sum_sq: f32 = raw.iter().map(|&x| x*x).sum();
            let inv_rms = 1.0 / (sum_sq / d_m as f32 + 1e-5).sqrt();
            self.tape.inv_rms[t] = inv_rms;
            for i in 0..d_m {
                self.tape.x_norm[raw_off+i] = self.norm_gamma.data[i] * (raw[i]*inv_rms) + self.norm_beta.data[i];
            }
            let x_n = &self.tape.x_norm[raw_off..raw_off+d_m];
            // 2. Data-Dependent Projections
            let del_off = t * d_m;
            self.w_delta
                .matvec(x_n, &mut self.tape.delta_raw[del_off..del_off + d_m]);
            for i in 0..d_m {
                self.tape.delta[del_off + i] = softplus(self.tape.delta_raw[del_off + i]);
            }
            let delta = &self.tape.delta[del_off..del_off + d_m];

            let b_off = t * d_s;
            self.w_b
                .matvec(x_n, &mut self.tape.b_proj[b_off..b_off + d_s]);
            let b_p = &self.tape.b_proj[b_off..b_off + d_s];

            let c_off = t * d_s;
            self.w_c
                .matvec(x_n, &mut self.tape.c_proj[c_off..c_off + d_s]);
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
                    let bar_a = (d_i * -softplus(self.a_mat.data[idx])).exp();
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

            self.w_gate
                .matvec(x_n, &mut self.tape.g_mem[m_off..m_off + d_m]);
            for i in 0..d_m {
                self.tape.g_mem[m_off + i] = sigmoid(self.tape.g_mem[m_off + i]);
            }

            self.w_proj
                .matvec(&self.tape.m_val[m_off..m_off + d_m], &mut self.buf_m_proj);
            for i in 0..d_m {
                self.tape.m_proj[m_off + i] = self.buf_m_proj[i];
                self.tape.m_inj[m_off + i] = self.tape.g_mem[m_off + i] * self.buf_m_proj[i];
            }

            // 5. Zero-Init Plastic Adapter
            let ad_off = t * rank;
            self.adapters[0]
                .down_proj
                .matvec(x_n, &mut self.tape.adapter_hidden[ad_off..ad_off + rank]);
            for r in 0..rank {
                let h = self.tape.adapter_hidden[ad_off + r];
                self.tape.adapter_act[ad_off + r] = h * sigmoid(h);
            }
            self.adapters[0].total_up_matvec(
                &self.tape.adapter_act[ad_off..ad_off + rank],
                &mut self.buf_ad_out,
            );

            // 6. Latent Aggregation & SiLU MLP Expansion
            let z_off = t * d_m;
            for i in 0..d_m {
                self.tape.z_raw[z_off + i] =
                    (y_ssm[i] * ssm_scale) + self.tape.m_inj[m_off + i] + self.buf_ad_out[i];
            }
            let z_raw = &self.tape.z_raw[z_off..z_off + d_m];

            let mlp_off = t * d_mlp;
            self.mlp_w1
                .matvec(z_raw, &mut self.tape.mlp_hidden[mlp_off..mlp_off + d_mlp]);
            for i in 0..d_mlp {
                let h = self.tape.mlp_hidden[mlp_off + i];
                self.tape.mlp_act[mlp_off + i] = h * sigmoid(h);
            }
            self.mlp_w2.matvec(
                &self.tape.mlp_act[mlp_off..mlp_off + d_mlp],
                &mut self.tape.z_final[z_off..z_off + d_m],
            );
            for i in 0..d_m {
                self.tape.z_final[z_off + i] += z_raw[i];
            }
        }
        let last = seq_len*d_m*d_s;
        self.h_persistent.copy_from_slice(&self.tape.h_states[last..last+d_m*d_s]);
        &self.tape.z_final[..seq_len*d_m]
    }

    pub fn zero_gradients(&mut self) {
        self.norm_gamma.zero_grad(); self.norm_beta.zero_grad(); self.a_mat.zero_grad();
        self.w_delta.zero_grad(); self.w_b.zero_grad(); self.w_c.zero_grad();
        self.w_qx.zero_grad(); self.w_qh.zero_grad(); self.w_gate.zero_grad(); self.w_proj.zero_grad();
        self.adapters[0].zero_grad(); self.mlp_w1.zero_grad(); self.mlp_w2.zero_grad();
    }

    /// Reverse-mode block VJP. External output adjoints are consumed without scaling.
    /// Returned raw-input adjoints use the same row-major `[time, latent]` layout.
    pub fn backward_chunk(&mut self, output_adjoints: &[f32], seq_len: usize, input_adjoints: &mut [f32]) {
        assert!(seq_len > 0 && seq_len <= self.cfg.chunk_len);
        let d_m = self.cfg.d_latent;
        assert_eq!(output_adjoints.len(), seq_len*d_m);
        assert_eq!(input_adjoints.len(), seq_len*d_m);
        let d_s = self.cfg.d_state;
        let d_k = self.cfg.d_mem_key;
        let d_mlp = d_m * 2;
        let rank = self.adapters[0].rank;
        let ssm_scale = 1.0 / (d_s as f32).sqrt();
        self.grad_h_next.fill(0.0);
        for t in (0..seq_len).rev() {
            let z_off = t*d_m;
            self.grad_z_final.copy_from_slice(&output_adjoints[z_off..z_off+d_m]);
            // 2. SiLU MLP Backward
            let mlp_off = t * d_mlp;
            self.mlp_w2
                .matvec_transpose(&self.grad_z_final, &mut self.buf_g_mlp_act);

            for i in 0..d_m {
                let gz_i = self.grad_z_final[i];
                let row_off = i * d_mlp;
                for j in 0..d_mlp {
                    self.mlp_w2.grad[row_off + j] += gz_i * self.tape.mlp_act[mlp_off + j];
                }
            }

            for i in 0..d_mlp {
                let h = self.tape.mlp_hidden[mlp_off + i];
                let sig_h = sigmoid(h);
                let silu_prime = sig_h * (1.0 + h * (1.0 - sig_h));
                self.buf_g_mlp_hidden[i] = self.buf_g_mlp_act[i] * silu_prime;
            }

            self.mlp_w1
                .matvec_transpose(&self.buf_g_mlp_hidden, &mut self.buf_g_zraw_mlp);

            for i in 0..d_mlp {
                let gh_i = self.buf_g_mlp_hidden[i];
                let row_off = i * d_m;
                for j in 0..d_m {
                    self.mlp_w1.grad[row_off + j] += gh_i * self.tape.z_raw[z_off + j];
                }
            }

            for i in 0..d_m {
                self.grad_z_raw[i] = self.grad_z_final[i] + self.buf_g_zraw_mlp[i];
            }

            // 3. Adapter Backward
            let ad_off = t * rank;
            self.adapters[0].total_up_matvec_transpose(&self.grad_z_raw, &mut self.buf_g_ad_act);

            for i in 0..d_m {
                let gz_i = self.grad_z_raw[i];
                let row_off = i * rank;
                for r in 0..rank {
                    self.adapters[0].up_proj.grad[row_off + r] +=
                        gz_i * self.tape.adapter_act[ad_off + r];
                }
            }

            for r in 0..rank {
                let h = self.tape.adapter_hidden[ad_off + r];
                let sig_h = sigmoid(h);
                let silu_prime = sig_h * (1.0 + h * (1.0 - sig_h));
                self.buf_g_ad_down[r] = self.buf_g_ad_act[r] * silu_prime;
            }

            self.grad_x_norm.fill(0.0);
            for r in 0..rank {
                let gad_r = self.buf_g_ad_down[r];
                let row_off = r * d_m;
                for j in 0..d_m {
                    self.grad_x_norm[j] += gad_r * self.adapters[0].down_proj.data[row_off + j];
                    self.adapters[0].down_proj.grad[row_off + j] +=
                        gad_r * self.tape.x_norm[t * d_m + j];
                }
            }

            // 4. Memory Injection Backward
            let m_off = t * d_m;
            for i in 0..d_m {
                let gz_i = self.grad_z_raw[i];
                let g_mem = self.tape.g_mem[m_off + i];
                self.buf_g_m_proj_out[i] = gz_i * g_mem;

                let d_sig = g_mem * (1.0 - g_mem);
                let g_wgate_pre = gz_i * self.tape.m_proj[m_off + i] * d_sig;
                let row_off = i * d_m;
                for j in 0..d_m {
                    self.grad_x_norm[j] += g_wgate_pre * self.w_gate.data[row_off + j];
                    self.w_gate.grad[row_off + j] += g_wgate_pre * self.tape.x_norm[t * d_m + j];
                }
            }

            self.w_proj
                .matvec_transpose(&self.buf_g_m_proj_out, &mut self.buf_g_m_val);
            for i in 0..d_m {
                let g_mp_i = self.buf_g_m_proj_out[i];
                let row_off = i * d_m;
                for j in 0..d_m {
                    self.w_proj.grad[row_off + j] += g_mp_i * self.tape.m_val[m_off + j];
                }
            }

            // Retrieval is a full-bank softmax over -hyperbolic_distance/tau.
            // Bank keys/values are detached stored state; only the query path learns.
            self.g_query_pnc.fill(0.0);
            let q_off = t * d_k;
            let q = &self.tape.q_poincare[q_off..q_off + d_k];
            let q_sq = dot_slice(q, q) as f64;
            for entry in 0..self.memory.count {
                let key_off = entry * d_k;
                let key = &self.memory.keys[key_off..key_off + d_k];
                let key_sq = self.memory.norm_sq[entry] as f64;
                let mut dot_g_value_minus_mean = 0.0f64;
                let value_off = entry * d_m;
                for j in 0..d_m {
                    dot_g_value_minus_mean += self.buf_g_m_val[j] as f64
                        * (self.memory.values[value_off + j] - self.tape.m_val[m_off + j]) as f64;
                }
                let g_score = self.tape.mem_weights[t * self.cfg.mem_capacity + entry] as f64
                    * dot_g_value_minus_mean;
                let mut sq = 0.0f64;
                for k in 0..d_k {
                    let diff = (q[k] - key[k]) as f64;
                    sq += diff * diff;
                }
                // The distance has a cusp at identical points.  We explicitly use
                // the symmetric subgradient zero there, avoiding division by zero.
                if sq > 0.0 {
                    let denom = (1.0 - q_sq) * (1.0 - key_sq);
                    assert!(denom > 0.0 && denom.is_finite());
                    let z = sq / denom;
                    let dd_dz = 1.0 / (z * (1.0 + z)).sqrt();
                    for k in 0..d_k {
                        let diff = (q[k] - key[k]) as f64;
                        let ddenom = -2.0 * q[k] as f64 * (1.0 - key_sq);
                        let dz = (2.0 * diff * denom - sq * ddenom) / (denom * denom);
                        self.g_query_pnc[k] +=
                            (g_score * (-1.0 / self.cfg.tau_mem as f64) * dd_dz * dz) as f32;
                    }
                }
            }
            let r = self.tape.q_norm[t];
            if r == 0.0 {
                self.g_query_euc.copy_from_slice(&self.g_query_pnc);
            } else {
                let q_euc = &self.tape.q_euc[q_off..q_off + d_k];
                let mut qdotg = 0.0;
                for k in 0..d_k {
                    qdotg += q_euc[k] * self.g_query_pnc[k];
                }
                let denom = r * (1.0 + r) * (1.0 + r);
                for k in 0..d_k {
                    self.g_query_euc[k] =
                        self.g_query_pnc[k] / (1.0 + r) - q_euc[k] * qdotg / denom;
                }
            }
            self.g_y_ssm.fill(0.0);
            let xn = &self.tape.x_norm[t * d_m..(t + 1) * d_m];
            let y = &self.tape.y_ssm[t * d_m..(t + 1) * d_m];
            for r_i in 0..d_k {
                let gq = self.g_query_euc[r_i];
                let row = r_i * d_m;
                for j in 0..d_m {
                    self.w_qx.grad[row + j] += gq * xn[j];
                    self.w_qh.grad[row + j] += gq * y[j];
                    self.grad_x_norm[j] += gq * self.w_qx.data[row + j];
                    self.g_y_ssm[j] += gq * self.w_qh.data[row + j];
                }
            }

            // 5. Multi-Channel SSM Recurrence Backward & Temporal State Flow
            let ssm_off = t * (d_m * d_s);
            let h_prev_off = t * (d_m * d_s);
            let c_off = t * d_s;
            let b_off = t * d_s;
            let del_off = t * d_m;

            self.buf_g_delta.fill(0.0);
            self.buf_g_b_proj.fill(0.0);
            self.buf_g_c_proj.fill(0.0);
            self.buf_g_h_prev.fill(0.0);

            for i in 0..d_m {
                let gz_i = self.grad_z_raw[i];
                let g_y_i = gz_i * ssm_scale + self.g_y_ssm[i];
                let d_i = self.tape.delta[del_off + i];
                let xn_i = self.tape.x_norm[t * d_m + i];

                for j in 0..d_s {
                    let idx = i * d_s + j;
                    let h_next = self.tape.h_states[(t + 1) * (d_m * d_s) + idx];
                    let c_val = self.tape.c_proj[c_off + j];
                    let bar_a = self.tape.bar_a[ssm_off + idx];
                    let a_raw = self.a_mat.data[idx];
                    let a_physical = -softplus(a_raw);
                    let b_val = self.tape.b_proj[b_off + j];

                    let g_h_total = g_y_i * c_val + self.grad_h_next[idx];

                    self.buf_g_c_proj[j] += g_y_i * h_next;
                    self.buf_g_h_prev[idx] += g_h_total * bar_a;

                    // dA/draw = -sigmoid(raw) for A=-softplus(raw).
                    self.a_mat.grad[idx] += g_h_total
                        * (d_i * bar_a)
                        * self.tape.h_states[h_prev_off + idx]
                        * -sigmoid(a_raw);
                    self.buf_g_delta[i] += g_h_total
                        * (a_physical * bar_a * self.tape.h_states[h_prev_off + idx]
                            + b_val * xn_i);
                    self.buf_g_b_proj[j] += g_h_total * (d_i * xn_i);
                    self.grad_x_norm[i] += g_h_total * self.tape.bar_b[ssm_off + idx];
                }
            }

            self.grad_h_next.copy_from_slice(&self.buf_g_h_prev);

            for i in 0..d_m {
                let d_sig = sigmoid(self.tape.delta_raw[del_off + i]);
                let gd_i = self.buf_g_delta[i] * d_sig;
                let row_off = i * d_m;
                for j in 0..d_m {
                    self.grad_x_norm[j] += gd_i * self.w_delta.data[row_off + j];
                    self.w_delta.grad[row_off + j] += gd_i * self.tape.x_norm[t * d_m + j];
                }
            }

            for j in 0..d_s {
                let gb_j = self.buf_g_b_proj[j];
                let gc_j = self.buf_g_c_proj[j];
                let row_off = j * d_m;
                for k in 0..d_m {
                    self.grad_x_norm[k] +=
                        gb_j * self.w_b.data[row_off + k] + gc_j * self.w_c.data[row_off + k];
                    self.w_b.grad[row_off + k] += gb_j * self.tape.x_norm[t * d_m + k];
                    self.w_c.grad[row_off + k] += gc_j * self.tape.x_norm[t * d_m + k];
                }
            }

            // Affine RMSNorm VJP to the caller's raw continuous row.
            let inv_rms = self.tape.inv_rms[t];
            let raw = &self.tape.x_raw[t*d_m..(t+1)*d_m];
            let mut dot_gx_raw = 0.0f32;
            for i in 0..d_m {
                let gx_i = self.grad_x_norm[i];
                self.norm_beta.grad[i] += gx_i;
                self.norm_gamma.grad[i] += gx_i * (raw[i] * inv_rms);
                dot_gx_raw += gx_i * self.norm_gamma.data[i] * raw[i];
            }
            for i in 0..d_m {
                let g_unnorm = self.grad_x_norm[i] * self.norm_gamma.data[i];
                input_adjoints[z_off+i] = inv_rms * (g_unnorm - raw[i] * (dot_gx_raw * inv_rms * inv_rms / d_m as f32));
            }
        }
    }

    pub fn apply_adamw(&mut self, lr: f32, beta1: f32, beta2: f32, wd: f32, eps: f32, step: usize) {
        self.norm_gamma.step_adamw(lr,beta1,beta2,0.0,eps,step);
        self.norm_beta.step_adamw(lr,beta1,beta2,0.0,eps,step);
        self.a_mat.step_adamw(lr,beta1,beta2,wd,eps,step);
        self.w_delta.step_adamw(lr,beta1,beta2,wd,eps,step);
        self.w_b.step_adamw(lr,beta1,beta2,wd,eps,step);
        self.w_c.step_adamw(lr,beta1,beta2,wd,eps,step);
        self.w_qx.step_adamw(lr,beta1,beta2,wd,eps,step);
        self.w_qh.step_adamw(lr,beta1,beta2,wd,eps,step);
        self.w_gate.step_adamw(lr,beta1,beta2,wd,eps,step);
        self.w_proj.step_adamw(lr,beta1,beta2,wd,eps,step);
        self.adapters[0].step_adamw(lr,beta1,beta2,wd,eps,step);
        self.mlp_w1.step_adamw(lr,beta1,beta2,wd,eps,step);
        self.mlp_w2.step_adamw(lr,beta1,beta2,wd,eps,step);
    }

    pub fn ema_consolidate_plasticity(&mut self) { self.adapters[0].consolidate(self.cfg.ema_alpha); }
}
impl PSSALayerV2 {
    pub const MAX_DEPTH: usize = 32;

    pub fn new(cfg: PSSAConfigV2, seed: u64) -> Self { Self::new_with_device(cfg, seed, Device::Cpu) }
    pub fn new_with_device(cfg: PSSAConfigV2, seed: u64, device: Device) -> Self {
        Self::new_with_depth_and_device(cfg, seed, 1, device)
    }
    pub fn new_with_depth(cfg: PSSAConfigV2, seed: u64, depth: usize) -> Self {
        Self::new_with_depth_and_device(cfg, seed, depth, Device::Cpu)
    }
    fn new_with_depth_and_device(cfg: PSSAConfigV2, seed: u64, depth: usize, device: Device) -> Self {
        cfg.validate();
        assert!((1..=Self::MAX_DEPTH).contains(&depth), "depth must be in 1..={}", Self::MAX_DEPTH);
        assert!(!device.is_gpu(), "Device::Gpu is not dispatched by PSSALayerV2; use Device::Cpu");
        // This is deliberately checked before any vectors are allocated.  It is
        // conservative (one complete legacy allocation per layer), and therefore
        // includes every additional block tape, backward scratch and activation.
        let base = model_allocation_estimate(&cfg).expect("model allocation overflow");
        let total = base.checked_mul(depth).expect("stacked model allocation overflow");
        assert!(total <= 1024 * 1024 * 1024, "stacked model allocation exceeds 1 GiB cap");
        let mut rng = SimpleRng::new(seed);
        let d_v=cfg.d_vocab; let d_m=cfg.d_latent; let l=cfg.chunk_len;
        // Keep the exact old draw order for depth one.  Extra layers begin only
        // after the old embedding/block/head sequence has completed.
        let embed_w=ParamMatrix::random_xavier(d_v,d_m,&mut rng);
        let block=PSSAContinuousBlockV2::new_with_rng(PSSAContinuousConfigV2::from(&cfg),&mut rng);
        let unembed_w=ParamMatrix::random_xavier(d_v,d_m,&mut rng);
        let mut extra_blocks=Vec::with_capacity(depth-1);
        for _ in 1..depth { extra_blocks.push(PSSAContinuousBlockV2::new_with_rng(PSSAContinuousConfigV2::from(&cfg),&mut rng)); }
        let scale=1.0/(depth as f32).sqrt();
        Self { cfg, step_counter:0, device, rng, vocabulary:Vec::new(), tokenizer_json:None,
            embed_w, block, extra_blocks, residual_scales:vec![scale;depth-1], unembed_w,
            tape:ModelEndpointTapeV2::new(l,d_v), embed_row_marks:vec![0;d_v],
            boundary_adjoints:(0..=depth).map(|_|vec![0.0;l*d_m]).collect(),
            continuous_inputs:vec![0.0;l*d_m], output_adjoints:vec![0.0;l*d_m], input_adjoints:vec![0.0;l*d_m],
            residual_block_adjoints:vec![0.0;l*d_m], residual_input_adjoints:vec![0.0;l*d_m],
            layer_activations:(0..depth-1).map(|_|vec![0.0;l*d_m]).collect(),
            inf_features:vec![0.0;d_m], inf_z_final:vec![0.0;d_m], inf_block_out:vec![0.0;d_m] }
    }
    pub fn depth(&self) -> usize { self.extra_blocks.len()+1 }
    pub fn reset_recurrent_state(&mut self) {
        self.block.reset_recurrent_state();
        for b in &mut self.extra_blocks { b.reset_recurrent_state(); }
    }

    #[inline(always)]
    pub fn forward_inference(&mut self, x_id: usize, logits_out: &mut [f32]) {
        assert!(x_id < self.cfg.d_vocab); assert_eq!(logits_out.len(),self.cfg.d_vocab);
        let d_m=self.cfg.d_latent;
        self.inf_features.copy_from_slice(&self.embed_w.data[x_id*d_m..(x_id+1)*d_m]);
        self.block.forward_continuous_inference(&self.inf_features,&mut self.inf_z_final);
        for (index, b) in self.extra_blocks.iter_mut().enumerate() {
            b.forward_continuous_inference(&self.inf_z_final, &mut self.inf_block_out);
            let scale=self.residual_scales[index];
            for i in 0..d_m { self.inf_features[i]=self.inf_z_final[i]+scale*self.inf_block_out[i]; }
            self.inf_z_final.copy_from_slice(&self.inf_features);
        }
        self.unembed_w.matvec(&self.inf_z_final,logits_out);
        let scale=1.0/(d_m as f32).sqrt(); for x in logits_out { *x *= scale; }
    }

    pub fn forward_train_chunk(&mut self, token_ids: &[usize], target_ids: &[usize]) -> f32 {
        assert!(!token_ids.is_empty(),"training chunk must be nonempty");
        assert_eq!(token_ids.len(),target_ids.len(),"token and target counts must match");
        let seq_len=token_ids.len().min(self.cfg.chunk_len);
        assert!(token_ids[..seq_len].iter().chain(&target_ids[..seq_len]).all(|&x|x<self.cfg.d_vocab),"token IDs must be in vocabulary");
        self.tape.x_ids[..seq_len].copy_from_slice(&token_ids[..seq_len]); self.tape.target_ids[..seq_len].copy_from_slice(&target_ids[..seq_len]);
        let d_m=self.cfg.d_latent; let d_v=self.cfg.d_vocab;
        for t in 0..seq_len { let id=token_ids[t]; self.continuous_inputs[t*d_m..(t+1)*d_m].copy_from_slice(&self.embed_w.data[id*d_m..(id+1)*d_m]); }
        self.block.forward_train_chunk(&self.continuous_inputs[..seq_len*d_m],seq_len);
        for layer in 0..self.extra_blocks.len() {
            if layer == 0 {
                let previous=&self.block.tape.z_final[..seq_len*d_m];
                self.extra_blocks[layer].forward_train_chunk(previous,seq_len);
                let branch=&self.extra_blocks[layer].tape.z_final[..seq_len*d_m]; let out=&mut self.layer_activations[layer][..seq_len*d_m]; let scale=self.residual_scales[layer];
                for i in 0..seq_len*d_m { out[i]=previous[i]+scale*branch[i]; }
            } else {
                let (prior, after)=self.layer_activations.split_at_mut(layer);
                let previous=&prior[layer-1][..seq_len*d_m];
                self.extra_blocks[layer].forward_train_chunk(previous,seq_len);
                let branch=&self.extra_blocks[layer].tape.z_final[..seq_len*d_m]; let out=&mut after[0][..seq_len*d_m]; let scale=self.residual_scales[layer];
                for i in 0..seq_len*d_m { out[i]=previous[i]+scale*branch[i]; }
            }
        }
        let final_z: &[f32]=if self.extra_blocks.is_empty() { &self.block.tape.z_final[..seq_len*d_m] } else { &self.layer_activations[self.extra_blocks.len()-1][..seq_len*d_m] };
        let logit_scale=1.0/(d_m as f32).sqrt(); let mut total_loss=0.0f32;
        for t in 0..seq_len {
            let z=&final_z[t*d_m..(t+1)*d_m]; let off=t*d_v; self.unembed_w.matvec(z,&mut self.tape.logits[off..off+d_v]);
            for i in 0..d_v { self.tape.logits[off+i]*=logit_scale; }
            let mut max_l=f32::NEG_INFINITY; for i in 0..d_v { max_l=max_l.max(self.tape.logits[off+i]); }
            let mut sum_exp=0.0f32; for i in 0..d_v { let e=(self.tape.logits[off+i]-max_l).exp(); self.tape.probs[off+i]=e; sum_exp+=e; }
            let inv=1.0/sum_exp.max(1e-8); for i in 0..d_v { self.tape.probs[off+i]*=inv; }
            let loss=(max_l-self.tape.logits[off+target_ids[t]])+sum_exp.ln(); self.tape.losses[t]=loss; total_loss+=loss;
        }
        total_loss/seq_len as f32
    }
    pub fn zero_gradients(&mut self) { self.embed_w.zero_grad(); self.block.zero_gradients(); for b in &mut self.extra_blocks { b.zero_gradients(); } self.unembed_w.zero_grad(); }
    pub fn backward_chunk(&mut self, seq_len: usize, accumulation_scale: f32) {
        assert!(seq_len>0 && seq_len<=self.cfg.chunk_len); assert!(accumulation_scale.is_finite());
        let d_m=self.cfg.d_latent; let d_v=self.cfg.d_vocab; let n=seq_len*d_m;
        let scale_loss=accumulation_scale/seq_len as f32; let logit_scale=1.0/(d_m as f32).sqrt(); let pending=self.step_counter+1;
        for t in 0..seq_len { self.embed_row_marks[self.tape.x_ids[t]]=pending; }
        self.output_adjoints[..n].fill(0.0);
        let final_z: &[f32]=if self.extra_blocks.is_empty() { &self.block.tape.z_final[..n] } else { &self.layer_activations[self.extra_blocks.len()-1][..n] };
        for t in (0..seq_len).rev() { let zoff=t*d_m; let loff=t*d_v;
            for i in 0..d_v { let indicator=if i==self.tape.target_ids[t]{1.0}else{0.0}; let g=(self.tape.probs[loff+i]-indicator)*scale_loss*logit_scale; let row=i*d_m;
                for j in 0..d_m { self.output_adjoints[zoff+j]+=g*self.unembed_w.data[row+j]; self.unembed_w.grad[row+j]+=g*final_z[zoff+j]; }
            }
        }
        let final_boundary=self.depth(); self.boundary_adjoints[final_boundary][..n].copy_from_slice(&self.output_adjoints[..n]);
        for layer in (0..self.extra_blocks.len()).rev() {
            let scale=self.residual_scales[layer];
            for i in 0..n { self.residual_block_adjoints[i]=self.output_adjoints[i]*scale; }
            self.extra_blocks[layer].backward_chunk(&self.residual_block_adjoints[..n],seq_len,&mut self.residual_input_adjoints[..n]);
            for i in 0..n { self.output_adjoints[i]+=self.residual_input_adjoints[i]; }
            self.boundary_adjoints[layer+1][..n].copy_from_slice(&self.output_adjoints[..n]);
        }
        self.block.backward_chunk(&self.output_adjoints[..n],seq_len,&mut self.input_adjoints[..n]);
        self.boundary_adjoints[0][..n].copy_from_slice(&self.input_adjoints[..n]);
        for t in (0..seq_len).rev() { let row=self.tape.x_ids[t]*d_m; let off=t*d_m; for j in 0..d_m { self.embed_w.grad[row+j]+=self.input_adjoints[off+j]; } }
    }
    pub fn apply_adamw(&mut self, lr:f32) { self.step_counter+=1; let s=self.step_counter; let c=&self.cfg;
        self.embed_w.step_adamw(lr,c.beta1,c.beta2,c.weight_decay,c.eps,s); self.block.apply_adamw(lr,c.beta1,c.beta2,c.weight_decay,c.eps,s);
        for b in &mut self.extra_blocks { b.apply_adamw(lr,c.beta1,c.beta2,c.weight_decay,c.eps,s); }
        self.unembed_w.step_adamw(lr,c.beta1,c.beta2,c.weight_decay,c.eps,s); }
    pub fn backward_and_step_chunk(&mut self, seq_len:usize) { self.zero_gradients(); self.backward_chunk(seq_len,1.0); self.apply_adamw(self.cfg.lr); }
    pub fn ema_consolidate_plasticity(&mut self) { self.block.ema_consolidate_plasticity(); for b in &mut self.extra_blocks { b.ema_consolidate_plasticity(); } }
    /// Maintains the established surprise threshold for every layer independently.
    pub fn insert_training_memory(&mut self, loss:f32, seq_len:usize) {
        assert!(seq_len>0 && seq_len<=self.cfg.chunk_len);
        if !(loss>3.5) { return; }
        let last=seq_len-1; let step=self.step_counter;
        fn insert(b:&mut PSSAContinuousBlockV2,last:usize,step:usize,loss:f32) { let k=b.cfg.d_mem_key; let m=b.cfg.d_latent;
            b.inf_q_pnc[..k].copy_from_slice(&b.tape.q_poincare[last*k..(last+1)*k]);
            b.inf_z_raw[..m].copy_from_slice(&b.tape.z_final[last*m..(last+1)*m]);
            let mut key=[0.0f32; 0]; // keeps the following split borrows visibly disjoint to the compiler
            let _=&mut key;
            let (q, v)=(&b.inf_q_pnc[..k], &b.inf_z_raw[..m]); b.memory.insert_protected(q,v,loss,step); }
        insert(&mut self.block,last,step,loss); for b in &mut self.extra_blocks { insert(b,last,step,loss); }
    }
    pub fn parameter_count(&self) -> usize {
        fn block(b:&PSSAContinuousBlockV2)->usize { b.norm_gamma.data.len()+b.norm_beta.data.len()+b.a_mat.data.len()+b.w_delta.data.len()+b.w_b.data.len()+b.w_c.data.len()+b.w_qx.data.len()+b.w_qh.data.len()+b.w_gate.data.len()+b.w_proj.data.len()+b.adapters.iter().map(|a|a.down_proj.data.len()+a.up_proj.data.len()).sum::<usize>()+b.mlp_w1.data.len()+b.mlp_w2.data.len() }
        self.embed_w.data.len()+self.unembed_w.data.len()+block(&self.block)+self.extra_blocks.iter().map(block).sum::<usize>()
    }
    pub fn all_parameters_finite(&self) -> bool {
        fn p(x:&ParamMatrix)->bool { x.data.iter().chain(&x.grad).chain(&x.m).chain(&x.v).all(|v|v.is_finite()) }
        fn v(x:&ParamVector)->bool { x.data.iter().chain(&x.grad).chain(&x.m).chain(&x.v).all(|z|z.is_finite()) }
        fn b(x:&PSSAContinuousBlockV2)->bool { v(&x.norm_gamma)&&v(&x.norm_beta)&&p(&x.a_mat)&&p(&x.w_delta)&&p(&x.w_b)&&p(&x.w_c)&&p(&x.w_qx)&&p(&x.w_qh)&&p(&x.w_gate)&&p(&x.w_proj)&&p(&x.mlp_w1)&&p(&x.mlp_w2)&&x.adapters.iter().all(|a|p(&a.down_proj)&&p(&a.up_proj)&&a.consolidated_up.iter().all(|z|z.is_finite())) }
        p(&self.embed_w)&&p(&self.unembed_w)&&b(&self.block)&&self.extra_blocks.iter().all(b)
    }
}

fn model_allocation_estimate(c:&PSSAConfigV2)->Option<usize> {
    // A checked conservative construction guard.  Four copies cover parameter
    // data/grad/moments and the remaining terms cover persistent/tape/scratch.
    let m=c.d_latent; let v=c.d_vocab; let s=c.d_state; let k=c.d_mem_key; let cap=c.mem_capacity; let l=c.chunk_len;
    let mut n=v.checked_mul(m)?.checked_mul(8)?; // shared endpoint params+tapes
    let block=m.checked_mul(s)?.checked_mul(2)?.checked_add(m.checked_mul(m)?.checked_mul(14)?)?.checked_add(s.checked_mul(m)?.checked_mul(2)?)?.checked_add(k.checked_mul(m)?.checked_mul(2)?)?.checked_add(cap.checked_mul(k.checked_add(m)?)?)?.checked_add(l.checked_mul(m.checked_mul(s.checked_add(18)?)?.checked_add(k)?.checked_add(cap)?.checked_add(64)?)?)?;
    n=n.checked_add(block.checked_mul(4)?)?;
    n.checked_mul(4)
}
