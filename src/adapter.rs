use crate::linalg::{sigmoid, SimpleRng};
use crate::pssa::ParamMatrix;

#[derive(Clone, Debug, PartialEq)]
pub struct PlasticAdapterV2 {
    pub down_proj: ParamMatrix, // [rank x d_latent]
    pub up_proj: ParamMatrix,   // [d_latent x rank] -> initialized to STRICT ZERO
    pub rank: usize,
    pub d_latent: usize,
}

impl PlasticAdapterV2 {
    /// Creates a new Plastic Adapter with zero-initialized up_proj.
    /// This mathematically guarantees that newly spawned adapters output +0.0 initially,
    /// avoiding Covariate Shift and representation destruction.
    pub fn new(d_latent: usize, rank: usize, rng: &mut SimpleRng) -> Self {
        Self {
            down_proj: ParamMatrix::random_xavier(rank, d_latent, rng),
            up_proj: ParamMatrix::zeros(d_latent, rank), // STRICT ZERO-INITIALIZATION
            rank,
            d_latent,
        }
    }

    /// Evaluates the adapter: out += W_up * SiLU(W_down * x)
    #[inline(always)]
    pub fn forward_into(&self, x: &[f32], act_buf: &mut [f32], out: &mut [f32]) {
        // 1. Down-projection: h = W_down * x
        self.down_proj.matvec(x, act_buf);

        // 2. Non-linear SiLU activation: h_act = h * sigmoid(h)
        for r in 0..self.rank {
            let h = act_buf[r];
            act_buf[r] = h * sigmoid(h);
        }

        // 3. Up-projection: out += W_up * h_act
        self.up_proj.matvec(act_buf, out);
    }

    /// Resets parameter gradients
    pub fn zero_grad(&mut self) {
        self.down_proj.zero_grad();
        self.up_proj.zero_grad();
    }

    /// Applies AdamW parameter updates
    pub fn step_adamw(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        eps: f32,
        step: usize,
    ) {
        self.down_proj.step_adamw(lr, beta1, beta2, weight_decay, eps, step);
        self.up_proj.step_adamw(lr, beta1, beta2, weight_decay, eps, step);
    }
}