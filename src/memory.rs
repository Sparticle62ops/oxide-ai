use crate::defense::{RateLimiterGate, UpdateOutcome};
use crate::linalg::dot_slice;
use std::f32;

#[derive(Clone, Debug, PartialEq)]
pub struct HyperbolicEpisodicBankV2 {
    pub capacity: usize,
    pub count: usize,
    pub dim_key: usize,
    pub dim_val: usize,
    pub write_head: usize,

    // Contiguous Structure-of-Arrays layout for CPU cache streaming
    pub keys: Vec<f32>,     // [capacity x dim_key]
    pub values: Vec<f32>,   // [capacity x dim_val]
    pub norm_sq: Vec<f32>,  // [capacity]
    pub confidence: Vec<f32>,
    pub last_seen_step: Vec<usize>,
}

impl HyperbolicEpisodicBankV2 {
    pub fn new(capacity: usize, dim_key: usize, dim_val: usize) -> Self {
        Self {
            capacity,
            count: 0,
            dim_key,
            dim_val,
            write_head: 0,
            keys: vec![0.0; capacity * dim_key],
            values: vec![0.0; capacity * dim_val],
            norm_sq: vec![0.0; capacity],
            confidence: vec![1.0; capacity],
            last_seen_step: vec![0; capacity],
        }
    }

    /// Diffeomorphic mapping from unconstrained Euclidean space into the open Poincaré ball:
    /// q_poincare = q_euclidean / (1 + ||q_euclidean||_2)
    #[inline(always)]
    pub fn diffeomorphic_project(q_euc: &[f32], out_pnc: &mut [f32]) -> f32 {
        let q_sq = dot_slice(q_euc, q_euc);
        let q_norm = q_sq.sqrt();
        let scale = 1.0 / (1.0 + q_norm);
        for i in 0..q_euc.len() {
            out_pnc[i] = q_euc[i] * scale;
        }
        q_norm
    }

    /// Exact Riemannian distance in the Poincaré ball manifold D^d:
    /// d_D(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
    #[inline(always)]
    pub fn poincare_distance(u: &[f32], u_sq: f32, v: &[f32], v_sq: f32) -> f32 {
        let mut sq_dist = 0.0f32;
        for i in 0..u.len() {
            let d = u[i] - v[i];
            sq_dist += d * d;
        }
        let denom = ((1.0 - u_sq) * (1.0 - v_sq)).max(1e-7);
        let alpha = 1.0 + 2.0 * sq_dist / denom;
        (alpha + (alpha * alpha - 1.0).max(0.0).sqrt()).ln()
    }

    /// Constant O(1) episodic memory insertion with circular eviction
    pub fn insert(&mut self, key_pnc: &[f32], val: &[f32]) -> usize {
        let idx = if self.count < self.capacity {
            let i = self.count;
            self.count += 1;
            i
        } else {
            let i = self.write_head;
            self.write_head = (self.write_head + 1) % self.capacity;
            i
        };

        let k_off = idx * self.dim_key;
        self.keys[k_off..k_off + self.dim_key].copy_from_slice(key_pnc);
        self.norm_sq[idx] = dot_slice(key_pnc, key_pnc);

        let v_off = idx * self.dim_val;
        self.values[v_off..v_off + self.dim_val].copy_from_slice(val);
        self.confidence[idx] = 1.0;
        self.last_seen_step[idx] = 0;
        idx
    }

    pub fn insert_protected(
        &mut self,
        key_pnc: &[f32],
        val: &[f32],
        surprise: f32,
        current_step: usize,
    ) -> Option<usize> {
        if self.count < self.capacity {
            let idx = self.insert(key_pnc, val);
            self.last_seen_step[idx] = current_step;
            return Some(idx);
        }

        let idx = self.write_head;
        match RateLimiterGate::apply_refractory_overwrite(
            &mut self.confidence[idx],
            &mut self.last_seen_step[idx],
            current_step,
            surprise,
        ) {
            UpdateOutcome::Defended { .. } => None,
            UpdateOutcome::Overwritten => {
                let inserted = self.insert(key_pnc, val);
                self.last_seen_step[inserted] = current_step;
                Some(inserted)
            }
            UpdateOutcome::Stable { .. } => None,
        }
    }

    /// Continuous soft-attention retrieval over the hyperbolic episodic store
    pub fn retrieve_soft_into(
        &self,
        q_pnc: &[f32],
        tau: f32,
        out_val: &mut [f32],
        out_weights: &mut [f32],
    ) -> f32 {
        if self.count == 0 {
            out_val.fill(0.0);
            if !out_weights.is_empty() {
                out_weights.fill(0.0);
            }
            return 0.0;
        }

        let u_sq = dot_slice(q_pnc, q_pnc);
        let mut min_dist = f32::MAX;
        let mut max_neg_dist = f32::NEG_INFINITY;
        let count = self.count;

        // 1. Calculate continuous Riemannian distances and attention logits
        for idx in 0..count {
            let k_off = idx * self.dim_key;
            let k_ptr = &self.keys[k_off..k_off + self.dim_key];
            let v_sq = self.norm_sq[idx];
            let dist = Self::poincare_distance(q_pnc, u_sq, k_ptr, v_sq);

            if dist < min_dist {
                min_dist = dist;
            }

            let score = -dist / tau.max(1e-4);
            out_weights[idx] = score;
            if score > max_neg_dist {
                max_neg_dist = score;
            }
        }

        // 2. Numerically stable softmax normalization
        let mut sum_exp = 0.0f32;
        for idx in 0..count {
            let w = (out_weights[idx] - max_neg_dist).exp();
            out_weights[idx] = w;
            sum_exp += w;
        }

        let inv_sum = 1.0 / sum_exp.max(1e-8);
        out_val.fill(0.0);

        // 3. Weighted episodic vector aggregation
        for idx in 0..count {
            let w = out_weights[idx] * inv_sum;
            out_weights[idx] = w;
            let v_off = idx * self.dim_val;
            let val_slice = &self.values[v_off..v_off + self.dim_val];
            for j in 0..self.dim_val {
                out_val[j] += w * val_slice[j];
            }
        }

        min_dist
    }
}