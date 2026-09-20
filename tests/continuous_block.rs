//! Direct validation of the vocabulary-free continuous PSSA block.
//!
//! The finite-difference tolerance is `5e-6 + 2% * max(|analytic|, |numeric|)`.
//! The absolute component accounts for f32 central-difference cancellation at a
//! 2e-3 perturbation. In particular, this budget is smaller than even the small
//! populated-memory Qh derivative, so a missing query chain term cannot pass.

use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};

const D: usize = 3;
const L: usize = 3;
const FD_STEP: f32 = 2.0e-3;
const FD_ABS: f32 = 5.0e-6;
const FD_REL: f32 = 0.02;

fn config() -> PSSAConfigV2 {
    PSSAConfigV2 {
        d_vocab: 11,
        d_latent: D,
        d_state: 2,
        d_mem_key: 2,
        mem_capacity: 3,
        chunk_len: L,
        lr: 1.0e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.0,
        eps: 1.0e-8,
        tau_mem: 0.65,
        ema_alpha: 0.1,
    }
}

/// Uses nonzero output paths and two distinct, detached bank entries.  This is
/// intentionally a direct block fixture: the model is only used to obtain the
/// private block constructor through `PSSALayerV2::new(...).block`.
fn block_fixture() -> PSSALayerV2 {
    let mut model = PSSALayerV2::new(config(), 0x5eed);
    let block = &mut model.block;

    block.norm_gamma.data.copy_from_slice(&[1.11, 0.87, 1.06]);
    block.norm_beta.data.copy_from_slice(&[0.03, -0.02, 0.01]);
    for matrix in [
        &mut block.w_delta,
        &mut block.w_b,
        &mut block.w_c,
        &mut block.w_qx,
        &mut block.w_qh,
        &mut block.w_gate,
        &mut block.w_proj,
        &mut block.adapters[0].down_proj,
        &mut block.mlp_w1,
        &mut block.mlp_w2,
    ] {
        for (i, value) in matrix.data.iter_mut().enumerate() {
            *value = 0.073 + 0.011 * ((i % 7) as f32 - 3.0);
        }
    }
    for (i, value) in block.adapters[0].up_proj.data.iter_mut().enumerate() {
        *value = 0.051 + 0.009 * ((i % 5) as f32 - 2.0);
    }
    for (i, value) in block.adapters[0].consolidated_up.iter_mut().enumerate() {
        *value = -0.037 + 0.007 * ((i % 6) as f32 - 2.5);
    }
    for (i, value) in block.h_persistent.iter_mut().enumerate() {
        *value = 0.08 + 0.015 * i as f32;
    }

    // The two entries differ in both key and value.  They are stored state, so
    // finite-difference calls recreate the exact same detached bank each time.
    block.memory.insert(&[0.14, -0.05], &[0.21, -0.16, 0.09]);
    block.memory.insert(&[-0.09, 0.12], &[-0.11, 0.18, 0.27]);
    model
}

fn raw_inputs() -> [f32; L * D] {
    [0.22, -0.31, 0.17, 0.13, 0.28, -0.24, -0.34, 0.19, 0.41]
}

fn external_adjoints() -> [f32; L * D] {
    [0.73, -0.41, 0.52, -0.29, 0.67, 0.38, 0.44, 0.16, -0.58]
}

fn objective(model: &mut PSSALayerV2, raw: &[f32], adjoints: &[f32]) -> f32 {
    let z = model.block.forward_train_chunk(raw, L);
    z.iter().zip(adjoints).map(|(z, g)| z * g).sum()
}

fn assert_close(label: &str, analytic: f32, numeric: f32) {
    let error = (analytic - numeric).abs();
    let limit = FD_ABS + FD_REL * analytic.abs().max(numeric.abs());
    assert!(
        error <= limit,
        "{label}: analytic={analytic:.8}, numeric={numeric:.8}, error={error:.8}, limit={limit:.8}"
    );
}

#[derive(Copy, Clone, Debug)]
enum Probe {
    Gamma,
    Beta,
    Rate,
    Delta,
    B,
    C,
    Qx,
    Qh,
    Gate,
    Projection,
    AdapterDown,
    AdapterFastUp,
    AdapterSlowUp,
    Mlp1,
    Mlp2,
}

fn value(model: &PSSALayerV2, probe: Probe) -> f32 {
    let b = &model.block;
    match probe {
        Probe::Gamma => b.norm_gamma.data[0],
        Probe::Beta => b.norm_beta.data[1],
        Probe::Rate => b.a_mat.data[0],
        Probe::Delta => b.w_delta.data[0],
        Probe::B => b.w_b.data[0],
        Probe::C => b.w_c.data[0],
        Probe::Qx => b.w_qx.data[0],
        Probe::Qh => b.w_qh.data[0],
        Probe::Gate => b.w_gate.data[0],
        Probe::Projection => b.w_proj.data[0],
        Probe::AdapterDown => b.adapters[0].down_proj.data[0],
        Probe::AdapterFastUp => b.adapters[0].up_proj.data[0],
        Probe::AdapterSlowUp => b.adapters[0].consolidated_up[0],
        Probe::Mlp1 => b.mlp_w1.data[0],
        Probe::Mlp2 => b.mlp_w2.data[0],
    }
}

fn put(model: &mut PSSALayerV2, probe: Probe, value: f32) {
    let b = &mut model.block;
    match probe {
        Probe::Gamma => b.norm_gamma.data[0] = value,
        Probe::Beta => b.norm_beta.data[1] = value,
        Probe::Rate => b.a_mat.data[0] = value,
        Probe::Delta => b.w_delta.data[0] = value,
        Probe::B => b.w_b.data[0] = value,
        Probe::C => b.w_c.data[0] = value,
        Probe::Qx => b.w_qx.data[0] = value,
        Probe::Qh => b.w_qh.data[0] = value,
        Probe::Gate => b.w_gate.data[0] = value,
        Probe::Projection => b.w_proj.data[0] = value,
        Probe::AdapterDown => b.adapters[0].down_proj.data[0] = value,
        Probe::AdapterFastUp => b.adapters[0].up_proj.data[0] = value,
        Probe::AdapterSlowUp => b.adapters[0].consolidated_up[0] = value,
        Probe::Mlp1 => b.mlp_w1.data[0] = value,
        Probe::Mlp2 => b.mlp_w2.data[0] = value,
    }
}

/// Consolidated coefficients are deliberately not optimizer parameters.  Their
/// derivative is nevertheless exactly the fast-up derivative because the block
/// uses their elementwise sum as its effective up matrix.
fn analytic_derivative(model: &PSSALayerV2, probe: Probe) -> f32 {
    let b = &model.block;
    match probe {
        Probe::Gamma => b.norm_gamma.grad[0],
        Probe::Beta => b.norm_beta.grad[1],
        Probe::Rate => b.a_mat.grad[0],
        Probe::Delta => b.w_delta.grad[0],
        Probe::B => b.w_b.grad[0],
        Probe::C => b.w_c.grad[0],
        Probe::Qx => b.w_qx.grad[0],
        Probe::Qh => b.w_qh.grad[0],
        Probe::Gate => b.w_gate.grad[0],
        Probe::Projection => b.w_proj.grad[0],
        Probe::AdapterDown => b.adapters[0].down_proj.grad[0],
        Probe::AdapterFastUp | Probe::AdapterSlowUp => b.adapters[0].up_proj.grad[0],
        Probe::Mlp1 => b.mlp_w1.grad[0],
        Probe::Mlp2 => b.mlp_w2.grad[0],
    }
}

fn numeric_derivative(probe: Probe, raw: &[f32], adjoints: &[f32]) -> f32 {
    let mut plus = block_fixture();
    let x = value(&plus, probe);
    put(&mut plus, probe, x + FD_STEP);
    let hi = objective(&mut plus, raw, adjoints);

    let mut minus = block_fixture();
    put(&mut minus, probe, x - FD_STEP);
    let lo = objective(&mut minus, raw, adjoints);
    (hi - lo) / (2.0 * FD_STEP)
}

#[test]
fn arbitrary_external_vjp_matches_raw_and_parameter_central_differences() {
    let raw = raw_inputs();
    let adjoints = external_adjoints();
    let mut analytic = block_fixture();
    analytic.block.forward_train_chunk(&raw, L);
    analytic.block.zero_gradients();
    let mut raw_grads = [0.0; L * D];
    analytic
        .block
        .backward_chunk(&adjoints, L, &mut raw_grads);

    // Every coordinate of a genuinely multi-time raw input VJP is checked.
    for coordinate in 0..raw.len() {
        let mut plus_raw = raw;
        plus_raw[coordinate] += FD_STEP;
        let mut minus_raw = raw;
        minus_raw[coordinate] -= FD_STEP;
        let hi = objective(&mut block_fixture(), &plus_raw, &adjoints);
        let lo = objective(&mut block_fixture(), &minus_raw, &adjoints);
        let numeric = (hi - lo) / (2.0 * FD_STEP);
        assert!(
            raw_grads[coordinate].abs().max(numeric.abs()) > 1.0e-7,
            "raw coordinate {coordinate} has a vacuous derivative"
        );
        assert_close(&format!("raw[{coordinate}]"), raw_grads[coordinate], numeric);
    }

    let probes = [
        Probe::Gamma,
        Probe::Beta,
        Probe::Rate,
        Probe::Delta,
        Probe::B,
        Probe::C,
        Probe::Qx,
        Probe::Qh,
        Probe::Gate,
        Probe::Projection,
        Probe::AdapterDown,
        Probe::AdapterFastUp,
        Probe::AdapterSlowUp,
        Probe::Mlp1,
        Probe::Mlp2,
    ];
    for probe in probes {
        let numeric = numeric_derivative(probe, &raw, &adjoints);
        let derivative = analytic_derivative(&analytic, probe);
        println!("FD {probe:?}: analytic={derivative:.9} numeric={numeric:.9}");
        assert!(
            derivative.abs().max(numeric.abs()) > 1.0e-7,
            "{probe:?} has a vacuous derivative"
        );
        assert_close(&format!("{probe:?}"), derivative, numeric);
    }
}

struct Vjp {
    input: [f32; L * D],
    probes: [f32; 4],
}

fn vjp(adjoints: &[f32]) -> Vjp {
    let raw = raw_inputs();
    let mut model = block_fixture();
    model.block.forward_train_chunk(&raw, L);
    model.block.zero_gradients();
    let mut input = [0.0; L * D];
    model.block.backward_chunk(adjoints, L, &mut input);
    Vjp {
        input,
        probes: [
            model.block.norm_gamma.grad[0],
            model.block.w_qh.grad[0],
            model.block.adapters[0].up_proj.grad[0],
            model.block.mlp_w2.grad[0],
        ],
    }
}

#[test]
fn external_output_adjoint_vjp_is_linear_and_scales_without_hidden_loss_scaling() {
    let a = external_adjoints();
    let b = [-0.21, 0.34, 0.19, 0.48, -0.36, 0.22, -0.17, 0.29, 0.61];
    let sum: [f32; L * D] = core::array::from_fn(|i| a[i] + b[i]);
    let scale = 2.75;
    let scaled: [f32; L * D] = core::array::from_fn(|i| a[i] * scale);
    let va = vjp(&a);
    let vb = vjp(&b);
    let vsum = vjp(&sum);
    let vscaled = vjp(&scaled);

    for i in 0..L * D {
        assert_close("VJP additivity/input", vsum.input[i], va.input[i] + vb.input[i]);
        assert_close("VJP scaling/input", vscaled.input[i], va.input[i] * scale);
    }
    for i in 0..va.probes.len() {
        assert_close("VJP additivity/parameter", vsum.probes[i], va.probes[i] + vb.probes[i]);
        assert_close("VJP scaling/parameter", vscaled.probes[i], va.probes[i] * scale);
    }
}

fn assert_vectors_close(label: &str, left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len(), "{label}: length mismatch");
    for (i, (&a, &b)) in left.iter().zip(right).enumerate() {
        let limit = 3.0e-6 + 3.0e-6 * a.abs().max(b.abs());
        assert!((a - b).abs() <= limit, "{label}[{i}]: {a} vs {b}");
    }
}

#[test]
fn continuous_chunk_split_and_repeated_inference_preserve_outputs_and_carry() {
    let raw = raw_inputs();
    let mut contiguous = block_fixture();
    let full = contiguous.block.forward_train_chunk(&raw, L).to_vec();
    let full_state = contiguous.block.h_persistent.clone();

    let mut split = block_fixture();
    let first = split.block.forward_train_chunk(&raw[..D], 1).to_vec();
    let second = split.block.forward_train_chunk(&raw[D..], 2).to_vec();
    let mut joined = first;
    joined.extend_from_slice(&second);
    assert_vectors_close("contiguous versus split output", &full, &joined);
    assert_vectors_close("contiguous versus split carry", &full_state, &split.block.h_persistent);

    let mut repeated = block_fixture();
    let mut one_at_a_time = Vec::with_capacity(L * D);
    let mut out = [0.0; D];
    for t in 0..L {
        repeated
            .block
            .forward_continuous_inference(&raw[t * D..(t + 1) * D], &mut out);
        one_at_a_time.extend_from_slice(&out);
    }
    assert_vectors_close("train chunk versus repeated inference", &full, &one_at_a_time);
    assert_vectors_close("train carry versus inference carry", &full_state, &repeated.block.h_persistent);
}

#[test]
fn backward_after_a_split_chunk_uses_detached_carry_not_the_previous_tape() {
    let raw = raw_inputs();
    let adjoints = external_adjoints();

    let mut split = block_fixture();
    split.block.forward_train_chunk(&raw[..D], 1);
    let detached_carry = split.block.h_persistent.clone();
    split.block.forward_train_chunk(&raw[D..], 2);
    split.block.zero_gradients();
    let mut split_input = [0.0; 2 * D];
    split
        .block
        .backward_chunk(&adjoints[D..], 2, &mut split_input);

    // This equivalent run supplies precisely the saved carry but has never
    // recorded the prior row in its tape.  Equality therefore detects accidental
    // reverse-time traversal across a chunk boundary.
    let mut isolated = block_fixture();
    isolated.block.h_persistent.copy_from_slice(&detached_carry);
    isolated.block.forward_train_chunk(&raw[D..], 2);
    isolated.block.zero_gradients();
    let mut isolated_input = [0.0; 2 * D];
    isolated
        .block
        .backward_chunk(&adjoints[D..], 2, &mut isolated_input);

    assert_vectors_close("detached split input VJP", &split_input, &isolated_input);
    assert_vectors_close(
        "detached split norm gradients",
        &split.block.norm_gamma.grad,
        &isolated.block.norm_gamma.grad,
    );
    assert_vectors_close(
        "detached split SSM gradients",
        &split.block.a_mat.grad,
        &isolated.block.a_mat.grad,
    );
}

#[test]
fn exposed_continuous_tapes_are_independent_of_vocabulary_width() {
    let narrow = block_fixture();
    let mut wide_config = config();
    wide_config.d_vocab = 97;
    let wide = PSSALayerV2::new(wide_config, 0x5eed);

    assert_eq!(narrow.block.cfg.d_latent, wide.block.cfg.d_latent);
    assert_eq!(narrow.block.cfg.d_state, wide.block.cfg.d_state);
    assert_eq!(narrow.block.cfg.d_mem_key, wide.block.cfg.d_mem_key);
    assert_eq!(narrow.block.tape.x_raw.len(), wide.block.tape.x_raw.len());
    assert_eq!(narrow.block.tape.z_final.len(), wide.block.tape.z_final.len());
    assert_eq!(narrow.block.tape.h_states.len(), wide.block.tape.h_states.len());
    assert_eq!(narrow.block.tape.mem_weights.len(), wide.block.tape.mem_weights.len());
    assert_eq!(narrow.block.memory.keys.len(), wide.block.memory.keys.len());
    assert_eq!(narrow.block.memory.values.len(), wide.block.memory.values.len());
}

#[test]
fn token_endpoint_is_the_only_embedding_gradient_scatter_site() {
    let token_ids = [1usize, 3, 5];
    let target_ids = [2usize, 4, 6];
    let mut token_model = block_fixture();
    token_model.forward_train_chunk(&token_ids, &target_ids);
    token_model.zero_gradients();
    token_model.backward_chunk(L, 1.0);

    // Rebuild the exact endpoint CE VJP, but send it directly to a separate
    // continuous block.  Its returned raw adjoints must be exactly what the
    // token endpoint scatters into its selected embedding rows.
    let mut direct = block_fixture();
    let mut raw = [0.0; L * D];
    for t in 0..L {
        raw[t * D..(t + 1) * D].copy_from_slice(
            &direct.embed_w.data[token_ids[t] * D..(token_ids[t] + 1) * D],
        );
    }
    direct.block.forward_train_chunk(&raw, L);
    direct.block.zero_gradients();
    let mut output_adjoints = [0.0; L * D];
    let logit_scale = 1.0 / (D as f32).sqrt();
    for t in (0..L).rev() {
        for vocabulary_row in 0..direct.cfg.d_vocab {
            let indicator = if vocabulary_row == target_ids[t] { 1.0 } else { 0.0 };
            let g = (token_model.tape.probs[t * direct.cfg.d_vocab + vocabulary_row] - indicator)
                * (1.0 / L as f32)
                * logit_scale;
            for j in 0..D {
                output_adjoints[t * D + j] += g * direct.unembed_w.data[vocabulary_row * D + j];
            }
        }
    }
    let mut direct_input = [0.0; L * D];
    direct
        .block
        .backward_chunk(&output_adjoints, L, &mut direct_input);

    for t in 0..L {
        assert_vectors_close(
            "outer embedding scatter",
            &token_model.embed_w.grad[token_ids[t] * D..(token_ids[t] + 1) * D],
            &direct_input[t * D..(t + 1) * D],
        );
    }
    for row in 0..token_model.cfg.d_vocab {
        if !token_ids.contains(&row) {
            assert!(
                token_model.embed_w.grad[row * D..(row + 1) * D]
                    .iter()
                    .all(|g| *g == 0.0),
                "unselected embedding row {row} received a block-side gradient"
            );
        }
    }
    assert_vectors_close(
        "direct continuous versus endpoint block parameters",
        &token_model.block.norm_gamma.grad,
        &direct.block.norm_gamma.grad,
    );
}
