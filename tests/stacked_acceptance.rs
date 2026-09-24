//! Acceptance coverage for the depth-2/depth-4 stacked model.
//!
//! The finite-difference objective recomputes CE from the f32 logits in f64.  The
//! model intentionally stores logits/probabilities in f32, so this avoids an
//! additional f32 reduction/cancellation source in `(loss(+h)-loss(-h))/(2h)`.
//! `4e-6 + 1.5%` is deliberately smaller than every accepted derivative: the
//! fixture asserts the chosen coordinate exceeds `3e-5`, over seven times the
//! absolute budget.  Coordinates are selected by the largest analytic magnitude
//! within their required family, never by silently accepting a vacuous zero.

use oxide_ai_pssa::checkpoint::{load_checkpoint, save_model, CheckpointFormat};
use oxide_ai_pssa::pssa::{
    PSSAConfigV2, PSSAContinuousBlockV2, PSSALayerV2, ParamMatrix, ParamVector,
};
use std::fs;

const IDS: [usize; 3] = [1, 3, 5];
const TARGETS: [usize; 3] = [3, 5, 7];
const FD_STEP: f32 = 2.0e-3;
const FD_ABS: f32 = 4.0e-6;
const FD_REL: f32 = 0.015;
const FD_MIN: f32 = 3.0e-5;

fn cfg() -> PSSAConfigV2 {
    PSSAConfigV2 {
        d_vocab: 9,
        d_latent: 3,
        d_state: 2,
        d_mem_key: 2,
        mem_capacity: 4,
        chunk_len: 3,
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.0,
        eps: 1.0e-8,
        tau_mem: 0.75,
        ema_alpha: 0.1,
    }
}

fn patterned(values: &mut [f32], base: f32, layer: usize) {
    for (i, value) in values.iter_mut().enumerate() {
        *value = base + 0.013 * ((i % 7) as f32 - 3.0) + 0.004 * layer as f32;
    }
}

/// A deliberately live full-model fixture.  In particular, the two output
/// matrices which production initialization leaves at zero (fast adapter up and
/// MLP2) are activated, and every bank has two fixed, distinct detached entries.
/// Query weights remain small enough that the Poincare mapping is far from a
/// saturated regime while retrieval has a plainly nonzero effect.
fn fixture(depth: usize) -> PSSALayerV2 {
    let mut model = PSSALayerV2::new_with_depth(cfg(), 0x5a17, depth);
    patterned(&mut model.embed_w.data, 0.17, 0);
    patterned(&mut model.unembed_w.data, -0.11, 1);
    for layer in 0..depth {
        let block = block_mut(&mut model, layer);
        patterned(&mut block.norm_gamma.data, 0.98, layer);
        patterned(&mut block.norm_beta.data, 0.025, layer);
        patterned(&mut block.a_mat.data, -0.58, layer);
        patterned(&mut block.w_delta.data, 0.13, layer);
        patterned(&mut block.w_b.data, 0.17, layer);
        patterned(&mut block.w_c.data, -0.15, layer);
        patterned(&mut block.w_qx.data, 0.055, layer);
        patterned(&mut block.w_qh.data, -0.047, layer);
        patterned(&mut block.w_gate.data, 0.16, layer);
        patterned(&mut block.w_proj.data, -0.22, layer);
        patterned(&mut block.adapters[0].down_proj.data, 0.07, layer);
        patterned(&mut block.adapters[0].up_proj.data, 0.065, layer);
        patterned(&mut block.adapters[0].consolidated_up, -0.018, layer);
        patterned(&mut block.mlp_w1.data, 0.075, layer);
        patterned(&mut block.mlp_w2.data, 0.055, layer);
        patterned(&mut block.h_persistent, 0.19, layer);
        block.memory.insert(&[0.22, -0.14], &[1.55, -0.95, 0.68]);
        block.memory.insert(&[-0.17, 0.19], &[-1.03, 1.23, -0.78]);
        assert_eq!(block.memory.count, 2, "fixture bank must be populated");
    }
    model
}

fn block(model: &PSSALayerV2, index: usize) -> &PSSAContinuousBlockV2 {
    if index == 0 { &model.block } else { &model.extra_blocks[index - 1] }
}
fn block_mut(model: &mut PSSALayerV2, index: usize) -> &mut PSSAContinuousBlockV2 {
    if index == 0 { &mut model.block } else { &mut model.extra_blocks[index - 1] }
}

fn ce_from_f32_logits(model: &mut PSSALayerV2) -> f64 {
    model.forward_train_chunk(&IDS, &TARGETS);
    let width = model.cfg.d_vocab;
    let mut total = 0.0f64;
    for (t, &target) in TARGETS.iter().enumerate() {
        let logits = &model.tape.logits[t * width..(t + 1) * width];
        let maximum = logits.iter().map(|&x| f64::from(x)).fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = logits.iter().map(|&x| (f64::from(x) - maximum).exp()).sum();
        total += maximum - f64::from(logits[target]) + sum_exp.ln();
    }
    total / TARGETS.len() as f64
}

#[derive(Copy, Clone, Debug)]
enum Family {
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
    AdapterUp,
    Mlp1,
    Mlp2,
}

const FAMILIES: [Family; 14] = [
    Family::Gamma, Family::Beta, Family::Rate, Family::Delta, Family::B,
    Family::C, Family::Qx, Family::Qh, Family::Gate, Family::Projection,
    Family::AdapterDown, Family::AdapterUp, Family::Mlp1, Family::Mlp2,
];

fn family_data(block: &PSSAContinuousBlockV2, family: Family) -> &[f32] {
    match family {
        Family::Gamma => &block.norm_gamma.data,
        Family::Beta => &block.norm_beta.data,
        Family::Rate => &block.a_mat.data,
        Family::Delta => &block.w_delta.data,
        Family::B => &block.w_b.data,
        Family::C => &block.w_c.data,
        Family::Qx => &block.w_qx.data,
        Family::Qh => &block.w_qh.data,
        Family::Gate => &block.w_gate.data,
        Family::Projection => &block.w_proj.data,
        Family::AdapterDown => &block.adapters[0].down_proj.data,
        Family::AdapterUp => &block.adapters[0].up_proj.data,
        Family::Mlp1 => &block.mlp_w1.data,
        Family::Mlp2 => &block.mlp_w2.data,
    }
}
fn family_data_mut(block: &mut PSSAContinuousBlockV2, family: Family) -> &mut [f32] {
    match family {
        Family::Gamma => &mut block.norm_gamma.data,
        Family::Beta => &mut block.norm_beta.data,
        Family::Rate => &mut block.a_mat.data,
        Family::Delta => &mut block.w_delta.data,
        Family::B => &mut block.w_b.data,
        Family::C => &mut block.w_c.data,
        Family::Qx => &mut block.w_qx.data,
        Family::Qh => &mut block.w_qh.data,
        Family::Gate => &mut block.w_gate.data,
        Family::Projection => &mut block.w_proj.data,
        Family::AdapterDown => &mut block.adapters[0].down_proj.data,
        Family::AdapterUp => &mut block.adapters[0].up_proj.data,
        Family::Mlp1 => &mut block.mlp_w1.data,
        Family::Mlp2 => &mut block.mlp_w2.data,
    }
}
fn family_grad(block: &PSSAContinuousBlockV2, family: Family) -> &[f32] {
    match family {
        Family::Gamma => &block.norm_gamma.grad,
        Family::Beta => &block.norm_beta.grad,
        Family::Rate => &block.a_mat.grad,
        Family::Delta => &block.w_delta.grad,
        Family::B => &block.w_b.grad,
        Family::C => &block.w_c.grad,
        Family::Qx => &block.w_qx.grad,
        Family::Qh => &block.w_qh.grad,
        Family::Gate => &block.w_gate.grad,
        Family::Projection => &block.w_proj.grad,
        Family::AdapterDown => &block.adapters[0].down_proj.grad,
        Family::AdapterUp => &block.adapters[0].up_proj.grad,
        Family::Mlp1 => &block.mlp_w1.grad,
        Family::Mlp2 => &block.mlp_w2.grad,
    }
}

fn strongest_coordinate(derivatives: &[f32], label: &str) -> usize {
    assert!(!derivatives.is_empty(), "{label} has no coordinates");
    let mut result = 0;
    for (i, &derivative) in derivatives.iter().enumerate() {
        assert!(derivative.is_finite(), "{label}[{i}] is non-finite");
        if derivative.abs() > derivatives[result].abs() { result = i; }
    }
    result
}

fn assert_fd(label: &str, analytic: f32, numeric: f64) {
    let analytic = f64::from(analytic);
    let magnitude = analytic.abs().max(numeric.abs());
    assert!(magnitude > f64::from(FD_MIN), "{label}: vacuous derivative analytic={analytic:.9} numeric={numeric:.9}");
    let error = (analytic - numeric).abs();
    let limit = f64::from(FD_ABS) + f64::from(FD_REL) * magnitude;
    assert!(
        error <= limit,
        "{label}: analytic={analytic:.9} numeric={numeric:.9} error={error:.9} limit={limit:.9}"
    );
}

fn numeric_shared(depth: usize, embedding: bool, coordinate: usize, original: f32) -> f64 {
    let mut hi = fixture(depth);
    let mut lo = fixture(depth);
    if embedding {
        hi.embed_w.data[coordinate] = original + FD_STEP;
        lo.embed_w.data[coordinate] = original - FD_STEP;
    } else {
        hi.unembed_w.data[coordinate] = original + FD_STEP;
        lo.unembed_w.data[coordinate] = original - FD_STEP;
    }
    (ce_from_f32_logits(&mut hi) - ce_from_f32_logits(&mut lo)) / f64::from(2.0 * FD_STEP)
}

fn numeric_layer(depth: usize, layer: usize, family: Family, coordinate: usize, original: f32) -> f64 {
    let mut hi = fixture(depth);
    let mut lo = fixture(depth);
    family_data_mut(block_mut(&mut hi, layer), family)[coordinate] = original + FD_STEP;
    family_data_mut(block_mut(&mut lo, layer), family)[coordinate] = original - FD_STEP;
    (ce_from_f32_logits(&mut hi) - ce_from_f32_logits(&mut lo)) / f64::from(2.0 * FD_STEP)
}

#[test]
fn every_stacked_block_and_shared_endpoint_has_a_live_full_model_finite_difference() {
    for depth in [2, 4] {
        let mut analytic = fixture(depth);
        analytic.forward_train_chunk(&IDS, &TARGETS);
        analytic.zero_gradients();
        analytic.backward_chunk(IDS.len(), 1.0);

        let embed_coordinate = strongest_coordinate(&analytic.embed_w.grad, "embedding");
        let embed = analytic.embed_w.grad[embed_coordinate];
        let numeric = numeric_shared(depth, true, embed_coordinate, analytic.embed_w.data[embed_coordinate]);
        assert_fd(&format!("depth{depth}/embed[{embed_coordinate}]"), embed, numeric);

        let head_coordinate = strongest_coordinate(&analytic.unembed_w.grad, "unembedding");
        let head = analytic.unembed_w.grad[head_coordinate];
        let numeric = numeric_shared(depth, false, head_coordinate, analytic.unembed_w.data[head_coordinate]);
        assert_fd(&format!("depth{depth}/head[{head_coordinate}]"), head, numeric);

        for layer in 0..depth {
            for family in FAMILIES {
                let coordinate = strongest_coordinate(
                    family_grad(block(&analytic, layer), family),
                    &format!("depth{depth}/layer{layer}/{family:?}"),
                );
                let analytic_derivative = family_grad(block(&analytic, layer), family)[coordinate];
                let original = family_data(block(&analytic, layer), family)[coordinate];
                let numeric = numeric_layer(depth, layer, family, coordinate, original);
                println!(
                    "FD depth={depth} layer={layer} family={family:?} coordinate={coordinate} analytic={analytic_derivative:.9} numeric={numeric:.9}"
                );
                assert_fd(
                    &format!("depth{depth}/layer{layer}/{family:?}[{coordinate}]"),
                    analytic_derivative,
                    numeric,
                );
            }
        }
    }
}

fn assert_matrix_state(left: &ParamMatrix, right: &ParamMatrix, label: &str) {
    assert_eq!(left.data, right.data, "{label}.data");
    assert_eq!(left.grad, right.grad, "{label}.grad");
    assert_eq!(left.m, right.m, "{label}.m");
    assert_eq!(left.v, right.v, "{label}.v");
}
fn assert_vector_state(left: &ParamVector, right: &ParamVector, label: &str) {
    assert_eq!(left.data, right.data, "{label}.data");
    assert_eq!(left.grad, right.grad, "{label}.grad");
    assert_eq!(left.m, right.m, "{label}.m");
    assert_eq!(left.v, right.v, "{label}.v");
}
fn assert_block_state(left: &PSSAContinuousBlockV2, right: &PSSAContinuousBlockV2, label: &str) {
    assert_vector_state(&left.norm_gamma, &right.norm_gamma, &format!("{label}.gamma"));
    assert_vector_state(&left.norm_beta, &right.norm_beta, &format!("{label}.beta"));
    for (a, b, name) in [
        (&left.a_mat, &right.a_mat, "a"), (&left.w_delta, &right.w_delta, "delta"),
        (&left.w_b, &right.w_b, "b"), (&left.w_c, &right.w_c, "c"),
        (&left.w_qx, &right.w_qx, "qx"), (&left.w_qh, &right.w_qh, "qh"),
        (&left.w_gate, &right.w_gate, "gate"), (&left.w_proj, &right.w_proj, "projection"),
        (&left.mlp_w1, &right.mlp_w1, "mlp1"), (&left.mlp_w2, &right.mlp_w2, "mlp2"),
        (&left.adapters[0].down_proj, &right.adapters[0].down_proj, "adapter_down"),
        (&left.adapters[0].up_proj, &right.adapters[0].up_proj, "adapter_up"),
    ] { assert_matrix_state(a, b, &format!("{label}.{name}")); }
    assert_eq!(left.adapters[0].consolidated_up, right.adapters[0].consolidated_up, "{label}.adapter_slow");
    assert_eq!(left.h_persistent, right.h_persistent, "{label}.carry");
    assert_eq!(left.memory.count, right.memory.count, "{label}.memory_count");
    assert_eq!(left.memory.write_head, right.memory.write_head, "{label}.memory_head");
    assert_eq!(left.memory.keys, right.memory.keys, "{label}.memory_keys");
    assert_eq!(left.memory.values, right.memory.values, "{label}.memory_values");
    assert_eq!(left.memory.norm_sq, right.memory.norm_sq, "{label}.memory_norms");
    assert_eq!(left.memory.confidence, right.memory.confidence, "{label}.memory_confidence");
    assert_eq!(left.memory.last_seen_step, right.memory.last_seen_step, "{label}.memory_seen");
}
fn assert_persisted_state(left: &PSSALayerV2, right: &PSSALayerV2) {
    assert_eq!(left.cfg.d_vocab, right.cfg.d_vocab);
    assert_eq!(left.cfg.d_latent, right.cfg.d_latent);
    assert_eq!(left.cfg.d_state, right.cfg.d_state);
    assert_eq!(left.cfg.d_mem_key, right.cfg.d_mem_key);
    assert_eq!(left.cfg.mem_capacity, right.cfg.mem_capacity);
    assert_eq!(left.cfg.chunk_len, right.cfg.chunk_len);
    assert_eq!(left.cfg.lr, right.cfg.lr);
    assert_eq!(left.cfg.beta1, right.cfg.beta1);
    assert_eq!(left.cfg.beta2, right.cfg.beta2);
    assert_eq!(left.cfg.weight_decay, right.cfg.weight_decay);
    assert_eq!(left.cfg.eps, right.cfg.eps);
    assert_eq!(left.cfg.tau_mem, right.cfg.tau_mem);
    assert_eq!(left.cfg.ema_alpha, right.cfg.ema_alpha);
    assert_eq!(left.depth(), right.depth());
    assert_eq!(left.step_counter, right.step_counter);
    assert_eq!(left.rng.state, right.rng.state);
    assert_eq!(left.vocabulary, right.vocabulary);
    assert_eq!(left.tokenizer_json, right.tokenizer_json);
    assert_eq!(left.residual_scales, right.residual_scales);
    assert_eq!(left.embed_row_marks, right.embed_row_marks);
    assert_matrix_state(&left.embed_w, &right.embed_w, "embed");
    assert_matrix_state(&left.unembed_w, &right.unembed_w, "head");
    for layer in 0..left.depth() { assert_block_state(block(left, layer), block(right, layer), &format!("block{layer}")); }
}

fn checkpoint_path(label: &str, depth: usize) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("pssa-stacked-acceptance-{label}-{depth}-{}", std::process::id()))
}

#[test]
fn v8_depth_two_and_four_round_trip_live_state_and_match_the_next_optimizer_update() {
    for depth in [2, 4] {
        let mut original = fixture(depth);
        original.forward_train_chunk(&IDS, &TARGETS);
        original.zero_gradients();
        original.backward_chunk(IDS.len(), 1.0);
        original.insert_training_memory(4.0, IDS.len());
        original.apply_adamw(0.01);
        let path = checkpoint_path("next-update", depth);
        save_model(&original, &path).unwrap();
        let mut restored = load_checkpoint(&path).unwrap();
        assert_eq!(restored.format, CheckpointFormat::V8);
        assert_persisted_state(&original, &restored.model);
        let resaved = checkpoint_path("resaved", depth);
        save_model(&restored.model, &resaved).unwrap();
        assert_eq!(fs::read(&path).unwrap(), fs::read(&resaved).unwrap(), "depth{depth} V8 bytes");
        fs::remove_file(resaved).unwrap();

        // This forward starts from the carried recurrence and banks above.  Thus
        // matching the following step checks restored moments as well as data.
        for model in [&mut original, &mut restored.model] {
            model.forward_train_chunk(&[5, 3, 1], &[7, 5, 3]);
            model.zero_gradients();
            model.backward_chunk(3, 1.0);
            model.apply_adamw(0.01);
        }
        assert_persisted_state(&original, &restored.model);
        fs::remove_file(path).unwrap();
    }
}

fn assert_close_vectors(label: &str, left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len(), "{label}: length");
    for (i, (&a, &b)) in left.iter().zip(right).enumerate() {
        let limit = 3.0e-6 + 3.0e-6 * a.abs().max(b.abs());
        assert!((a - b).abs() <= limit, "{label}[{i}]: {a} versus {b}");
    }
}
fn carries(model: &PSSALayerV2) -> Vec<Vec<f32>> {
    (0..model.depth()).map(|layer| block(model, layer).h_persistent.clone()).collect()
}

#[test]
fn depth_two_and_four_chunk_split_and_repeated_inference_match_unsplit_logits_and_carries() {
    for depth in [2, 4] {
        let mut unsplit = fixture(depth);
        unsplit.forward_train_chunk(&IDS, &TARGETS);
        let full_logits = unsplit.tape.logits.clone();
        let full_carries = carries(&unsplit);

        let mut split = fixture(depth);
        split.forward_train_chunk(&IDS[..1], &TARGETS[..1]);
        let first_logits = split.tape.logits[..split.cfg.d_vocab].to_vec();
        split.forward_train_chunk(&IDS[1..], &TARGETS[1..]);
        let mut joined = first_logits;
        joined.extend_from_slice(&split.tape.logits[..2 * split.cfg.d_vocab]);
        assert_close_vectors(&format!("depth{depth} split logits"), &full_logits, &joined);
        for (layer, expected) in full_carries.iter().enumerate() {
            assert_close_vectors(&format!("depth{depth} split carry layer{layer}"), expected, &block(&split, layer).h_persistent);
        }

        let mut inference = fixture(depth);
        let mut repeated = Vec::new();
        let mut logits = vec![0.0; inference.cfg.d_vocab];
        for &id in &IDS { inference.forward_inference(id, &mut logits); repeated.extend_from_slice(&logits); }
        assert_close_vectors(&format!("depth{depth} inference logits"), &full_logits, &repeated);
        for (layer, expected) in full_carries.iter().enumerate() {
            assert_close_vectors(&format!("depth{depth} inference carry layer{layer}"), expected, &block(&inference, layer).h_persistent);
        }
    }
}

#[test]
fn stacked_layers_have_independent_state_backing_and_mutating_one_does_not_change_another() {
    let mut model = fixture(4);
    for left in 0..model.depth() {
        for right in left + 1..model.depth() {
            assert_ne!(block(&model, left).h_persistent.as_ptr(), block(&model, right).h_persistent.as_ptr());
            assert_ne!(block(&model, left).memory.values.as_ptr(), block(&model, right).memory.values.as_ptr());
            assert_ne!(block(&model, left).w_delta.data.as_ptr(), block(&model, right).w_delta.data.as_ptr());
            assert_ne!(block(&model, left).tape.x_raw.as_ptr(), block(&model, right).tape.x_raw.as_ptr());
        }
    }
    let prior_carry = model.block.h_persistent.clone();
    let prior_value = model.block.memory.values[0];
    let prior_parameter = model.block.w_delta.data[0];
    model.extra_blocks[1].h_persistent[0] += 7.0;
    model.extra_blocks[1].memory.values[0] -= 3.0;
    model.extra_blocks[1].w_delta.data[0] += 2.0;
    assert_eq!(model.block.h_persistent, prior_carry);
    assert_eq!(model.block.memory.values[0], prior_value);
    assert_eq!(model.block.w_delta.data[0], prior_parameter);
}

fn fnv1a64(payload: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in payload { hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3); }
    hash
}
fn reseal(bytes: &mut [u8]) {
    let checksum = fnv1a64(&bytes[22..]).to_le_bytes();
    bytes[14..22].copy_from_slice(&checksum);
}
fn u64_at(bytes: &[u8], offset: usize) -> usize { usize::try_from(u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())).unwrap() }
fn skip_floats(bytes: &[u8], offset: usize) -> usize { let count = u64_at(bytes, offset); offset + 8 + count * 4 }

/// Minimal layout walk over a known-good, empty-vocabulary V8 test checkpoint.
/// It finds the embedded endpoint's Adam-v payload after validating all variable
/// prefixes instead of relying on a brittle hard-coded offset for that tensor.
fn embed_v_data_offset(bytes: &[u8]) -> usize {
    let payload = 22;
    let mut offset = payload + 6 * 8 + 7 * 4;
    let depth = u64_at(bytes, offset);
    offset += 8;
    assert_eq!(u64_at(bytes, offset), depth - 1, "unexpected scale length");
    offset += 8 + (depth - 1) * 4;
    offset += 8 + 8; // step counter and RNG state
    let vocab_count = u64_at(bytes, offset);
    offset += 8;
    assert_eq!(vocab_count, 0, "test fixture must retain an empty vocabulary");
    for component in 0..4 {
        let vector_start = offset;
        offset = skip_floats(bytes, offset);
        if component == 3 { return vector_start + 8; }
    }
    unreachable!("matrix has four data/grad/m/v vectors")
}

fn assert_rejected_with(path: &std::path::Path, expected: &str) {
    let error = match load_checkpoint(path) {
        Ok(_) => panic!("expected malformed V8 rejection containing {expected:?}"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains(expected), "expected {expected:?}, got {error:?}");
    assert!(!error.contains("checksum"), "forgery was not correctly resealed: {error}");
}

#[test]
fn resealed_v8_truncation_impossible_allocation_and_negative_moment_report_structural_reasons() {
    let model = fixture(2);
    let path = checkpoint_path("malformed", 2);
    save_model(&model, &path).unwrap();
    let good = fs::read(&path).unwrap();

    let mut impossible_allocation = good.clone();
    impossible_allocation[22..30].copy_from_slice(&u64::MAX.to_le_bytes()); // d_vocab
    reseal(&mut impossible_allocation);
    fs::write(&path, &impossible_allocation).unwrap();
    assert_rejected_with(&path, "overflow calculating");

    let mut negative_moment = good.clone();
    let moment = embed_v_data_offset(&negative_moment);
    negative_moment[moment..moment + 4].copy_from_slice(&(-1.0f32).to_le_bytes());
    reseal(&mut negative_moment);
    fs::write(&path, &negative_moment).unwrap();
    assert_rejected_with(&path, "negative embed_w.v");

    // Preserve a self-consistent container header/checksum after removing the
    // V8 tail.  The reader must identify a payload truncation, not integrity.
    let mut truncated = good[..good.len() - 4].to_vec();
    let payload_length = u64::try_from(truncated.len() - 22).unwrap();
    truncated[6..14].copy_from_slice(&payload_length.to_le_bytes());
    reseal(&mut truncated);
    fs::write(&path, &truncated).unwrap();
    assert_rejected_with(&path, "truncated");
    fs::remove_file(path).unwrap();
}
