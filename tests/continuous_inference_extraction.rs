use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};

fn cfg() -> PSSAConfigV2 {
    PSSAConfigV2 {
        d_vocab: 11,
        d_latent: 5,
        d_state: 3,
        d_mem_key: 3,
        mem_capacity: 4,
        chunk_len: 4,
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        tau_mem: 0.5,
        ema_alpha: 0.1,
    }
}

fn raw_embedding(m: &PSSALayerV2, id: usize) -> Vec<f32> {
    // The continuous block owns affine RMS normalization; pass the raw row.
    let d = m.cfg.d_latent;
    m.embed_w.data[id * d..(id + 1) * d].to_vec()
}

#[test]
fn continuous_inference_matches_legacy_token_endpoint_and_recurrent_carry() {
    let mut endpoint = PSSALayerV2::new(cfg(), 91);
    let mut continuous = PSSALayerV2::new(cfg(), 91);
    let mut logits = vec![0.0; endpoint.cfg.d_vocab];
    let mut z = vec![0.0; endpoint.cfg.d_latent];
    for id in [1usize, 7, 3] {
        let x = raw_embedding(&continuous, id);
        continuous.block.forward_continuous_inference(&x, &mut z);
        endpoint.forward_inference(id, &mut logits);
        let mut expected = vec![0.0; endpoint.cfg.d_vocab];
        endpoint.unembed_w.matvec(&z, &mut expected);
        let scale = 1.0 / (endpoint.cfg.d_latent as f32).sqrt();
        for (got, want) in logits.iter().zip(expected.iter()) {
            assert_eq!(*got, *want * scale, "id {id}");
        }
        assert_eq!(continuous.block.h_persistent, endpoint.block.h_persistent, "id {id}");
    }
}

#[test]
fn continuous_inference_warmed_path_reuses_caller_buffers() {
    let mut m = PSSALayerV2::new(cfg(), 92);
    let x = raw_embedding(&m, 2);
    let mut z = vec![0.0; m.cfg.d_latent];
    m.block.forward_continuous_inference(&x, &mut z);
    let z_ptr = z.as_ptr();
    m.reset_recurrent_state();
    m.block.forward_continuous_inference(&x, &mut z);
    assert_eq!(z_ptr, z.as_ptr());
    assert!(z.iter().all(|v| v.is_finite()));
}
