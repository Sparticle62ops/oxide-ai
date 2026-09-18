use oxide_ai_pssa::checkpoint::{load_checkpoint, save_model, CheckpointFormat};
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use std::fs;

fn cfg() -> PSSAConfigV2 {
    PSSAConfigV2 { d_vocab: 9, d_latent: 4, d_state: 2, d_mem_key: 2, mem_capacity: 3, chunk_len: 3,
        lr: 0.01, beta1: 0.9, beta2: 0.999, weight_decay: 0.0, eps: 1e-8, tau_mem: 0.7, ema_alpha: 0.1 }
}
fn close(a: f32, b: f32) { assert!((a-b).abs() < 4e-5, "{a} != {b}"); }

#[test]
fn depth_one_stays_layout_compatible_and_depth_stack_livepaths_match() {
    let one = PSSALayerV2::new(cfg(), 41);
    let one_explicit = PSSALayerV2::new_with_depth(cfg(), 41, 1);
    assert_eq!(one.depth(), 1);
    assert_eq!(one.embed_w.data, one_explicit.embed_w.data);
    assert_eq!(one.block.w_delta.data, one_explicit.block.w_delta.data);
    assert_eq!(one.unembed_w.data, one_explicit.unembed_w.data);

    let mut train = PSSALayerV2::new_with_depth(cfg(), 41, 4);
    let mut infer = PSSALayerV2::new_with_depth(cfg(), 41, 4);
    let ids = [1, 2, 3]; let targets = [2, 3, 4];
    train.forward_train_chunk(&ids, &targets);
    let expected = train.tape.logits.clone();
    let mut logits = vec![0.0; cfg().d_vocab];
    for t in 0..ids.len() {
        infer.forward_inference(ids[t], &mut logits);
        for i in 0..logits.len() { close(logits[i], expected[t*logits.len()+i]); }
    }
    assert_eq!(train.extra_blocks.len(), 3);
    assert!(train.residual_scales.iter().all(|&x| x.to_bits() == (1.0_f32 / 2.0).to_bits()));
}

#[test]
fn depth_four_backpropagates_all_boundaries_and_layer_parameters() {
    let mut m=PSSALayerV2::new_with_depth(cfg(), 177, 4);
    // Make the normally zero MLP output and adapter output paths active, and give
    // each independently-owned bank a nonempty detached entry.
    for b in std::iter::once(&mut m.block).chain(m.extra_blocks.iter_mut()) {
        for x in &mut b.mlp_w2.data { *x=0.03; }
        for x in &mut b.adapters[0].up_proj.data { *x=0.02; }
        b.memory.insert(&[0.08, -0.04], &[0.1, -0.05, 0.07, 0.02]);
    }
    let ids=[1,3,5]; let targets=[3,5,7];
    m.forward_train_chunk(&ids,&targets); m.zero_gradients(); m.backward_chunk(3,1.0);
    assert_eq!(m.boundary_adjoints.len(), 5);
    for (i,boundary) in m.boundary_adjoints.iter().enumerate() {
        assert!(boundary[..12].iter().all(|x| x.is_finite()), "nonfinite boundary {i}");
        assert!(boundary[..12].iter().any(|x| x.abs()>1e-8), "vacuous boundary {i}");
    }
    assert!(m.block.norm_gamma.grad.iter().any(|x|x.abs()>1e-9));
    assert!(m.extra_blocks.first().unwrap().norm_gamma.grad.iter().any(|x|x.abs()>1e-9));
    assert!(m.extra_blocks.last().unwrap().norm_gamma.grad.iter().any(|x|x.abs()>1e-9));
    let before=m.parameter_count(); assert!(before>0);
    m.insert_training_memory(4.0,3);
    assert!(std::iter::once(&m.block).chain(m.extra_blocks.iter()).all(|b|b.memory.count==2));
    m.apply_adamw(0.01); assert_eq!(m.step_counter,1); assert!(m.all_parameters_finite());
}

#[test]
fn v8_round_trip_and_malformed_depth_or_scales_are_rejected_before_model_use() {
    let mut m=PSSALayerV2::new_with_depth(cfg(), 89, 2);
    let ids=[1,2,3]; let targets=[2,3,4]; m.forward_train_chunk(&ids,&targets);m.zero_gradients();m.backward_chunk(3,1.0);m.insert_training_memory(4.0,3);m.apply_adamw(0.01);
    let root=std::env::temp_dir().join(format!("pssa-v8-stack-{}",std::process::id())); let p=root.with_extension("pssa"); let again=root.with_extension("again.pssa");
    save_model(&m,&p).unwrap(); let loaded=load_checkpoint(&p).unwrap(); assert_eq!(loaded.format,CheckpointFormat::V8); assert_eq!(loaded.model.depth(),2); assert_eq!(loaded.model.step_counter,m.step_counter); assert_eq!(loaded.model.parameter_count(),m.parameter_count()); save_model(&loaded.model,&again).unwrap(); assert_eq!(fs::read(&p).unwrap(),fs::read(&again).unwrap());
    let bytes=fs::read(&p).unwrap();
    let mut bad_depth=bytes.clone(); let depth_offset=22+6*8+7*4; bad_depth[depth_offset..depth_offset+8].copy_from_slice(&33u64.to_le_bytes()); reseal(&mut bad_depth); fs::write(&p,&bad_depth).unwrap(); let error=match load_checkpoint(&p) { Ok(_) => panic!("corrupt depth was accepted"), Err(error) => error.to_string() }; assert!(error.contains("depth outside supported range"), "expected depth validation, got {error}"); assert!(!error.contains("checksum"), "forgery was not correctly resealed: {error}");
    let mut bad_scale=bytes; let first_scale=depth_offset+8+8; bad_scale[first_scale..first_scale+4].copy_from_slice(&0.25f32.to_le_bytes()); reseal(&mut bad_scale); fs::write(&p,&bad_scale).unwrap(); let error=match load_checkpoint(&p) { Ok(_) => panic!("corrupt residual scale was accepted"), Err(error) => error.to_string() }; assert!(error.contains("invalid residual scale"), "expected scale validation, got {error}"); assert!(!error.contains("checksum"), "forgery was not correctly resealed: {error}");
    let _=fs::remove_file(p);let _=fs::remove_file(again);
}
fn reseal(bytes:&mut [u8]) { let payload=&bytes[22..]; let mut h=0xcbf2_9ce4_8422_2325u64;for &b in payload {h=(h^u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);}bytes[14..22].copy_from_slice(&h.to_le_bytes()); }

#[derive(Copy,Clone)] enum Probe { Embed, Head, FirstNorm, LastNorm }
fn depth_two_fixture() -> PSSALayerV2 { let mut m=PSSALayerV2::new_with_depth(cfg(),313,2); for b in std::iter::once(&mut m.block).chain(m.extra_blocks.iter_mut()) { for x in &mut b.mlp_w2.data {*x=0.025;} for x in &mut b.adapters[0].up_proj.data {*x=0.017;} b.memory.insert(&[0.08,-0.04],&[0.10,-0.05,0.07,0.02]); b.memory.insert(&[-0.06,0.05],&[-0.04,0.09,-0.02,0.06]); } m }
fn get(m:&PSSALayerV2,p:Probe)->f32 {match p {Probe::Embed=>m.embed_w.data[4],Probe::Head=>m.unembed_w.data[0],Probe::FirstNorm=>m.block.norm_gamma.data[0],Probe::LastNorm=>m.extra_blocks[0].norm_gamma.data[0]}}
fn set(m:&mut PSSALayerV2,p:Probe,x:f32) {match p {Probe::Embed=>m.embed_w.data[4]=x,Probe::Head=>m.unembed_w.data[0]=x,Probe::FirstNorm=>m.block.norm_gamma.data[0]=x,Probe::LastNorm=>m.extra_blocks[0].norm_gamma.data[0]=x}}
fn grad(m:&PSSALayerV2,p:Probe)->f32 {match p {Probe::Embed=>m.embed_w.grad[4],Probe::Head=>m.unembed_w.grad[0],Probe::FirstNorm=>m.block.norm_gamma.grad[0],Probe::LastNorm=>m.extra_blocks[0].norm_gamma.grad[0]}}
#[test]
fn depth_two_shared_and_first_last_layer_gradients_match_nonvacuous_finite_differences() { let ids=[1,3,5];let targets=[3,5,7];for p in [Probe::Embed,Probe::Head,Probe::FirstNorm,Probe::LastNorm] {let mut a=depth_two_fixture();a.forward_train_chunk(&ids,&targets);a.zero_gradients();a.backward_chunk(3,1.0);let analytic=grad(&a,p);let x=get(&a,p);let h=0.002;let mut hi=depth_two_fixture();set(&mut hi,p,x+h);let high=hi.forward_train_chunk(&ids,&targets);let mut lo=depth_two_fixture();set(&mut lo,p,x-h);let low=lo.forward_train_chunk(&ids,&targets);let numeric=(high-low)/(2.0*h);let limit=5e-6+0.02*analytic.abs().max(numeric.abs());assert!(analytic.abs().max(numeric.abs())>1e-7,"vacuous depth-two finite difference");assert!((analytic-numeric).abs()<=limit,"analytic={analytic} numeric={numeric} limit={limit}");}}

#[test]
fn depth_two_tiny_repeated_corpus_smoke_learns() {
    let mut m=PSSALayerV2::new_with_depth(cfg(), 73, 2);
    let ids=[1,2,1];let targets=[2,1,2];let mut first=0.0;let mut last=0.0;
    for n in 0..100 { m.reset_recurrent_state(); let loss=m.forward_train_chunk(&ids,&targets); if n==0 {first=loss;} last=loss;m.zero_gradients();m.backward_chunk(3,1.0);m.apply_adamw(0.01); }
    assert!(last<first, "smoke learning did not improve: {first} -> {last}");
}
