use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
struct A;
thread_local! { static ON: Cell<bool>=const {Cell::new(false)}; static N: Cell<usize>=const {Cell::new(0)}; }
unsafe impl GlobalAlloc for A { unsafe fn alloc(&self,l:Layout)->*mut u8 {let p=unsafe{System.alloc(l)};ON.with(|x|if x.get(){N.with(|n|n.set(n.get()+1))});p} unsafe fn dealloc(&self,p:*mut u8,l:Layout){unsafe{System.dealloc(p,l)}} unsafe fn realloc(&self,p:*mut u8,l:Layout,n:usize)->*mut u8{let q=unsafe{System.realloc(p,l,n)};ON.with(|x|if x.get(){N.with(|c|c.set(c.get()+1))});q} }
#[global_allocator] static ALLOC:A=A;
fn cfg()->PSSAConfigV2 {PSSAConfigV2{d_vocab:7,d_latent:4,d_state:2,d_mem_key:2,mem_capacity:2,chunk_len:3,lr:0.01,beta1:0.9,beta2:0.999,weight_decay:0.0,eps:1e-8,tau_mem:0.7,ema_alpha:0.1}}
#[test]
fn warmed_stacked_training_inference_and_memory_insertion_do_not_allocate() {
 let mut m=PSSALayerV2::new_with_depth(cfg(),9,3);let ids=[1,2,3];let tar=[2,3,4];let mut logits=vec![0.0;7];
 m.forward_train_chunk(&ids,&tar);m.zero_gradients();m.backward_chunk(3,1.0);m.insert_training_memory(4.0,3);m.forward_inference(1,&mut logits);
 N.with(|n|n.set(0));ON.with(|x|x.set(true));
 for _ in 0..4 {m.reset_recurrent_state();m.forward_train_chunk(&ids,&tar);m.zero_gradients();m.backward_chunk(3,1.0);m.insert_training_memory(4.0,3);m.forward_inference(1,&mut logits);}
 ON.with(|x|x.set(false));assert_eq!(N.with(Cell::get),0,"warmed stacked hot paths allocated");
}
