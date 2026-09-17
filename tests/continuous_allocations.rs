//! Allocation proof for warmed direct continuous-block paths.
//!
//! This is a separate integration-test binary so its global allocator is not
//! shared with unrelated allocation tests.  The counters are thread-local: test
//! harness activity on other threads cannot affect the assertion.

use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct TestThreadAllocator;

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static REALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record(reallocation: bool) {
    ACTIVE.with(|active| {
        if active.get() {
            let counter = if reallocation { &REALLOCATIONS } else { &ALLOCATIONS };
            counter.with(|count| count.set(count.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for TestThreadAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        record(false);
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        record(true);
        new_pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: TestThreadAllocator = TestThreadAllocator;

fn warm_counters() {
    ACTIVE.with(|value| value.set(false));
    ALLOCATIONS.with(|value| value.set(0));
    REALLOCATIONS.with(|value| value.set(0));
}

fn begin_counting() {
    ALLOCATIONS.with(|value| value.set(0));
    REALLOCATIONS.with(|value| value.set(0));
    ACTIVE.with(|value| value.set(true));
}

fn finish_counting() -> (usize, usize) {
    ACTIVE.with(|value| value.set(false));
    (
        ALLOCATIONS.with(Cell::get),
        REALLOCATIONS.with(Cell::get),
    )
}

fn config() -> PSSAConfigV2 {
    PSSAConfigV2 {
        d_vocab: 7,
        d_latent: 8,
        d_state: 3,
        d_mem_key: 3,
        mem_capacity: 3,
        chunk_len: 3,
        lr: 1.0e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.0,
        eps: 1.0e-8,
        tau_mem: 0.7,
        ema_alpha: 0.1,
    }
}

#[test]
fn warmed_continuous_forward_backward_and_inference_do_not_allocate() {
    // All storage used by the tested calls—including direct output and input
    // VJP buffers—is allocated before the measurement begins.
    let mut model = PSSALayerV2::new(config(), 417);
    let block = &mut model.block;
    for (i, value) in block.mlp_w2.data.iter_mut().enumerate() {
        *value = 0.02 * (i as f32 % 5.0 - 2.0);
    }
    for (i, value) in block.adapters[0].up_proj.data.iter_mut().enumerate() {
        *value = 0.015 * (i as f32 % 7.0 - 3.0);
    }
    let key_a = [0.08, -0.04, 0.03];
    let key_b = [-0.07, 0.05, 0.11];
    let value_a = [0.10, -0.08, 0.06, -0.04, 0.02, -0.01, 0.03, -0.05];
    let value_b = [-0.03, 0.07, -0.09, 0.12, -0.06, 0.04, 0.08, -0.02];
    block.memory.insert(&key_a, &value_a);
    block.memory.insert(&key_b, &value_b);

    let raw = [
        0.12, -0.21, 0.17, 0.05, -0.11, 0.29, -0.07, 0.13,
        -0.16, 0.09, 0.25, -0.19, 0.04, 0.14, -0.08, 0.22,
        0.31, -0.18, 0.06, 0.15, -0.27, 0.10, 0.03, -0.12,
    ];
    let output_adjoints = [
        0.2, -0.1, 0.3, -0.2, 0.1, 0.4, -0.3, 0.2,
        -0.2, 0.5, -0.1, 0.3, 0.2, -0.4, 0.1, -0.3,
        0.1, 0.2, -0.5, 0.3, -0.2, 0.4, -0.1, 0.2,
    ];
    let mut input_adjoints = [0.0; 24];
    let mut inference_output = [0.0; 8];

    // Warm every direct continuous path and the thread-local counters before
    // activation.  Nothing in the measured loop is constructed by test code.
    warm_counters();
    block.reset_recurrent_state();
    block.forward_train_chunk(&raw, 3);
    block.zero_gradients();
    block.backward_chunk(&output_adjoints, 3, &mut input_adjoints);
    block.reset_recurrent_state();
    block.forward_continuous_inference(&raw[..8], &mut inference_output);

    begin_counting();
    for _ in 0..8 {
        block.reset_recurrent_state();
        block.forward_train_chunk(&raw, 3);
        block.zero_gradients();
        block.backward_chunk(&output_adjoints, 3, &mut input_adjoints);
        block.reset_recurrent_state();
        block.forward_continuous_inference(&raw[..8], &mut inference_output);
    }
    let (allocations, reallocations) = finish_counting();

    assert_eq!(allocations, 0, "warmed continuous paths allocated {allocations} times");
    assert_eq!(
        reallocations, 0,
        "warmed continuous paths reallocated {reallocations} times"
    );
}
