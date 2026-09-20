//! Deterministic checkpoint-continuation probe, not a language-quality benchmark.
use oxide_ai_pssa::checkpoint;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: continuation_probe INPUT_CHECKPOINT OUTPUT_CHECKPOINT".into());
    }
    let mut model = checkpoint::load_checkpoint(&args[1])?.model;
    let populated_memory = model.block.memory.count;
    let len = model.cfg.chunk_len.min(3);
    let input: Vec<usize> = (0..len).map(|i| (i + 1) % model.cfg.d_vocab).collect();
    let target: Vec<usize> = (0..len).map(|i| (i + 2) % model.cfg.d_vocab).collect();
    let mut losses = Vec::new();
    // Preserve checkpoint carry, bank and moments across two further updates.
    for _ in 0..2 {
        model.zero_gradients();
        losses.push(model.forward_train_chunk(&input, &target));
        model.backward_chunk(len, 1.0);
        model.apply_adamw(0.00037);
    }
    checkpoint::save_model(&model, &args[2])?;
    println!("{}", json!({"memory_count_on_load":populated_memory,"losses":losses,"optimizer_step":model.step_counter}));
    Ok(())
}
