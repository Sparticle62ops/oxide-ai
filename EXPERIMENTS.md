# Reproducible language-model diagnostics

Software correctness tests and low training loss do not demonstrate coherent general-language generation. Use whole-document train/validation/test splits, fit BPE only on training text, select checkpoints on validation, and keep every predeclared generation sample.

## Durable training

```sh
cargo build --locked -j 1 --release
mkdir -p runs
./target/release/oxide_ai_pssa train /path/to/train.txt \
  --tokenizer bpe --vocab-size 2048 --latent 64 --state 8 --key 16 \
  --memory 32 --chunk 64 --lr 0.001 --accumulate 4 --warmup-steps 20 \
  --epochs 4 --max-tokens 32000 --seed 42 \
  --run-dir runs/pilot-01 --out runs/pilot-01/final.pssa
```

`--run-dir` must not exist. It stores the initial model, every completed epoch, configuration, and atomic epoch metrics. The final output must not use reserved artifact names (`run.json`, `metrics.json`, or `epoch-*.pssa`) in that directory. Save stdout/stderr separately. Write errors fail the command; earlier completed epoch artifacts remain available. Atomic replacement prevents partially written JSON from replacing its prior version; this is not a guarantee against every filesystem or power-loss failure.

The cap is 32,000 **encoded prefix tokens per epoch**, shared across documents and reused over four epochs—not 32,000 transitions across the entire run. Each document has one fewer scored next-token transition than encoded tokens. Tokenizer fitting still sees all supplied training text. Checkpoint files contain model, optimizer, RNG, memory and tokenizer state, but these artifacts **do not enable CLI training-job resumption**.

## Evaluation and unselected generation

```sh
cargo build --locked -j 1 --release --example language_probe
./target/release/examples/language_probe evaluate \
  runs/pilot-01/epoch-0001.pssa /path/to/validation.txt > validation.json
./target/release/examples/language_probe generate \
  runs/pilot-01/epoch-0001.pssa /path/to/prompts.json > generations.json
```

The probe requires a self-contained BPE checkpoint. Evaluation reports natural-log cross entropy and perplexity per BPE transition, encoded input tokens, scored transitions, OOV denominator, and accuracy. Recurrent state resets per document; frozen training memory is retained. BPE and word perplexities are not directly comparable units.

Prompt input format:

```json
{
  "sampling": {"temperatures": [0.0, 0.7], "max_new_tokens": 64, "seed": 1337},
  "prompts": [{"text": "A computer program can"}]
}
```

The probe retains every prompt/temperature combination, with fresh fixed seed 1337 per sample. It disables repetition penalties and vocabulary/nucleus truncation, except the existing generator's `<unk>` exclusion. Unlike the interactive CLI defaults, it does not use top-k 24, top-p 0.85, or repetition penalty 1.25. The current BPE streaming decoder can omit an incomplete final UTF-8 suffix at the length cutoff; this known limitation is reported in sample metadata.

Assess grammar, relevance, repetition, internal consistency, and verbatim training overlap separately. Do not select attractive samples, conceal failed experiments, or interpret zero OOV as language understanding. A small single-layer pilot is diagnostic, not a general-language capability claim. No corpus is included in this document; establish provenance/licensing before redistributing training data.
