# Continuous block extraction: tested baseline for depth experiments

The token model now owns one embedding, one vocabulary head, its endpoint tape and optimizer clock. `PSSAContinuousBlockV2` independently owns affine RMS normalization, SSM, detached episodic memory, adapter, MLP, recurrent carry and a vocabulary-free continuous tape. Forward consumes raw continuous rows; backward consumes arbitrary output adjoints and returns raw-input adjoints. Only the outer model scatters these into embeddings.

Validation on Rust1.90.0 with locked dependencies, one build job, release opt3, native CPU target, warnings denied and LTO disabled: **41 tests passed, zero failed**. New tests cover arbitrary external-adjoint finite differences, populated-memory gradients, chunk carry/detachment, endpoint ownership, and actual zero-allocation hot paths. Finite-difference tolerance is 5e-6 absolute plus2% relative. Existing regression assertions remain intact.

External golden checks reproduced initialization and two BPE training epochs byte-for-byte against the pre-extraction executable, plus two further optimizer updates from a checkpoint containing four memory entries. Greedy and sampled inference also matched byte-for-byte. V7 on-disk field order is unchanged. Loader allocation accounting includes the new buffers.

This is still a **single-block model**. Width64 held-out validation perplexity reached114.80 after three completed training passes in the earlier control; unselected outputs remained repetitive and incoherent. Neither these tests nor lower perplexity establish coherent language.

Next: validate genuine residual stacking, then compare width64/depth1, width256/depth1, width64/depth2 and width64/depth4 using identical data, tokenizer, schedule and scored-token exposure. Report gradient diagnostics and compute costs along with all fixed generations. CLI redesign and new GPU support are deferred until language quality is demonstrated.

```sh
CARGO_BUILD_JOBS=1 CARGO_PROFILE_RELEASE_LTO=false \
RUSTFLAGS="-Dwarnings -C target-cpu=native" \
cargo +1.90.0 test --locked --release -j1
```
