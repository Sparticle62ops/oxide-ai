# Stacked-model research status — 2026-09-17

This remains a development branch, **not a completed language-quality result**.

## Executed acceptance gate

Rust 1.90.0; locked dependencies; native release optimizations; one Cargo job; LTO disabled; warnings denied.

```sh
CARGO_BUILD_JOBS=1 CARGO_PROFILE_RELEASE_LTO=false \
RUSTFLAGS='-Dwarnings -C target-cpu=native' \
cargo test --locked --release -j1 --all-targets -- --nocapture
```

Latest outcome: **56 passed, 0 failed**, across 18 executable targets (including targets without tests). The separate release research-runner build also exited successfully.

This supersedes the prior failing snapshot, but preserves its failures in the private audit trail:
- The runner mixed optimizer-group indices with chunk indices, producing an empty group before completing its own test. Both range endpoints now use chunk coordinates; a coverage test checks every chunk exactly once, including partial tails.
- Exact frozen numeric controls changed under the JSON parser's default float round trip. Enabling `serde_json/float_roundtrip` fixes the evidenced one-bit beta2 drift without weakening provenance comparisons.
- Several old tests and the performance example still referenced pre-extraction fields. Their migrations are field-path changes only.
- A shared additive output-head offset did not strengthen the gradient fixture because softmax cancels common offsets. The accepted fixture instead increases class-to-class head contrast. Original finite-difference tolerances and the strict non-vacuity threshold are unchanged.

Additional acceptance covers every trainable coordinate in small depth-2/depth-4 populated-memory fixtures, alongside the original live per-family checks. It is not an exhaustive test at production widths or every supported depth.

## Runner integrity

Periodic checkpoints are retained rather than pruned. Initial validation is recorded. Resume state verifies exposure and update counts against the dataset cursor and rejects escaping checkpoint paths. Rejected incompatible invocations do not overwrite an existing experiment. Preflight provenance binds subsequent training. Completed-run checks require the final checkpoint to match the authoritative final epoch checkpoint.

Uninterrupted and segmented runs are byte-identical at depths 1, 2 and 4, with matching nontiming metrics. Tests also exercise partial accumulation groups, retained recovery files, changed input/options rejection, preflight isolation, and corrupted state counters.

## Controlled study

All four configurations preflighted successfully: width64/depth1, width256/depth1, width64/depth2, width64/depth4. Each has 840951 scored transitions per pass, four passes, and 13184 total optimizer updates. Corpus and tokenizer SHA-256 provenance is recorded externally; the initial tokenizer source is verified as the width64, step-zero V7 control.

The runner itself is manifest-driven and supports small test configurations. `scripts/verify-preflight.ts` is the separate strict frozen-study gate: it checks exact controls, source identity, shared fingerprints, SHA-256 records, and fixed generation policy before the long comparison. Run it with Bun and the private study root; no private corpus or complete recovery files are supplied in this repository.

Only a bounded 64-update slice of the fresh baseline has run so far. It is resumable under the unchanged four-pass schedule; no completed comparison or coherent-language claim exists. The test split remains unused for selection.

## Next gate

Complete the four training configurations sequentially, compare final-epoch validation and every unselected generation, and replicate any apparent improvement. CLI redesign and new WGPU work remain blocked until reproducible coherent/logical language is demonstrated. Do not merge the draft PR merely because correctness tests pass.
