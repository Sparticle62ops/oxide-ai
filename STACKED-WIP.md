# Stacked-model development snapshot — 2026-09-17

This is a recovery snapshot, **not an accepted or completed width/depth experiment**.

## Included work
- Continuous residual stacks, shared embedding/head, independent per-layer state and Adam moments.
- V8 stacked checkpoints with V7 retained for depth one.
- Cross-layer acceptance, state-alias, checkpoint and allocation tests.
- Resumable width/depth research runner (not a CLI redesign).

## Latest verified gate
Rust 1.90.0, locked dependencies, release opt-level 3, one Cargo job, LTO disabled, `RUSTFLAGS='-Dwarnings -C target-cpu=native'`.

The new `stacked_acceptance` suite now compiles after separating checksum calculation from the mutable write. Latest execution: **4 passed, 1 failed**. State independence, forward/inference equivalence, malformed checkpoint rejection, and V8 optimizer continuation passed. The finite-difference test stopped on a fixture non-vacuity assertion: depth4/layer1/Gate[5] analytic 0.000028169 versus numerical 0.000028047, below the required derivative magnitude of 0.00003. This is not evidence of a gradient mismatch; the fixture needs a stronger signal before the full finite-difference gate can be accepted. Do not lower the threshold to hide it.

The earlier complete suite passed 47 tests before these newest tests. It has not yet been rerun on this snapshot. Study-runner unit tests have not run. No controlled corpus training has started; coherent logical language remains unachieved.

## Remaining runner work
- Stop pruning periodic checkpoints; preserve all recovery artifacts.
- Validate the four run manifests against the frozen protocol before launching.
- Record SHA-256 provenance externally in addition to the runner's restart fingerprints.
- Report compute and end-to-end costs explicitly.

## Resume order
1. Strengthen the live gradient fixture without changing numerical tolerances; run the focused acceptance gate.
2. Run all library/integration tests and the research-runner unit test.
3. Fix checkpoint-retention behavior and validate frozen provenance/transition counts.
4. Run width64/depth1, width256/depth1, width64/depth2, width64/depth4 with identical four-pass exposure (840951 transitions/pass), seed42, tokenizer/corpus, and schedule.
5. Review every retained generation. No CLI redesign or new WGPU implementation until coherent/logical language is demonstrated.

The user authorized publishing through the existing connection with Tasklet attribution on 2026-09-17. Retain explicit `sparticle62ops <sparticle62ops@users.noreply.github.com>` co-author credit. Original locally attributed commits remain in the private recovery bundle. Private transcripts, corpus data and complete recovery archives must not be published wholesale.
