# Reproducible Rust builds

The repository pins **Rust 1.90.0** with the minimal profile in `rust-toolchain.toml`. Keep `Cargo.lock`; use `--locked` for every comparison build.

## First setup

Install rustup through the official instructions at https://rustup.rs/ for your operating system. From this repository, rustup installs the pinned toolchain when needed. Subsequent builds reuse that installation when its directories are retained.

```sh
cargo --version
cargo test --locked --release --lib --tests
cargo test --locked --release --example width_depth_study
```

## Research acceptance settings (POSIX shell)

```sh
export CARGO_BUILD_JOBS=1
export CARGO_PROFILE_RELEASE_LTO=false
export RUSTFLAGS='-Dwarnings -C target-cpu=native'
cargo test --locked --release -j1 --lib --tests -- --nocapture
cargo test --locked --release -j1 --example width_depth_study
cargo build --locked --release -j1 --example width_depth_study
```

These are the current CPU comparison settings; do not mix optimization settings between configurations. `target-cpu=native` binaries may not work on another CPU. For distributed binaries, build separately for the intended platform without that flag.

## Avoid repeated installation and compilation

Retain `$RUSTUP_HOME` (normally `~/.rustup`), `$CARGO_HOME` (normally `~/.cargo`), and `target/` or `$CARGO_TARGET_DIR` between sessions. On ephemeral CI, restore trusted caches keyed by OS, architecture, Rust version, Cargo.lock, and build flags. Cargo's registry/git directories cache dependencies; retaining build output avoids recompiling unchanged dependencies. Cache misses still require downloading/installing dependencies.

Do **not** commit installed compiler directories or `target/` into Git. A Rust installation is specific to its host OS/CPU and contains large compiler/runtime files; cloning those files would not perform a portable installation. A compatible prebuilt program can run inference without Rust, but compiling new source or executing Rust tests still requires the toolchain. A prebuilt development container or persistent build machine is the solution when zero per-session installation is required.

## Current acceptance status

See `STACKED-WIP.md`. The latest all-target release acceptance run passed 56 tests with no failures. The controlled four-configuration training comparison is not complete, and passing correctness tests is not a claim that coherent language generation works. Do not redesign the CLI or implement new WGPU functionality before that language-quality gate.

For the complete gate, use `cargo test --locked --release -j1 --all-targets -- --nocapture` with the settings above. Before launching the frozen corpus study, run `bun scripts/verify-preflight.ts STUDY_ROOT RUNNER_BINARY` against the prepared private manifests, preflight outputs and SHA-256 provenance. This separate strict validator enforces the protocol; the Rust runner remains parameterized so small deterministic resume tests are possible.
