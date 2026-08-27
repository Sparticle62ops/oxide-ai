# Oxide AI

Oxide AI is an experimental Rust implementation of a **Plastic State-Space Architecture (PSSA)** for continual language-model training and text generation. It combines a recurrent state-space layer with a hyperbolic episodic memory bank, modular low-rank adapters, plastic updates, and a closed-form ridge-regression consolidation step.

The repository is a research prototype rather than a production language model. It runs locally on the CPU, uses a deliberately small reference corpus for bootstrapping, and has no external model or ML framework dependency.

## Features

- Recurrent sequence processing with learned continuous state-space matrices.
- Token embeddings, unembeddings, normalization, SiLU feed-forward expansion, and autoregressive sampling.
- Hyperbolic/Poincare-inspired memory retrieval with bounded top-4 search.
- Plastic states for stable reinforcement, novelty-driven growth, refractory overwrite protection, and router tuning.
- Modular low-rank adapters for targeted updates.
- Refractory rate limiting intended to reduce damage from repeated contradictory updates.
- Ridge-regression consolidation from fast plastic updates into the base transition matrix.
- Binary `.pssa` model export/import.
- Dataset loading from local text, directories, URLs, Hugging Face datasets, and the built-in science corpus.
- A verification benchmark covering contradiction adaptation, distractor-gap recall, consolidation, spam mitigation, serialization, and generation.

## Status

This codebase is under active development. The checked-in source currently has no automated unit or integration tests. The CLI build should be treated as experimental; validate the current branch with `cargo check` before relying on it.

## Requirements

- Rust toolchain with Edition 2024 support, including Cargo.
- Network access only when using an HTTP/HTTPS dataset or Hugging Face dataset.
- Sufficient memory and disk space for larger downloaded corpora and serialized models.

The only direct runtime dependency is [`ureq`](https://crates.io/crates/ureq), used for HTTPS and HTTP dataset downloads. No GPU runtime is required. (Support for wGpu is under development!)

## Quick Start

Clone the repository and build it:

```bash
git clone https://github.com/Sparticle62ops/oxide-ai.git
cd oxide-ai
cargo build --release
```

The executable is written to `target/release/oxide_ai_pssa`. You can also use Cargo for every command:

```bash
cargo run -- help
```

Train a model using the bundled downloaded corpus:

```bash
cargo run -- train data/downloaded.txt --epochs 4 --out data/model.pssa
```

Start the interactive REPL:

```bash
cargo run -- chat data/downloaded.txt --model data/model.pssa --temp 0.70
```

Generate one completion:

```bash
cargo run -- generate "quantum mechanics" data/downloaded.txt --model data/model.pssa
```

If `data/model.pssa` does not exist, `chat` and `generate` automatically train a four-epoch model before loading it. Training time depends heavily on corpus size and CPU speed.

## CLI Reference

General form:

```text
oxide_ai_pssa <COMMAND> [OPTIONS]
```

Commands:

| Command | Purpose |
| --- | --- |
| `train [source]` | Train a PSSA model and write a `.pssa` file. |
| `chat [source]` or `repl [source]` | Launch an interactive conversational REPL. |
| `generate <prompt>` | Generate one completion from a prompt. |
| `download <repo>` | Download text from a Hugging Face dataset repository. |
| `benchmark` | Run the built-in milestone verification suite. |
| `help` | Print command and option help. |

Options:

| Option | Default | Applies to | Description |
| --- | --- | --- | --- |
| `-d, --data <source>` | `data/downloaded.txt` when present, otherwise `science` | `train`, `chat`, `generate` | Dataset source or comma-separated list of sources. |
| `-m, --model <path>` | `data/model.pssa` | `chat`, `generate` | Model to load. |
| `-e, --epochs <number>` | `4` | `train` | Number of training epochs. |
| `-p, --prompt <text>` | empty | `generate` | Prompt text. Required for generation. |
| `-t, --temp <float>` | `0.70` for chat, `0.25` for generation | `chat` | Sampling temperature. |
| `-o, --out <path>` | `data/model.pssa` | `train`, `download` | Output model or dataset path. |

Positional arguments and long/short options can be mixed. For example:

```bash
cargo run -- train data/downloaded.txt -e 2 -o data/experiment.pssa
cargo run -- train --data data/downloaded.txt --epochs 2 --out data/experiment.pssa
```

### Chat commands

Inside the REPL:

- `/exit` or `quit` exits the process.
- `/info` prints the loaded model path, memory slot count, and adapter count.
- `/temp <value>` reports a temperature value, but the current implementation does not apply that value to subsequent turns. Pass `--temp` when launching `chat` instead.

## Dataset Sources

`DatasetManager` accepts one or more comma-separated sources:

```bash
# Built-in reference corpus
cargo run -- train science --epochs 4

# Local text file
cargo run -- train data/downloaded.txt

# Every readable file in a directory
cargo run -- train data/

# Remote text or JSON-like response
cargo run -- train https://example.org/corpus.txt

# Hugging Face repository, loaded through the datasets server
cargo run -- train hf:owner/dataset

# Multiple sources
cargo run -- train science,data/downloaded.txt
```

The loader first tries to read local paths. A missing non-special source is treated as a Hugging Face repository name. Remote Hugging Face loading probes the datasets server and several conventional raw-file names. JSON-like responses are reduced using common fields such as `text`, `content`, `article`, `story`, `instruction`, `output`, `sentence`, and `summary`.

Tokenization lowercases text when used by the CLI, keeps alphanumeric words plus hyphens and apostrophes, and separates `.`, `,`, `?`, and `!` into individual tokens. Vocabularies are capped at 10,000 entries; rare tokens in larger corpora map to `<unk>`.

Download a Hugging Face dataset into a local text file:

```bash
cargo run -- download wikimedia/wikipedia --out data/downloaded.txt
```

Network downloads are not validated or curated by Oxide AI. Review licensing, privacy, and content before training on an external corpus.

## Training Pipeline

The `train` command performs two phases:

1. **Continuous recurrent ingestion:** token transitions are processed through the PSSA layer. The model updates state, memory, adapters, and routing behavior with a cosine learning-rate schedule.
2. **Ridge consolidation:** online transition statistics are accumulated and consolidated into the base matrix using closed-form ridge regression.

The default training configuration uses a vocabulary-sized input/output, latent dimension 256, recurrent state dimension 128, memory-key dimension 32, and up to 4,096 memory entries. The seed is fixed at `42` for model initialization.

The resulting binary contains model weights, configuration, memory, adapters, and related state. It is not a portable interchange format for other ML frameworks and should be loaded through `PSSALayer::import_from_pssa_bytes`.

## Inference

Generation is autoregressive and uses:

- Temperature scaling.
- A top-16 candidate limit followed by top-p filtering.
- Repetition penalty over a recent 64-token window.
- Immediate self-transition suppression.
- `<unk>` suppression.
- A maximum of 45 new tokens in the current CLI configuration, ending early after two generated periods.

The tokenizer used at inference time is rebuilt from the selected dataset. For useful results, use the same corpus or vocabulary source that was used to train the model. The CLI warns when the current dataset vocabulary size differs from the serialized model vocabulary.

## Benchmark

Run the internal milestone suite with:

```bash
cargo run -- benchmark
```

The suite exercises synthetic streams for contradictory facts, MQAR-style distractors, burst repetition, model serialization, and short generation prompts. It prints milestone results but is not currently wired into Cargo's test harness and should not be interpreted as a quality evaluation on general language tasks.

## Project Layout

| Path | Responsibility |
| --- | --- |
| `src/main.rs` | Binary entry point; forwards process arguments to the CLI. |
| `src/cli.rs` | Argument parsing, training, chat, generation, downloading, and benchmark orchestration. |
| `src/dataset.rs` | Tokenization, vocabulary construction, built-in/synthetic corpora, local and remote loading. |
| `src/pssa.rs` | PSSA layer, forward pass, plastic learning, consolidation, and `.pssa` serialization. |
| `src/inference.rs` | Autoregressive sampling and generation constraints. |
| `src/memory.rs` | Fixed-capacity hyperbolic memory bank and retrieval/update logic. |
| `src/adapter.rs` | Low-rank modular adapter projections and updates. |
| `src/defense.rs` | Refractory rate-limiter primitives for stable updates and overwrite defense. |
| `src/linalg.rs` | Small allocation-conscious vector, matrix, math, and deterministic RNG utilities. |
| `src/diagnostics.rs` | CLI banner formatting. |
| `data/downloaded.txt` | Checked-in corpus used as the default when present. |
| `data/model.pssa` | Checked-in serialized model artifact. |

## Development

Format and compile the project locally:

```bash
cargo fmt --all -- --check
cargo check
cargo test
```

There are currently no test cases, so `cargo test` only verifies that the test targets compile. Add focused tests for tokenization, source resolution, serialization round trips, memory retrieval, and sampling behavior as the implementation stabilizes.

## Limitations

- This is a CPU-oriented prototype with hand-written linear algebra and no GPU acceleration.
- The CLI parser is intentionally minimal and does not provide shell-style quoting or rich validation beyond basic numeric parsing.
- A missing or unreadable dataset silently falls back to the built-in science corpus in several loading paths.
- Model and tokenizer vocabularies must remain compatible; a size warning does not repair a mismatch.
- Downloaded content can be large and may contain JSON, malformed text, or data unsuitable for training.
- The REPL temperature command currently acknowledges a value without changing the active configuration.
- Benchmark output is milestone-oriented and does not measure perplexity, factuality, latency, or safety.
- Serialized `.pssa` files are project-specific binary artifacts without version migration tooling.

## License

See [LICENSE](LICENSE) for the project license.
