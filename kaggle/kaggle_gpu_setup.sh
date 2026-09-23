#!/usr/bin/env bash
# Oxide AI on a Kaggle GPU notebook.
#
# Usage: create a new Kaggle notebook, set Accelerator to GPU (T4 or P100) and
# Internet to On, then run this whole file in one cell with:
#
#     !bash kaggle_gpu_setup.sh
#
# or paste the body into a cell prefixed with %%bash.
set -euo pipefail

BRANCH="${BRANCH:-feature/wgpu-wiring}"
REPO="${REPO:-https://github.com/Sparticle62ops/oxide-ai.git}"
WORK="${WORK:-/kaggle/working}"

echo "### 1. Vulkan driver (wgpu speaks Vulkan, not CUDA directly)"
# Kaggle images ship CUDA but usually not the Vulkan loader or the NVIDIA ICD.
# Without these, wgpu finds no adapter even though a GPU is attached.
apt-get update -qq
apt-get install -y -qq libvulkan1 vulkan-tools mesa-vulkan-drivers >/dev/null 2>&1 || true

# Point the loader at the NVIDIA ICD if the driver is present but unregistered.
mkdir -p /usr/share/vulkan/icd.d
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  NVLIB="$(ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>/dev/null || true)"
  if [ -n "$NVLIB" ]; then
    cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<JSON
{"file_format_version":"1.0.0","ICD":{"library_path":"$NVLIB","api_version":"1.3.0"}}
JSON
  fi
fi
vulkaninfo --summary 2>/dev/null | head -20 || echo "(vulkaninfo unavailable, probe will tell us)"

echo
echo "### 2. Rust toolchain"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
rustc --version

echo
echo "### 3. Clone and build"
cd "$WORK"
rm -rf oxide-ai
git clone --quiet --branch "$BRANCH" "$REPO"
cd oxide-ai
cargo build --release

echo
echo "### 4. GPU probe"
# Passes only if the GPU kernel result matches the CPU reference.
./target/release/oxide_ai_pssa gpu-probe

echo
echo "### 5. Chained training"
# --resume lets each run continue the previous checkpoint instead of
# starting from scratch, so loss accumulates across runs.
mkdir -p "$WORK/chain"
PREV=""
for i in 1 2 3 4 5 6 7 8; do
  OUT="$WORK/chain/ck$(printf '%02d' "$i").pssa"
  if [ -z "$PREV" ]; then
    ./target/release/oxide_ai_pssa train data/downloaded.txt -o "$OUT" --max-tokens 200000 -e 1
  else
    ./target/release/oxide_ai_pssa train data/downloaded.txt -o "$OUT" --max-tokens 200000 -e 1 --resume "$PREV"
  fi
  PREV="$OUT"
done

echo
echo "### 6. Sample"
./target/release/oxide_ai_pssa generate -m "$PREV" -p "The sun is"
./target/release/oxide_ai_pssa generate -m "$PREV" -p "Anarchism is"

echo
echo "Done. Checkpoints are in $WORK/chain and download from the notebook Output tab."
