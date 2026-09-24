#!/usr/bin/env bash
# Update an existing Kaggle checkout to main and continue the chain from the
# newest checkpoint already in /kaggle/working/chain, instead of retraining
# from scratch. Safe to run after kaggle_gpu_setup.sh died partway.
set -euo pipefail

REPO="${REPO:-https://github.com/Sparticle62ops/oxide-ai.git}"
WORK="${WORK:-/kaggle/working}"
BRANCH="${BRANCH:-main}"
TOTAL="${TOTAL:-16}"
# Each link reads a different WINDOW-sized slice instead of the same prefix, so the
# chain walks the whole corpus. The offset wraps around at the end of the file.
WINDOW="${WINDOW:-200000}"

source "$HOME/.cargo/env" 2>/dev/null || true
if ! command -v cargo >/dev/null 2>&1; then
  echo "### 0. Rust toolchain (clean container)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  source "$HOME/.cargo/env"
fi
cargo --version

echo "### 1. Update checkout to $BRANCH"
if [ -d "$WORK/oxide-ai/.git" ]; then
  cd "$WORK/oxide-ai"
  git fetch --quiet origin "$BRANCH"
  git checkout --quiet -B "$BRANCH" "origin/$BRANCH"
else
  cd "$WORK"
  git clone --quiet --branch "$BRANCH" "$REPO"
  cd oxide-ai
fi
git log --oneline -1

echo
echo "### 2. Build"
cargo build --release
HELP_TEXT="$(./target/release/oxide_ai_pssa help 2>&1 || true)"
case "$HELP_TEXT" in
  *--resume*) echo "--resume present" ;;
  *) echo "ERROR: this checkout has no --resume, stopping"; exit 1 ;;
esac

echo
echo "### 3. Continue the chain"
mkdir -p "$WORK/chain"
PREV=""
START=1
for i in $(seq 1 "$TOTAL"); do
  CK="$WORK/chain/ck$(printf '%02d' "$i").pssa"
  if [ -f "$CK" ]; then PREV="$CK"; START=$((i + 1)); fi
done
if [ -n "$PREV" ]; then
  echo "resuming from $PREV, next is ck$(printf '%02d' "$START")"
else
  echo "no existing checkpoints, starting fresh"
fi

for i in $(seq "$START" "$TOTAL"); do
  OUT="$WORK/chain/ck$(printf '%02d' "$i").pssa"
  SKIP=$(( (i - 1) * WINDOW ))
  echo "--- ck$(printf '%02d' "$i") (corpus offset $SKIP) ---"
  if [ -z "$PREV" ]; then
    ./target/release/oxide_ai_pssa train data/downloaded.txt -o "$OUT" --max-tokens "$WINDOW" --skip-tokens "$SKIP" -e 1
  else
    ./target/release/oxide_ai_pssa train data/downloaded.txt -o "$OUT" --max-tokens "$WINDOW" --skip-tokens "$SKIP" -e 1 --resume "$PREV"
  fi
  PREV="$OUT"
done

echo
echo "### 4. Sample"
./target/release/oxide_ai_pssa generate -m "$PREV" -p "The sun is"
./target/release/oxide_ai_pssa generate -m "$PREV" -p "Anarchism is"
