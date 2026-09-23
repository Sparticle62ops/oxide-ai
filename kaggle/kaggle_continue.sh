#!/usr/bin/env bash
# Update an existing Kaggle checkout to main and continue the chain from the
# newest checkpoint already in /kaggle/working/chain, instead of retraining
# from scratch. Safe to run after kaggle_gpu_setup.sh died partway.
set -euo pipefail

REPO="${REPO:-https://github.com/Sparticle62ops/oxide-ai.git}"
WORK="${WORK:-/kaggle/working}"
BRANCH="${BRANCH:-main}"
TOTAL="${TOTAL:-8}"

source "$HOME/.cargo/env" 2>/dev/null || true

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
./target/release/oxide_ai_pssa help 2>&1 | grep -q -- '--resume' || {
  echo "ERROR: this checkout has no --resume, stopping"; exit 1; }
echo "--resume present"

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
  echo "--- ck$(printf '%02d' "$i") ---"
  if [ -z "$PREV" ]; then
    ./target/release/oxide_ai_pssa train data/downloaded.txt -o "$OUT" --max-tokens 200000 -e 1
  else
    ./target/release/oxide_ai_pssa train data/downloaded.txt -o "$OUT" --max-tokens 200000 -e 1 --resume "$PREV"
  fi
  PREV="$OUT"
done

echo
echo "### 4. Sample"
./target/release/oxide_ai_pssa generate -m "$PREV" -p "The sun is"
./target/release/oxide_ai_pssa generate -m "$PREV" -p "Anarchism is"
