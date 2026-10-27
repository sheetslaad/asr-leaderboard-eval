#!/bin/zsh

set -euo pipefail

SCRIPT_DIR=${0:A:h}
PROJECT_ROOT=${SCRIPT_DIR:h}
cd "$PROJECT_ROOT"

manifest=data/earnings22_manifest_n300.csv

if [[ ! -f "$manifest" ]]; then
  .venv/bin/python -m src.sample_earnings22 \
    --n 300 \
    --seed 42 \
    --output "$manifest"
fi

mkdir -p results

models=(
  facebook/wav2vec2-base-960h
  openai/whisper-small.en
  openai/whisper-tiny.en
  openai/whisper-base.en
  openai/whisper-medium.en
)

for model in "${models[@]}"; do
  safe_model=${model//\//_}
  extra_args=()
  if [[ "$model" == openai/whisper-* ]]; then
    # Earnings22 contains short conversational clips. Forcing timestamp-token
    # generation on every clip triggered pathological repetitions, so use
    # ordinary short-form decoding unless a clip exceeds 30 seconds.
    output="results/${safe_model}_earnings22_n300_shortform.csv"
    extra_args=(--whisper-timestamps auto)
  else
    output="results/${safe_model}_earnings22_n300.csv"
  fi

  print "\n===== ${model} ====="
  .venv/bin/python -m src.run_eval \
    --dataset hf-audio/open-asr-leaderboard \
    --config earnings22 \
    --split test \
    --manifest "$manifest" \
    --model "$model" \
    --output "$output" \
    "${extra_args[@]}" \
    2>&1 | tee -a results/earnings22_n300_eval.log
done

print "\nAll sampled Earnings22 evaluations completed."
