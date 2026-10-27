#!/bin/zsh

set -euo pipefail

SCRIPT_DIR=${0:A:h}
PROJECT_ROOT=${SCRIPT_DIR:h}
cd "$PROJECT_ROOT"

mkdir -p results

models=(
  openai/whisper-tiny.en
  openai/whisper-base.en
  facebook/wav2vec2-base-960h
  openai/whisper-small.en
  openai/whisper-medium.en
)

for model in "${models[@]}"; do
  safe_model=${model//\//_}
  extra_args=()
  if [[ "$model" == openai/whisper-* ]]; then
    output="results/${safe_model}_earnings22_full_shortform.csv"
    extra_args=(--whisper-timestamps auto)
  else
    output="results/${safe_model}_earnings22_full.csv"
  fi

  print "\n===== ${model} ====="
  .venv/bin/python -m src.run_eval \
    --dataset hf-audio/open-asr-leaderboard \
    --config earnings22 \
    --split test \
    --full \
    --model "$model" \
    --output "$output" \
    "${extra_args[@]}" \
    2>&1 | tee -a results/earnings22_full_eval.log
done

print "\nAll Earnings22 evaluations completed."
