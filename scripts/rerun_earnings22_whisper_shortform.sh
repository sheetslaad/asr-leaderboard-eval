#!/bin/zsh

set -euo pipefail

SCRIPT_DIR=${0:A:h}
PROJECT_ROOT=${SCRIPT_DIR:h}
cd "$PROJECT_ROOT"

manifest=data/earnings22_manifest_n300.csv
if [[ ! -f "$manifest" ]]; then
  print "Missing $manifest. Run src.sample_earnings22 first."
  exit 1
fi

mkdir -p results

# Run the most informative models first. Wav2Vec2 is not included because its
# CTC decoder does not use Whisper timestamp generation and needs no rerun.
models=(
  openai/whisper-medium.en
  openai/whisper-small.en
  openai/whisper-base.en
  openai/whisper-tiny.en
)

for model in "${models[@]}"; do
  safe_model=${model//\//_}
  output="results/${safe_model}_earnings22_n300_shortform.csv"

  print "\n===== ${model}: short-form decoding ====="
  .venv/bin/python -m src.run_eval \
    --dataset hf-audio/open-asr-leaderboard \
    --config earnings22 \
    --split test \
    --manifest "$manifest" \
    --model "$model" \
    --whisper-timestamps auto \
    --output "$output" \
    2>&1 | tee -a results/earnings22_n300_shortform_eval.log
done

print "\nAll corrected Earnings22 Whisper evaluations completed."
