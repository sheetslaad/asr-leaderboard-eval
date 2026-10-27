# ASR accuracy, efficiency, and failure analysis

This project recreates part of the [Hugging Face Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) with four English Whisper checkpoints and Wav2Vec2 Base. It evaluates the complete LibriSpeech `test-clean` split, tests generalization on a 300-segment Earnings22 sample, and examines normalization, runtime, sampling, metadata reliability, and severe errors that aggregate WER can hide.

See [`REPORT.md`](REPORT.md) for the findings, interpretation, limitations, automation plan, and AI-use disclosure. The notebooks contain the detailed calculations and review tools:

- [`notebooks/01_dataset_exploration.ipynb`](notebooks/01_dataset_exploration.ipynb): sampling, full-versus-sample distributions, repeated random-sample comparisons, and Earnings22 representativeness.
- [`notebooks/02_model_comparison.ipynb`](notebooks/02_model_comparison.ipynb): normalization, confidence intervals, published comparisons, runtime, metadata breakdowns, severe-error metrics, and playable error examples.

Both notebooks are stored with outputs and read the saved manifests and predictions; rerunning them does not perform model inference.

## Setup

Python 3.11 is recommended. From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user \
  --name asr-leaderboard-eval \
  --display-name "Python (asr-leaderboard-eval)"
```

## Reproduce the saved analysis

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_dataset_exploration.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/02_model_comparison.ipynb
```

These commands were verified from a fresh kernel after the final analysis changes.

## Reproduce inference

The evaluator saves completed rows and resumes by default. The fixed 300-recording LibriSpeech manifest can be run with:

```bash
.venv/bin/python -m src.run_eval --manifest data/manifest_n300.csv --model openai/whisper-tiny.en
.venv/bin/python -m src.run_eval --manifest data/manifest_n300.csv --model openai/whisper-base.en
.venv/bin/python -m src.run_eval --manifest data/manifest_n300.csv --model facebook/wav2vec2-base-960h
.venv/bin/python -m src.run_eval --manifest data/manifest_n300.csv --model openai/whisper-small.en
.venv/bin/python -m src.run_eval --manifest data/manifest_n300.csv --model openai/whisper-medium.en
```

For the complete 2,620-recording result, replace the manifest with `data/librispeech_test_clean_metadata.csv` and provide a distinct output path. For example:

```bash
.venv/bin/python -m src.run_eval \
  --manifest data/librispeech_test_clean_metadata.csv \
  --model openai/whisper-small.en \
  --output results/openai_whisper-small.en_test_clean_full_reproduction.csv
```

Do not overwrite the included result files unless a configuration change is intentional.

### Runtime benchmark

```bash
.venv/bin/python -m src.benchmark_runtime \
  --manifest data/manifest_n300.csv \
  --models \
    openai/whisper-tiny.en \
    openai/whisper-base.en \
    facebook/wav2vec2-base-960h \
    openai/whisper-small.en \
    openai/whisper-medium.en \
  --per-bucket 10 \
  --repeats 3 \
  --device auto
```

### Earnings22

Run the fixed 300-segment comparison on all five models with:

```bash
caffeinate -i zsh scripts/run_earnings22_sample.sh
```

To run the complete split instead:

```bash
caffeinate -i zsh scripts/run_earnings22_full.sh
```

### Acoustic metadata

```bash
.venv/bin/python -m src.extract_acoustic_metadata \
  --manifest data/manifest_n300.csv \
  --output data/acoustic_metadata_n300.csv
```

The exact evaluation and scoring settings are recorded in [`configs/evaluation.json`](configs/evaluation.json). Do not rerun `src.sample_data` unless you intentionally want to replace the fixed LibriSpeech sample.

## Repository layout

```text
configs/    evaluation and scoring configuration
data/       fixed manifests and reusable metadata
notebooks/  executable analysis and audio-review tools
results/    saved predictions and runtime measurements
scripts/    repeatable multi-model runs
src/        sampling, inference, normalization, metadata, and timing code
```
