"""Extract lightweight, reproducible acoustic diagnostics for a manifest.

The features are deliberately simple and dependency-light. They are suitable for
relative diagnostics, not clinical or production-grade VAD/SNR measurement.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.load_data import DATASET


def frame_rms(waveform, sample_rate, frame_ms=25, hop_ms=10):
    frame_length = max(1, round(sample_rate * frame_ms / 1000))
    hop_length = max(1, round(sample_rate * hop_ms / 1000))
    if len(waveform) < frame_length:
        waveform = np.pad(waveform, (0, frame_length - len(waveform)))
    starts = np.arange(0, len(waveform) - frame_length + 1, hop_length)
    frames = np.stack([waveform[start:start + frame_length] for start in starts])
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    return rms, hop_length / sample_rate


def seconds_to_first_active(active, hop_seconds):
    positions = np.flatnonzero(active)
    return float(positions[0] * hop_seconds) if len(positions) else np.nan


def extract_features(waveform, sample_rate, reference_words):
    waveform = np.asarray(waveform, dtype=np.float64)
    rms, hop_seconds = frame_rms(waveform, sample_rate)
    frame_dbfs = 20 * np.log10(rms)
    overall_rms = np.sqrt(np.mean(np.square(waveform)) + 1e-12)

    # Method A: active within 30 dB of the recording's loudest frame, with a
    # -45 dBFS floor. Method B uses a stricter 25 dB relative threshold.
    threshold_a = max(float(frame_dbfs.max() - 30), -45.0)
    threshold_b = max(float(frame_dbfs.max() - 25), -40.0)
    active_a = frame_dbfs >= threshold_a
    active_b = frame_dbfs >= threshold_b
    voiced_a = max(float(active_a.sum() * hop_seconds), hop_seconds)

    noise_rms = float(np.quantile(rms, 0.10))
    speech_rms = float(np.quantile(rms, 0.75))
    estimated_snr_db = 20 * np.log10((speech_rms + 1e-12) / (noise_rms + 1e-12))

    return {
        "decoded_duration_sec": len(waveform) / sample_rate,
        "rms_dbfs": 20 * np.log10(overall_rms),
        "estimated_snr_db": estimated_snr_db,
        "leading_silence_a_sec": seconds_to_first_active(active_a, hop_seconds),
        "leading_silence_b_sec": seconds_to_first_active(active_b, hop_seconds),
        "voiced_duration_a_sec": voiced_a,
        "articulation_rate_a_wps": reference_words / voiced_a,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_n300.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/acoustic_metadata_n300.csv"))
    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest, dtype={"sample_id": str, "speaker_id": str}
    )
    required = {"sample_id", "speaker_id", "reference", "duration_sec"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    pending = set(manifest["sample_id"])
    records = []
    dataset = load_dataset(DATASET, "clean", split="test", streaming=True)
    print(f"Extracting acoustic metadata for {len(pending)} utterances...")
    for sample in dataset:
        sample_id = str(sample["id"])
        if sample_id not in pending:
            continue
        decoded = sample["audio"].get_all_samples()
        waveform = decoded.data
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        row = manifest.loc[manifest["sample_id"] == sample_id].iloc[0]
        features = extract_features(
            waveform.numpy(), decoded.sample_rate, len(str(row["reference"]).split())
        )
        records.append({
            "sample_id": sample_id,
            "speaker_id": str(row["speaker_id"]),
            "manifest_duration_sec": float(row["duration_sec"]),
            **features,
        })
        pending.remove(sample_id)
        if len(records) % 25 == 0:
            print(f"  {len(records)}/{len(manifest)}")
        if not pending:
            break

    if pending:
        raise RuntimeError(f"Missing {len(pending)} manifest recordings: {sorted(pending)[:10]}")
    output = pd.DataFrame(records).sort_values("sample_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
