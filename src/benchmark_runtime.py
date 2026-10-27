import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import pipeline

from src.run_eval import load_manifest_samples


DEFAULT_MODELS = [
    "openai/whisper-tiny.en",
    "openai/whisper-base.en",
    "openai/whisper-small.en",
    "openai/whisper-medium.en",
]
BUCKET_ORDER = ["short", "medium", "long"]


def select_benchmark_manifest(manifest, per_bucket, seed):
    """Select a reproducible duration-stratified subset of the n=300 manifest."""
    required = {"sample_id", "duration_sec", "duration_bucket"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            "Manifest is missing required columns: " + ", ".join(sorted(missing))
        )

    selected = []
    for offset, bucket in enumerate(BUCKET_ORDER):
        candidates = manifest.loc[manifest["duration_bucket"] == bucket]
        if len(candidates) < per_bucket:
            raise ValueError(
                f"Bucket {bucket!r} has {len(candidates)} rows; "
                f"cannot select {per_bucket}"
            )
        selected.append(
            candidates.sample(n=per_bucket, random_state=seed + offset)
        )

    return pd.concat(selected, ignore_index=True)


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def clear_device_cache(device):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def resolve_device(requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def decode_audio(samples):
    """Decode once so disk/codec work is excluded from inference timing."""
    decoded_samples = []
    for sample in samples:
        decoded = sample["audio"].get_all_samples()
        waveform = decoded.data
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)

        duration_sec = len(waveform) / decoded.sample_rate
        decoded_samples.append(
            {
                "sample_id": str(sample["id"]),
                "duration_sec": duration_sec,
                "waveform": waveform.numpy(),
                "sampling_rate": decoded.sample_rate,
            }
        )
    return decoded_samples


def pipeline_input(sample):
    """Build a fresh dict because the Transformers pipeline mutates its input."""
    return {
        "raw": sample["waveform"],
        "sampling_rate": sample["sampling_rate"],
    }


def benchmark_model(model_name, audio_samples, device, repeats):
    """Return per-call timing rows and model-level resource metadata."""
    clear_device_cache(device)
    synchronize(device)

    load_started = time.perf_counter()
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )
    synchronize(device)
    load_seconds = time.perf_counter() - load_started

    parameter_count = sum(parameter.numel() for parameter in transcriber.model.parameters())

    # Warm up kernels and one-time allocations without including them in results.
    inference_kwargs = {}
    if getattr(transcriber.model.config, "model_type", None) == "whisper":
        inference_kwargs["return_timestamps"] = True
    transcriber(pipeline_input(audio_samples[0]), **inference_kwargs)
    synchronize(device)

    mps_allocated_mb = np.nan
    mps_driver_mb = np.nan
    if device == "mps":
        mps_allocated_mb = torch.mps.current_allocated_memory() / (1024**2)
        mps_driver_mb = torch.mps.driver_allocated_memory() / (1024**2)

    rows = []
    for repeat in range(1, repeats + 1):
        print(f"  Repeat {repeat}/{repeats}")
        for position, sample in enumerate(audio_samples, start=1):
            synchronize(device)
            started = time.perf_counter()
            transcriber(pipeline_input(sample), **inference_kwargs)
            synchronize(device)
            inference_seconds = time.perf_counter() - started

            rows.append(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "position": position,
                    "sample_id": sample["sample_id"],
                    "duration_sec": sample["duration_sec"],
                    "inference_sec": inference_seconds,
                    "real_time_factor": inference_seconds
                    / sample["duration_sec"],
                    "throughput_x": sample["duration_sec"]
                    / inference_seconds,
                }
            )

    metadata = {
        "model": model_name,
        "device": device,
        "parameters_millions": parameter_count / 1_000_000,
        "load_seconds": load_seconds,
        "mps_allocated_mb_after_warmup": mps_allocated_mb,
        "mps_driver_mb_after_warmup": mps_driver_mb,
    }

    del transcriber
    clear_device_cache(device)
    return rows, metadata


def summarize(detail, metadata_rows):
    metadata = pd.DataFrame(metadata_rows).set_index("model")
    summary_rows = []

    for model, group in detail.groupby("model", sort=False):
        meta = metadata.loc[model]
        total_audio_seconds = group["duration_sec"].sum()
        total_inference_seconds = group["inference_sec"].sum()

        summary_rows.append(
            {
                "model": model,
                "device": meta["device"],
                "parameters_millions": meta["parameters_millions"],
                "load_seconds": meta["load_seconds"],
                "utterances_per_repeat": group["sample_id"].nunique(),
                "repeats": group["repeat"].nunique(),
                "audio_seconds_per_repeat": group.loc[
                    group["repeat"] == group["repeat"].min(), "duration_sec"
                ].sum(),
                "median_latency_sec": group["inference_sec"].median(),
                "p95_latency_sec": group["inference_sec"].quantile(0.95),
                "real_time_factor": total_inference_seconds / total_audio_seconds,
                "throughput_x": total_audio_seconds / total_inference_seconds,
                "mps_allocated_mb_after_warmup": meta[
                    "mps_allocated_mb_after_warmup"
                ],
                "mps_driver_mb_after_warmup": meta[
                    "mps_driver_mb_after_warmup"
                ],
            }
        )

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper inference on a fixed duration-stratified subset."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest_n300.csv"),
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--per-bucket", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, mps, cpu, or a Transformers device such as cuda:0",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/runtime_benchmark_n30"),
    )
    args = parser.parse_args()

    if args.per_bucket <= 0 or args.repeats <= 0:
        parser.error("--per-bucket and --repeats must be greater than zero")

    manifest = pd.read_csv(args.manifest, dtype={"sample_id": str})
    benchmark_manifest = select_benchmark_manifest(
        manifest,
        per_bucket=args.per_bucket,
        seed=args.seed,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    subset_path = args.output_prefix.with_name(
        args.output_prefix.name + "_manifest.csv"
    )
    detail_path = args.output_prefix.with_name(
        args.output_prefix.name + "_detail.csv"
    )
    summary_path = args.output_prefix.with_name(
        args.output_prefix.name + "_summary.csv"
    )
    benchmark_manifest.to_csv(subset_path, index=False)

    samples, _ = load_manifest_samples(subset_path)
    print("Decoding benchmark audio outside the timed region...")
    audio_samples = decode_audio(samples)

    device = resolve_device(args.device)
    print(f"Benchmark device: {device}")
    print(
        f"Benchmark set: {len(audio_samples)} utterances "
        f"({sum(x['duration_sec'] for x in audio_samples):.1f} audio seconds)"
    )

    timing_rows = []
    metadata_rows = []
    for model_name in args.models:
        print(f"\nBenchmarking {model_name}")
        rows, metadata = benchmark_model(
            model_name,
            audio_samples,
            device=device,
            repeats=args.repeats,
        )
        timing_rows.extend(rows)
        metadata_rows.append(metadata)

        # Save progress after each model so a later failure does not lose prior timings.
        pd.DataFrame(timing_rows).to_csv(detail_path, index=False)
        summarize(pd.DataFrame(timing_rows), metadata_rows).to_csv(
            summary_path,
            index=False,
        )

    summary = summarize(pd.DataFrame(timing_rows), metadata_rows)
    print("\nBenchmark complete")
    print(summary.to_string(index=False))
    print(f"Subset manifest: {subset_path}")
    print(f"Detailed timings: {detail_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
