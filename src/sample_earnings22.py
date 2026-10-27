import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


DATASET = "hf-audio/open-asr-leaderboard"
CONFIG = "earnings22"
SPLIT = "test"


def proportional_allocation(stratum_sizes, n):
    """Allocate n rows proportionally while retaining every non-empty stratum."""
    if n < len(stratum_sizes):
        raise ValueError(
            f"n={n} is too small to cover all {len(stratum_sizes)} strata"
        )

    ideal = n * stratum_sizes / stratum_sizes.sum()
    allocation = np.floor(ideal).astype(int).clip(lower=1)
    allocation = np.minimum(allocation, stratum_sizes)

    while allocation.sum() < n:
        candidates = allocation[allocation < stratum_sizes].index
        if len(candidates) == 0:
            raise ValueError("No remaining rows are available to complete the sample")
        priority = (ideal - allocation).loc[candidates]
        allocation.loc[priority.idxmax()] += 1

    while allocation.sum() > n:
        candidates = allocation[allocation > 1].index
        if len(candidates) == 0:
            raise ValueError("Cannot reduce allocation while retaining every stratum")
        priority = (ideal - allocation).loc[candidates]
        allocation.loc[priority.idxmin()] -= 1

    return allocation.astype(int)


def build_manifest(n=300, seed=42, output=None):
    # Removing audio before iteration lets Parquet return metadata without
    # decoding every waveform.
    dataset = load_dataset(
        DATASET,
        CONFIG,
        split=SPLIT,
        streaming=True,
    ).remove_columns("audio")

    records = []
    print("Reading Earnings22 metadata without decoding audio...")
    for sample in dataset:
        sample_id = str(sample["id"])
        records.append(
            {
                "sample_id": sample_id,
                "source_call_id": sample_id.split("/", 1)[0],
                "reference": sample["text"],
                "duration_sec": float(sample["audio_length_s"]),
            }
        )

    population = pd.DataFrame(records)
    if not 0 < n <= len(population):
        raise ValueError(f"n must be between 1 and {len(population)}")

    population["duration_bucket"] = pd.qcut(
        population["duration_sec"],
        q=3,
        labels=["short", "medium", "long"],
        duplicates="drop",
    )
    population["stratum"] = (
        population["source_call_id"].astype(str)
        + "|"
        + population["duration_bucket"].astype(str)
    )

    stratum_sizes = population.groupby("stratum", observed=True).size().sort_index()
    allocation = proportional_allocation(stratum_sizes, n)
    rng = np.random.default_rng(seed)

    selected_parts = []
    for stratum, group in population.groupby("stratum", observed=True, sort=True):
        take = int(allocation.loc[stratum])
        chosen_positions = rng.choice(len(group), size=take, replace=False)
        chosen = group.iloc[np.sort(chosen_positions)].copy()
        chosen["stratum_population"] = len(group)
        chosen["stratum_sampled"] = take
        chosen["sampling_weight"] = len(group) / take
        selected_parts.append(chosen)

    manifest = (
        pd.concat(selected_parts, ignore_index=True)
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    if len(manifest) != n or manifest["sample_id"].duplicated().any():
        raise RuntimeError("Sample completeness check failed")
    if manifest["source_call_id"].nunique() != population["source_call_id"].nunique():
        raise RuntimeError("The sample does not cover every source call")

    output_path = output or Path("data") / f"earnings22_manifest_n{n}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    metadata_path = output_path.with_name("earnings22_metadata.csv")
    population.sort_values("sample_id").to_csv(metadata_path, index=False)

    summary = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "population_utterances": int(len(population)),
        "population_calls": int(population["source_call_id"].nunique()),
        "population_audio_hours": float(population["duration_sec"].sum() / 3600),
        "sample_utterances": int(len(manifest)),
        "sample_calls": int(manifest["source_call_id"].nunique()),
        "sample_audio_hours": float(manifest["duration_sec"].sum() / 3600),
        "seed": seed,
        "sampling": "proportional within source-call x duration-tertile strata",
        "duration_quantiles_seconds": {
            str(q): float(population["duration_sec"].quantile(q))
            for q in (0, 1 / 3, 2 / 3, 1)
        },
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Population: {len(population)} segments, {summary['population_calls']} calls")
    print(f"Sample: {len(manifest)} segments, {summary['sample_calls']} calls")
    print(f"Sample audio: {summary['sample_audio_hours']:.2f} hours")
    print(f"Saved manifest to {output_path}")
    print(f"Saved full metadata to {metadata_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build_manifest(n=args.n, seed=args.seed, output=args.output)
