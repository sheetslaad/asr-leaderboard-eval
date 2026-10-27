
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


DATASET = "openslr/librispeech_asr"
CONFIG = "clean"
SPLIT = "test"


def build_manifest(n=300, seed=42):
    ds = load_dataset(
        DATASET,
        CONFIG,
        split=SPLIT,
        streaming=True,
    )

    records = []

    print("Reading dataset metadata...")

    for sample in ds:
        decoded = sample["audio"].get_all_samples()

        duration = (
            decoded.data.shape[-1] / decoded.sample_rate
        )

        records.append({
            "sample_id": str(sample["id"]),
            "speaker_id": str(sample["speaker_id"]),
            "reference": sample["text"],
            "duration_sec": duration,
        })

    df = pd.DataFrame(records)

    print(f"Total utterances: {len(df)}")
    print(f"Total speakers: {df['speaker_id'].nunique()}")

    # Duration strata using quantiles of the full test set.
    df["duration_bucket"] = pd.qcut(
        df["duration_sec"],
        q=3,
        labels=["short", "medium", "long"],
        duplicates="drop",
    )

    # Sample speakers first to improve speaker coverage.
    rng = np.random.default_rng(seed)

    speakers = rng.permutation(
        df["speaker_id"].unique()
    )

    selected = []
    selected_ids = set()

    # First pass: at least one utterance per speaker.
    for speaker in speakers:
        candidates = df[
            df["speaker_id"] == speaker
        ]

        if len(selected) >= n:
            break

        chosen = candidates.sample(
            n=1,
            random_state=seed,
        ).iloc[0]

        selected.append(chosen)
        selected_ids.add(chosen["sample_id"])

    # Second pass: balance duration categories.
    remaining_n = n - len(selected)

    if remaining_n > 0:
        remaining = df[
            ~df["sample_id"].isin(selected_ids)
        ]

        # Allocate approximately equally across duration buckets.
        groups = [
            group for _, group in
            remaining.groupby(
                "duration_bucket",
                observed=True,
            )
        ]

        per_group = remaining_n // len(groups)

        for group in groups:
            take = min(per_group, len(group))

            chosen = group.sample(
                n=take,
                random_state=seed,
            )

            selected.extend(
                row for _, row in chosen.iterrows()
            )
            selected_ids.update(
                chosen["sample_id"].tolist()
            )

    # Fill any remainder without replacement.
    if len(selected) < n:
        remaining = df[
            ~df["sample_id"].isin(selected_ids)
        ]

        extra = remaining.sample(
            n=n - len(selected),
            random_state=seed,
        )

        selected.extend(
            row for _, row in extra.iterrows()
        )

    manifest = pd.DataFrame(selected)

    if len(manifest) != n:
        raise ValueError(
            f"Expected {n} samples, got {len(manifest)}"
        )

    manifest = manifest.sort_values(
        "sample_id"
    ).reset_index(drop=True)

    Path("data").mkdir(exist_ok=True)

    output = Path("data") / f"manifest_n{n}.csv"
    manifest.to_csv(output, index=False)

    summary = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "n": n,
        "seed": seed,
        "total_speakers": int(
            manifest["speaker_id"].nunique()
        ),
        "duration_summary": (
            manifest["duration_sec"].describe().to_dict()
        ),
        "duration_buckets": (
            manifest["duration_bucket"]
            .value_counts()
            .to_dict()
        ),
    }

    with open(
        Path("data") / f"manifest_n{n}_summary.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved manifest to {output}")
    print(f"Selected speakers: {summary['total_speakers']}")
    print(
        manifest["duration_bucket"]
        .value_counts()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_manifest(n=args.n, seed=args.seed)