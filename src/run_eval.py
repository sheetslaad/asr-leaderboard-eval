import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from jiwer import wer
from transformers import pipeline

from src.load_data import DATASET, load_samples


RESULT_COLUMNS = [
    "sample_id",
    "reference",
    "prediction",
    "duration_sec",
]


def load_manifest_samples(
    manifest_path,
    dataset_name=DATASET,
    dataset_config="clean",
    dataset_split="test",
):
    """Load exactly the dataset utterances named by a manifest."""
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})

    if "sample_id" not in manifest.columns:
        raise ValueError(
            f"Manifest {manifest_path} must contain a sample_id column"
        )
    if manifest.empty:
        raise ValueError(f"Manifest {manifest_path} is empty")
    if manifest["sample_id"].isna().any():
        raise ValueError(f"Manifest {manifest_path} contains blank sample IDs")

    duplicate_ids = manifest.loc[
        manifest["sample_id"].duplicated(), "sample_id"
    ].tolist()
    if duplicate_ids:
        raise ValueError(
            "Manifest contains duplicate sample IDs: "
            + ", ".join(duplicate_ids[:10])
        )

    expected_ids = manifest["sample_id"].tolist()
    expected_id_set = set(expected_ids)
    found = {}

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        streaming=True,
    )

    print(
        f"Finding {len(expected_ids)} manifest samples in "
        f"{dataset_name}/{dataset_config}/{dataset_split}..."
    )
    for sample in dataset:
        sample_id = str(sample["id"])
        if sample_id in expected_id_set:
            found[sample_id] = sample
            if len(found) == len(expected_ids):
                break

    missing_ids = [sample_id for sample_id in expected_ids if sample_id not in found]
    if missing_ids:
        raise ValueError(
            f"Found {len(found)}/{len(expected_ids)} manifest samples. "
            "Missing IDs: " + ", ".join(missing_ids[:20])
        )

    # Catch a stale or accidentally mismatched manifest before spending time on ASR.
    if "reference" in manifest.columns:
        manifest_references = dict(
            zip(manifest["sample_id"], manifest["reference"].fillna(""))
        )
        mismatches = [
            sample_id
            for sample_id in expected_ids
            if str(found[sample_id]["text"]) != str(manifest_references[sample_id])
        ]
        if mismatches:
            raise ValueError(
                "Manifest references do not match the selected dataset for IDs: "
                + ", ".join(mismatches[:20])
            )

    print(f"Manifest completeness check passed: {len(found)}/{len(expected_ids)}")
    return [found[sample_id] for sample_id in expected_ids], expected_ids


def load_checkpoint(output_path, expected_ids, resume):
    if not resume or not output_path.exists():
        return {}

    checkpoint = pd.read_csv(output_path, dtype={"sample_id": str})
    missing_columns = set(RESULT_COLUMNS) - set(checkpoint.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot resume from {output_path}: missing columns "
            + ", ".join(sorted(missing_columns))
        )
    if checkpoint["sample_id"].duplicated().any():
        raise ValueError(f"Cannot resume from {output_path}: duplicate sample IDs")
    # A model can legitimately emit an empty transcript. Pandas reads an empty
    # CSV field as NaN, so restore it before validating resumable rows.
    checkpoint["prediction"] = checkpoint["prediction"].fillna("")
    required_nonempty = ["sample_id", "reference", "duration_sec"]
    if checkpoint[required_nonempty].isna().any().any():
        raise ValueError(f"Cannot resume from {output_path}: incomplete rows")

    unexpected_ids = set(checkpoint["sample_id"]) - set(expected_ids)
    if unexpected_ids:
        raise ValueError(
            f"Cannot resume from {output_path}: unexpected sample IDs: "
            + ", ".join(sorted(unexpected_ids)[:20])
        )

    rows = checkpoint[RESULT_COLUMNS].to_dict("records")
    completed = {row["sample_id"]: row for row in rows}
    print(f"Resuming from {output_path}: {len(completed)} samples complete")
    return completed


def save_checkpoint(output_path, results_by_id, expected_ids):
    rows = [
        results_by_id[sample_id]
        for sample_id in expected_ids
        if sample_id in results_by_id
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of leading samples to evaluate without --manifest/--full",
    )
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV whose sample_id column defines the exact evaluation set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Results CSV path (default: derived from model and manifest)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing results CSV and start the evaluation again",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET,
        help="Hugging Face dataset repository",
    )
    parser.add_argument(
        "--config",
        default="clean",
        help="Hugging Face dataset configuration",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Hugging Face dataset split",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate the complete selected split",
    )
    parser.add_argument(
        "--whisper-timestamps",
        choices=["always", "auto", "never"],
        default="always",
        help=(
            "Whisper timestamp decoding: always preserves the original project "
            "configuration; auto enables it only for audio longer than 30 seconds"
        ),
    )
    args = parser.parse_args()

    if args.full and args.manifest:
        parser.error("Use either --full or --manifest, not both")

    if args.manifest:
        samples, expected_ids = load_manifest_samples(
            args.manifest,
            dataset_name=args.dataset,
            dataset_config=args.config,
            dataset_split=args.split,
        )
        run_name = args.manifest.stem
    elif args.full:
        samples = load_dataset(
            args.dataset,
            args.config,
            split=args.split,
        )
        expected_ids = [str(sample_id) for sample_id in samples["id"]]
        run_name = f"{args.config}_{args.split}_full".replace(".", "_")
    else:
        if args.n <= 0:
            parser.error("--n must be greater than zero")
        if (
            args.dataset == DATASET
            and args.config == "clean"
            and args.split == "test"
        ):
            samples = load_samples(args.n)
        else:
            dataset = load_dataset(
                args.dataset,
                args.config,
                split=args.split,
                streaming=True,
            )
            samples = list(dataset.take(args.n))
        expected_ids = [str(sample["id"]) for sample in samples]
        run_name = f"{args.config}_{args.split}_n{args.n}".replace(".", "_")

    if len(samples) != len(expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise RuntimeError("Evaluation sample completeness check failed")

    model_name = args.model.replace("/", "_")
    output_path = args.output or Path("results") / f"{model_name}_{run_name}.csv"
    results_by_id = load_checkpoint(
        output_path,
        expected_ids,
        resume=not args.no_resume,
    )

    pending_count = len(expected_ids) - len(results_by_id)
    if pending_count:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"Samples remaining: {pending_count}/{len(expected_ids)}")

        transcriber = pipeline(
            "automatic-speech-recognition",
            model=args.model,
            device=device,
        )

        for i, sample in enumerate(samples, start=1):
            sample_id = str(sample["id"])
            if sample_id in results_by_id:
                continue

            decoded = sample["audio"].get_all_samples()
            waveform = decoded.data

            # Convert to mono while preserving the working TorchCodec path.
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=0)

            audio = {
                "raw": waveform.numpy(),
                "sampling_rate": decoded.sample_rate,
            }

            # Whisper uses timestamp generation for robust long-form handling. CTC
            # pipelines such as Wav2Vec2 should use their normal non-timestamp path.
            inference_kwargs = {}
            if getattr(transcriber.model.config, "model_type", None) == "whisper":
                if args.whisper_timestamps == "always":
                    inference_kwargs["return_timestamps"] = True
                elif args.whisper_timestamps == "auto":
                    inference_kwargs["return_timestamps"] = (
                        len(waveform) / decoded.sample_rate > 30.0
                    )
            prediction = transcriber(audio, **inference_kwargs)["text"]
            reference = sample["text"]

            results_by_id[sample_id] = {
                "sample_id": sample_id,
                "reference": reference,
                "prediction": prediction,
                "duration_sec": len(waveform) / decoded.sample_rate,
            }
            save_checkpoint(output_path, results_by_id, expected_ids)

            print(f"\nSample {i}/{len(samples)}")
            print("Reference:", reference[:120])
            print("Prediction:", prediction[:120])
            print(f"Checkpoint saved to {output_path}")

    final_ids = set(results_by_id)
    if len(results_by_id) != len(expected_ids) or final_ids != set(expected_ids):
        missing_ids = set(expected_ids) - final_ids
        raise RuntimeError(
            "Evaluation did not complete every selected sample. Missing IDs: "
            + ", ".join(sorted(missing_ids)[:20])
        )

    save_checkpoint(output_path, results_by_id, expected_ids)
    df = pd.DataFrame(
        [results_by_id[sample_id] for sample_id in expected_ids],
        columns=RESULT_COLUMNS,
    )
    score = wer(df["reference"].tolist(), df["prediction"].tolist())

    print(f"\nCompleted samples: {len(df)}/{len(expected_ids)}")
    print(f"Raw WER: {score:.2%}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
