
from datasets import load_dataset

DATASET = "openslr/librispeech_asr"

def load_samples(n=10):
    ds = load_dataset(
        DATASET,
        "clean",
        split="test",
        streaming=True,
    )

    samples = []
    for sample in ds.take(n):
        samples.append(sample)

    return samples


if __name__ == "__main__":
    samples = load_samples(10)

    for i, sample in enumerate(samples):
        print(f"Sample {i + 1}")
        print("Reference:", sample["text"])
        print("Audio:", sample["audio"])
        print()