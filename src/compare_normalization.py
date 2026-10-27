
import argparse

import pandas as pd
from jiwer import wer

from src.normalize import normalize_basic, normalize_custom, normalize_standard
from transformers import AutoTokenizer
from transformers.models.whisper.english_normalizer import (
    BasicTextNormalizer,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    references = df["reference"].fillna("").tolist()
    predictions = df["prediction"].fillna("").tolist()

    whisper_basic = BasicTextNormalizer()
    whisper_english = AutoTokenizer.from_pretrained("openai/whisper-tiny.en").normalize

    strategies = {
        "Raw": lambda x: x,
        "Lowercase": normalize_basic,
        "Lowercase + punctuation": normalize_standard,
        "Custom product rules": normalize_custom,
        "Whisper Basic": whisper_basic,
        "Whisper English": whisper_english,
    }

    print("\nNormalization comparison")
    print("-" * 45)

    for name, normalize in strategies.items():
        refs = [normalize(x) for x in references]
        preds = [normalize(x) for x in predictions]

        score = wer(refs, preds)
        print(f"{name:28s} WER: {score:.2%}")


if __name__ == "__main__":
    main()
