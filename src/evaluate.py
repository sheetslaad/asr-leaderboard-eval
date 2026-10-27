
import torch
from transformers import pipeline

MODEL = "openai/whisper-tiny.en"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")
print(f"Loading model: {MODEL}")

transcriber = pipeline(
    task="automatic-speech-recognition",
    model=MODEL,
    device=device,
)

print("Model loaded successfully!")