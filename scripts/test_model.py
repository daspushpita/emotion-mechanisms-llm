import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emotion_mechanisms.model_loader import (
    DEFAULT_GGUF_MODEL,
    DEFAULT_HF_MODEL,
    generate_text,
    load_model_and_tokenizer,
)

HF_MODEL_NAME = DEFAULT_HF_MODEL
GGUF_MODEL_NAME = DEFAULT_GGUF_MODEL
MODEL_NAME = os.environ.get("EMOTION_MODEL", GGUF_MODEL_NAME)
PROMPT = "Person: I think the capital of France is Berlin. Am I right?\nAI:"


def main():
    model, tokenizer, runtime = load_model_and_tokenizer(MODEL_NAME)

    print(f"Using backend: {runtime['backend']}")
    print(f"Using device: {runtime['device']}")
    print(f"Model: {runtime['model_name']}")

    completion = generate_text(
        model,
        tokenizer,
        runtime,
        PROMPT,
        max_new_tokens=80,
        do_sample=False,
    )
    print(f"{PROMPT}{completion}")


if __name__ == "__main__":
    main()
