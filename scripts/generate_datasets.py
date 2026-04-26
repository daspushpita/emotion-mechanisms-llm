from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _apply_env_overrides() -> None:
    import emotion_mechanisms.config as cfg
    import emotion_mechanisms.model_loader as model_loader

    hf_model_id = os.environ.get("HF_MODEL_ID")
    if hf_model_id:
        model_loader.HF_MODEL = hf_model_id

    topics_path = os.environ.get("DATASET_TOPICS_PATH")
    if topics_path:
        cfg.TOPICS_PATH = Path(topics_path)

    processed_dir = os.environ.get("DATASET_PROCESSED_DIR")
    if processed_dir:
        cfg.PROCESSED_DIR = Path(processed_dir)

    stories_per_batch = os.environ.get("DATASET_STORIES_PER_BATCH")
    if stories_per_batch:
        cfg.N_STORIES_PER_BATCH = int(stories_per_batch)

    samples_per_topic_emotion = os.environ.get("DATASET_SAMPLES_PER_TOPIC_EMOTION")
    if samples_per_topic_emotion:
        cfg.N_SAMPLES_PER_TOPIC_EMOTION = int(samples_per_topic_emotion)

    max_new_tokens = os.environ.get("DATASET_MAX_NEW_TOKENS")
    if max_new_tokens:
        cfg.MAX_NEW_TOKENS = int(max_new_tokens)


def _load_generator_module():
    module_path = PROJECT_ROOT / "scripts" / "my_dataset" / "generate_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_dataset_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load dataset generator from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _apply_env_overrides()
    module = _load_generator_module()
    module.main()


if __name__ == "__main__":
    main()
