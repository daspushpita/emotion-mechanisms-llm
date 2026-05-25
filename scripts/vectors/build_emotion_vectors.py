from __future__ import annotations
import json, h5py
import numpy as np
import shutil
import sys
from pathlib import Path
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import emotion_mechanisms.config as cfg
from emotion_mechanisms.model_loader import load_model_and_tokenizer
from emotion_mechanisms.hooks import ActivationExtractor, _get_layers

#Load the dataset and group by emotion
def load_and_group_dataset(emotional_dataset: Path, neutral_dataset: Path) -> tuple[dict[str, list[str]], list[str]]:
    emotions = {emo: [] for emo in cfg.EMOTIONS}
    neutral = []
    with open(emotional_dataset, "r", encoding="utf-8") as f_emo:
        for line in f_emo:
            row = json.loads(line)
            emotions[row["emotion"]].append(row["generated_text"])
    
    with open(neutral_dataset, "r", encoding="utf-8") as f_neu:
        for line in f_neu:
            row = json.loads(line)
            neutral.append(row["generated_text"])
    return emotions, neutral

def _load_and_trim_datasets(max_stories: int | None):
    emotions, neutral = load_and_group_dataset(cfg.EMOTIONAL_STORIES_DATASET, cfg.NEUTRAL_STORIES_DATASET)
    if max_stories is not None:
        emotions = {emo: stories[:max_stories] for emo, stories in emotions.items()}
        neutral = neutral[:max_stories]
    return emotions, neutral

def _setup_model_and_extractor():
    #Load the model and the tokenizer
    model, tokenizer, runtime = load_model_and_tokenizer(model_name="hf",
                                analysis=True, analysis_model=cfg.ANALYSIS_MODEL_32B)

    #Set up the activation extractor
    n_layers = len(_get_layers(model))
    
    if cfg.GIVEN_LAYER_LIST:
        layer_indices = [i for i in cfg.LAYER_INDICES_32B if i < n_layers]
    else:
        layer_indices = list(range(0, n_layers, cfg.LAYER_SAMPLE_STRIDE))
    extractor = ActivationExtractor(model, layer_indices, cfg.TOKEN_POSITION)
    input_device = runtime.get("input_device", next(model.parameters()).device)
    return model, tokenizer, extractor, layer_indices, input_device

def _count_existing_stories(f_out: h5py.File, path_prefix: str, layer_indices: list) -> int:
    key = f"{path_prefix}/layer_{layer_indices[0]}"
    return f_out[key].shape[0] if key in f_out else 0

def _append_story_to_hdf5(f_out: h5py.File, path_prefix: str, story_activations: dict, layer_indices: list) -> None:
    """Append one story's activations into resizable HDF5 datasets."""
    for layer_idx in layer_indices:
        vec = story_activations[layer_idx]  # shape: (hidden_dim,)
        key = f"{path_prefix}/layer_{layer_idx}"
        if key in f_out:
            ds = f_out[key]
            n = ds.shape[0]
            ds.resize(n + 1, axis=0)
            ds[n] = vec
        else:
            f_out.create_dataset(key, data=vec[np.newaxis, :], maxshape=(None, vec.shape[0]), chunks=True)

def _checkpoint_path() -> Path | None:
    checkpoint_path = cfg.ACTIVATIONS_CHECKPOINT_PATH
    return None if checkpoint_path is None else Path(checkpoint_path)

def _flush_and_maybe_checkpoint(f_out: h5py.File, story_count: int) -> None:
    flush_every = max(int(cfg.ACTIVATIONS_FLUSH_EVERY), 1)
    if story_count % flush_every != 0:
        return

    f_out.flush()

    checkpoint_every = max(int(cfg.ACTIVATIONS_CHECKPOINT_EVERY), 1)
    checkpoint_path = _checkpoint_path()
    if checkpoint_path is None or story_count % checkpoint_every != 0:
        return

    activations_path = Path(cfg.ACTIVATIONS_PATH)
    if checkpoint_path == activations_path:
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(activations_path, checkpoint_path)

def _finalize_outputs(f_out: h5py.File) -> None:
    f_out.flush()

    checkpoint_path = _checkpoint_path()
    if checkpoint_path is None:
        return

    activations_path = Path(cfg.ACTIVATIONS_PATH)
    if checkpoint_path == activations_path:
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(activations_path, checkpoint_path)

def _extract_and_save_emotional_activations(f_out: h5py.File, emotions, tokenizer, input_device, extractor, layer_indices, story_count: int) -> int:
    total_emotional = sum(len(stories) for stories in emotions.values())
    already_done = sum(_count_existing_stories(f_out, f"emotional/{emo}", layer_indices) for emo in emotions)
    with tqdm(total=total_emotional, initial=already_done, desc="Emotional stories") as pbar:
        for emotion, stories in emotions.items():
            n_done = _count_existing_stories(f_out, f"emotional/{emotion}", layer_indices)
            for story in stories[n_done:]:
                inputs = tokenizer(story, return_tensors="pt", truncation=True, max_length=2048).to(input_device)
                story_activations = dict(extractor.extract(**inputs))
                _append_story_to_hdf5(f_out, f"emotional/{emotion}", story_activations, layer_indices)
                story_count += 1
                _flush_and_maybe_checkpoint(f_out, story_count)
                pbar.set_postfix(emotion=emotion)
                pbar.update(1)
    return story_count

def _extract_and_save_neutral_activations(f_out: h5py.File, neutral, tokenizer, input_device, extractor, layer_indices, story_count: int) -> int:
    n_done = _count_existing_stories(f_out, "neutral", layer_indices)
    for story in tqdm(neutral[n_done:], total=len(neutral), initial=n_done, desc="Neutral stories"):
        inputs = tokenizer(story, return_tensors="pt", truncation=True, max_length=2048).to(input_device)
        story_activations = dict(extractor.extract(**inputs))
        _append_story_to_hdf5(f_out, "neutral", story_activations, layer_indices)
        story_count += 1
        _flush_and_maybe_checkpoint(f_out, story_count)
    return story_count

def main(max_stories: int | None = None):
    # Loading the dataset
    emotions, neutral = _load_and_trim_datasets(max_stories=max_stories)

    #Load the model and the tokenizer and set up the activation extractor
    model, tokenizer, extractor, layer_indices, input_device = _setup_model_and_extractor()

    activations_path = Path(cfg.ACTIVATIONS_PATH)
    activations_path.parent.mkdir(parents=True, exist_ok=True)

    story_count = 0
    with h5py.File(activations_path, "a") as f_out:
        story_count = _extract_and_save_emotional_activations(
            f_out, emotions, tokenizer, input_device, extractor, layer_indices, story_count)
        
        story_count = _extract_and_save_neutral_activations(
            f_out, neutral, tokenizer, input_device, extractor, layer_indices, story_count)
        
        _finalize_outputs(f_out)


if __name__ == "__main__":
    main()
