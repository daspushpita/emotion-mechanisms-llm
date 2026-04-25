from __future__ import annotations
import json, h5py
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import emotion_mechanisms.config as cfg
from emotion_mechanisms.model_loader import load_model_and_tokenizer
from emotion_mechanisms.hooks import ActivationExtractor

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
                                analysis=True, analysis_model=cfg.ANALYSIS_MODEL_7B)

    #Set up the activation extractor
    layer_indices = list(range(0, len(model.model.layers), cfg.LAYER_SAMPLE_STRIDE))
    extractor = ActivationExtractor(model, layer_indices, cfg.TOKEN_POSITION)
    input_device = runtime.get("input_device", next(model.parameters()).device)
    return model, tokenizer, extractor, layer_indices, input_device

def _extract_emotional_activations(emotions, tokenizer, input_device, extractor, layer_indices):
    emotional_activations = {emo: [] for emo in cfg.EMOTIONS}
    total_emotional = sum(len(stories) for stories in emotions.values())

    with tqdm(total=total_emotional, desc="Emotional stories") as pbar:
        for emotion, stories in emotions.items():
            for story in stories:
                inputs = tokenizer(story, return_tensors="pt", truncation=True, max_length=2048).to(input_device)
                activations = extractor.extract(**inputs)
                story_activations = dict(activations)

                emotional_activations[emotion].append(story_activations)
                pbar.set_postfix(emotion=emotion)
                pbar.update(1)
    return emotional_activations

def _extract_neutral_activations(neutral, tokenizer, input_device, extractor, layer_indices):
    neutral_activations = []
    for story in tqdm(neutral, desc="Neutral stories"):
        inputs = tokenizer(story, return_tensors="pt", truncation=True, max_length=2048).to(input_device)

        activations = extractor.extract(**inputs)
        story_activations = dict(activations)

        neutral_activations.append(story_activations)
    return neutral_activations

def _save_activations(emotional_activations, neutral_activations, layer_indices):
    # Each saved dataset has shape: (number_of_stories, hidden_dim)
    cfg.ACTIVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(cfg.ACTIVATIONS_PATH, "w") as fout:
        for emotion, story_activation_list in emotional_activations.items():
            for layer_idx in layer_indices:
                layer_vectors = []
                for story_activations in story_activation_list:
                    layer_vectors.append(story_activations[layer_idx])
                layer_vectors = np.stack(layer_vectors)
                fout.create_dataset(f"emotional/{emotion}/\"layer_{layer_idx}\"".replace('\"', ''), data=layer_vectors)

        for layer_idx in layer_indices:
            layer_vectors = []
            for story_activations in neutral_activations:
                layer_vectors.append(story_activations[layer_idx])
            layer_vectors = np.stack(layer_vectors)
            fout.create_dataset(f"neutral/\"layer_{layer_idx}\"".replace('\"', ''), data=layer_vectors)
            
def main(max_stories: int | None = None):
    # Loading the dataset
    emotions, neutral = _load_and_trim_datasets(max_stories=max_stories)

    #Load the model and the tokenizer and set up the activation extractor
    model, tokenizer, extractor, layer_indices, input_device = _setup_model_and_extractor()

    # Extract activations for emotional stories
    emotional_activations = _extract_emotional_activations(emotions, tokenizer, input_device, extractor, layer_indices)
    
    # Extract activations for neutral stories
    neutral_activations = _extract_neutral_activations(neutral, tokenizer, input_device, extractor, layer_indices)

    # Save the activations to disk, each saved dataset has shape: (number_of_stories, hidden_dim)
    _save_activations(emotional_activations, neutral_activations, layer_indices)


if __name__ == "__main__":
    main()
