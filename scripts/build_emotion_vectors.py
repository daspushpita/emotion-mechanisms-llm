from __future__ import annotations
import json, h5py
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emotion_mechanisms.config import (
    ALL_EMOTIONS,
    EMOTIONS,
    EMOTIONAL_STORIES_DATASET,
    NEUTRAL_STORIES_DATASET,
    ACTIVATIONS_PATH,
    ANALYSIS_MODEL_7B,
    LAYER_SAMPLE_STRIDE,
    TOKEN_POSITION,
)

from emotion_mechanisms.model_loader import load_model_and_tokenizer
from emotion_mechanisms.hooks import ActivationExtractor

emotional_dataset = EMOTIONAL_STORIES_DATASET
neutral_dataset = NEUTRAL_STORIES_DATASET

#Load the dataset and group by emotion
def load_and_group_dataset(emotional_dataset: Path, neutral_dataset: Path) -> tuple[dict[str, list[str]], list[str]]:
    emotions = {emo: [] for emo in EMOTIONS}
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
    
def main():
    # Loading the dataset
    emotions, neutral = load_and_group_dataset(emotional_dataset, neutral_dataset)

    #Load the model and the tokenizer
    model, tokenizer, runtime = load_model_and_tokenizer(model_name="hf", 
                                analysis=True, analysis_model=ANALYSIS_MODEL_7B)
    
    #Set up the activation extractor
    layer_indices = list(range(0, len(model.model.layers), LAYER_SAMPLE_STRIDE))
    extractor = ActivationExtractor(model, layer_indices)
    
    emotional_activations = {emo: [] for emo in EMOTIONS}
    neutral_activations = []

    def _select_token(act, position: str):
        # act: (1, seq_len, hidden_dim) CPU tensor
        act = act.float()
        return act[0, -1, :].numpy() if position == "last" else act[0].mean(dim=0).numpy()

    #Extract activations for emotional stories
    for emotion, stories in emotions.items():
        for story in stories:
            inputs = tokenizer(story, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            activations = extractor.extract(**inputs)
            emotional_activations[emotion].append({l: _select_token(a, TOKEN_POSITION) for l, a in activations.items()})

    #Extract activations for neutral stories
    for story in neutral:
        inputs = tokenizer(story, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        activations = extractor.extract(**inputs)
        neutral_activations.append({l: _select_token(a, TOKEN_POSITION) for l, a in activations.items()})

    #Save the activations to disk — one dataset per (emotion, layer) of shape (n_stories, hidden_dim)
    ACTIVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(ACTIVATIONS_PATH, "w") as fout:
        for emotion, vecs in emotional_activations.items():
            for layer_idx in layer_indices:
                fout.create_dataset(f"emotional/{emotion}/layer_{layer_idx}", data=np.stack([v[layer_idx] for v in vecs]))
        for layer_idx in layer_indices:
            fout.create_dataset(f"neutral/layer_{layer_idx}", data=np.stack([v[layer_idx] for v in neutral_activations]))

if __name__ == "__main__":
    main()

