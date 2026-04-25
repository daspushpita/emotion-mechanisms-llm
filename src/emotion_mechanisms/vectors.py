from __future__ import annotations

import h5py
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emotion_mechanisms.config import (
    EMOTIONS,
    ACTIVATIONS_PATH,
    RESULTS_DIR,
    LAYER_SAMPLE_STRIDE,
)

PROBES_DIR = RESULTS_DIR / "probes"


class EmotionVectorExtractor:

    def __init__(self, activations_path: Path = ACTIVATIONS_PATH, layer_indices: list[int] | None = None):
        self.activations_path = activations_path
        self.layer_indices = layer_indices  # resolved after load_activations if None
        self.activations = {}               # {layer_idx: {"emotional": {emotion: ndarray}, "neutral": ndarray}}
        self.mean_differences = {}          # {layer_idx: {emotion: ndarray}}

    def load_activations(self):
        with h5py.File(self.activations_path, "r") as fin:
            if self.layer_indices is None:
                max_layer = max(int(k.replace("layer_", "")) for k in fin[f"emotional/{EMOTIONS[0]}"].keys())
                self.layer_indices = list(range(0, max_layer + 1, LAYER_SAMPLE_STRIDE))

            for layer_idx in self.layer_indices:
                self.activations[layer_idx] = {"emotional": {}, "neutral": None}
                for emotion in EMOTIONS:
                    self.activations[layer_idx]["emotional"][emotion] = fin[f"emotional/{emotion}/layer_{layer_idx}"][:]
                self.activations[layer_idx]["neutral"] = fin[f"neutral/layer_{layer_idx}"][:]


    def mean_diff(self, layer: int) -> dict[str, np.ndarray]:
        # for each emotion: mean(emotional) - mean(neutral), optionally PCA-denoised
        self.mean_differences[layer] = {}
        neutral_mean = self.activations[layer]["neutral"].mean(axis=0)
        for emotion in EMOTIONS:
            emotional_mean = self.activations[layer]["emotional"][emotion].mean(axis=0)
            self.mean_differences[layer][emotion] = emotional_mean - neutral_mean
        return self.mean_differences[layer]

    def probe_accuracy(self, layer: int) -> dict[str, float]:
        # fit a binary logistic probe per emotion at this layer, return CV accuracies
        accuracies = {}
        for emotion in EMOTIONS:
            x_pos = self.activations[layer]["emotional"][emotion]
            x_neg = np.concatenate([self.activations[layer]["emotional"][e] for e in EMOTIONS if e != emotion]
                    + [self.activations[layer]["neutral"]])
            X = np.vstack([x_pos, x_neg])
            y = np.array([1] * len(x_pos) + [0] * len(x_neg))
            clf = LogisticRegression(max_iter=1000)
            accuracies[emotion] = cross_val_score(clf, X, y, cv=5).mean()
        return accuracies


    def layer_sweep(self, layers: list[int]) -> dict[int, dict[str, float]]:
        return {layer: self.probe_accuracy(layer) for layer in layers}

    def save(self, directions: dict[str, np.ndarray], path: Path = PROBES_DIR) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for emotion, direction in directions.items():
            np.save(path / f"{emotion}.npy", direction)


def main() -> None:
    extractor = EmotionVectorExtractor()
    extractor.load_activations()
    sweep_layers = [40, 45, 50]
    all_accuracies = extractor.layer_sweep(sweep_layers)
    best_layer = max(sweep_layers, key=lambda l: np.mean(list(all_accuracies[l].values())))
    print(f"Best layer: {best_layer}")
    print(f"Probe accuracies: {all_accuracies[best_layer]}")
    directions = extractor.mean_diff(best_layer)
    extractor.save(directions)


if __name__ == "__main__":
    main()
