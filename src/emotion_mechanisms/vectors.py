from __future__ import annotations

import h5py
import numpy as np
import sys
from tqdm import tqdm
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import emotion_mechanisms.config as cfg

PROBES_DIR = cfg.RESULTS_DIR / "probes"


class EmotionVectorExtractor:

    def __init__(self, activations_path: Path = cfg.ACTIVATIONS_PATH, 
                layer_indices: list[int] | None = None,
                pca_variance_threshold: float = 0.50):
        self.activations_path = activations_path
        self.layer_indices = layer_indices  # resolved after load_activations if None
        self.activations = {}               # {layer_idx: {"emotional": {emotion: ndarray}, "neutral": ndarray}}
        self.mean_differences = {}          # {layer_idx: {emotion: ndarray}}
        self.pca_variance_threshold = pca_variance_threshold
        
    def load_activations(self):
        with h5py.File(self.activations_path, "r") as fin:
            if self.layer_indices is None:
                self.layer_indices = sorted(
                    int(k.replace("layer_", "")) for k in fin[f"emotional/{cfg.EMOTIONS[0]}"].keys()
                )

            for layer_idx in self.layer_indices:
                self.activations[layer_idx] = {"emotional": {}, "neutral": None}
                for emotion in cfg.EMOTIONS:
                    self.activations[layer_idx]["emotional"][emotion] = fin[f"emotional/{emotion}/layer_{layer_idx}"][:]
                self.activations[layer_idx]["neutral"] = fin[f"neutral/layer_{layer_idx}"][:]

    def normalize(self, v, eps=1e-8):
        return v / (np.linalg.norm(v) + eps)

    def pca_denoise(self, layer: int) -> dict[str, np.ndarray]:
        """Fit PCA on neutral activations and determine the number of components
        needed to retain the specified variance threshold. 
        And project out all the neutral PCA components from the emotional activations, to get the "PCA-denoised" emotional activations.
        """
        #First fit PCA on the neutral activations for this layer, and determine how many components we need to retain to preserve the specified variance threshold.
        neutral_activations = self.activations[layer]["neutral"]
        pca = PCA()
        pca.fit(neutral_activations)
        running_sum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.searchsorted(running_sum, self.pca_variance_threshold) + 1
        print(f"Layer {layer}: Retaining {n_components} PCA components to preserve {self.pca_variance_threshold*100:.1f}% variance")
        neutral_pca_directions = pca.components_[:n_components]
        self.neutral_pca_directions = neutral_pca_directions
        
        # This is the centroid of all emotional activations for this layer. 
        # This first takes the mean activation for a certain emotion across all the stories,
        # and then this takes the mean of those emotion centroids to get the overall emotional centroid. 

        all_emotion_mean = np.mean([np.mean(self.activations[layer]["emotional"][emotion], axis=0) for emotion in self.activations[layer]["emotional"]], axis=0)
        
        processed_denoised_emotions = {}
        
        for emotion in cfg.EMOTIONS:
            # Mean emotional vector for this emotion and layer (i.e. the mean across all the stories)...
            mean_emotional_vector = np.mean(self.activations[layer]["emotional"][emotion], axis=0)
            emotion_direction = mean_emotional_vector - all_emotion_mean
            
            # Now we project out the neutral PCA directions 
            # from this mean-centered emotional vector, 
            # to get the PCA-denoised emotional vector 
            # for this emotion and layer.
            for neutral_dir in neutral_pca_directions:
                emotion_direction -= np.dot(emotion_direction, neutral_dir) * neutral_dir
            
            processed_denoised_emotions[emotion] = self.normalize(emotion_direction)
        return processed_denoised_emotions

    def mean_diff(self, layer: int) -> dict[str, np.ndarray]:
        # for each emotion: mean(emotional) - mean(neutral), optionally PCA-denoised
        self.mean_differences[layer] = {}
        neutral_mean = self.activations[layer]["neutral"].mean(axis=0)
        for emotion in cfg.EMOTIONS:
            emotional_mean = self.activations[layer]["emotional"][emotion].mean(axis=0)
            self.mean_differences[layer][emotion] = emotional_mean - neutral_mean
        return self.mean_differences[layer]

    def probe_metrics(self, layer: int) -> dict[str, float]:
        # Binary probe per emotion: this emotion vs. all other emotions.
        PROBE_SCORING = {"accuracy": "accuracy",
                        "balanced_accuracy": "balanced_accuracy",
                        "f1": "f1",
                        "average_precision": "average_precision",
                        "roc_auc": "roc_auc"}
        metrics = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
        
        for emotion in tqdm(cfg.EMOTIONS, desc=f"Probe directions layer {layer}", unit="emotion"):
            x_pos = self.activations[layer]["emotional"][emotion]
            x_neg = [self.activations[layer]["emotional"][other_emotion] for other_emotion in cfg.EMOTIONS if other_emotion != emotion]
        
            x_neg = np.vstack(x_neg)
            X = np.vstack([x_pos, x_neg])
            y = np.array([1] * len(x_pos) + [0] * len(x_neg))

            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            scores = cross_validate(clf, X, y, cv=cv, scoring=PROBE_SCORING)
            
            majority_baseline = max(np.mean(y == 0), np.mean(y == 1))
            metrics[emotion] = {metric: float(np.mean(scores[f"test_{metric}"])) for metric in PROBE_SCORING}
            metrics[emotion]["majority_baseline"] = float(majority_baseline)
            metrics[emotion]["positive_rate"] = float(np.mean(y))

        return metrics


    def probe_directions(self, layer: int) -> dict[str, np.ndarray]:
        directions = {}
        for emotion in tqdm(cfg.EMOTIONS, desc=f"Probe directions layer {layer}", unit="emotion"):
            x_pos = self.activations[layer]["emotional"][emotion]
            x_neg = [self.activations[layer]["emotional"][other_emotion] for other_emotion in cfg.EMOTIONS
                if other_emotion != emotion]

            x_neg = np.vstack(x_neg)
            X = np.vstack([x_pos, x_neg])
            y = np.array([1] * len(x_pos) + [0] * len(x_neg))

            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X, y)
            w = clf.coef_[0]
            directions[emotion] = w / (np.linalg.norm(w) + 1e-8)

        return directions

    def layer_sweep(self, layers: list[int]) -> dict[int, dict[str, float]]:
        return {layer: self.probe_metrics(layer) for layer in layers}

    def save(self, directions: dict[str, np.ndarray], path: Path = PROBES_DIR) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for emotion, direction in directions.items():
            np.save(path / f"{emotion}.npy", direction)


def main(activations_path: Path = cfg.ACTIVATIONS_PATH, 
                layer_indices: list[int] | None = None,
                pca_variance_threshold: float = 0.50) -> None:
    
    extractor = EmotionVectorExtractor(activations_path=activations_path,
                layer_indices=layer_indices,
                pca_variance_threshold=pca_variance_threshold)
    
    extractor.load_activations()
    if cfg.GIVEN_LAYER_LIST:
        sweep_layers = cfg.LAYER_INDICES_32B
    else:
        sweep_layers = list(range(0, max(extractor.activations.keys()) + 1, cfg.LAYER_SAMPLE_STRIDE))

    all_metrics = extractor.layer_sweep(sweep_layers)
    best_layer = max(sweep_layers, key=lambda layer: np.mean([m["balanced_accuracy"] for m in all_metrics[layer].values()]))

    print(f"Best layer: {best_layer}")
    print(f"Probe metrics: {all_metrics[best_layer]}")
    
    print(f"All metrics: {all_metrics}")

    directions = extractor.mean_diff(best_layer)
    extractor.save(directions)


if __name__ == "__main__":
    main()
