from __future__ import annotations
import json, h5py
import numpy as np
import sys
from pathlib import Path
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import emotion_mechanisms.config as cfg
import emotion_mechanisms.vectors as vectors


base_activations_path = Path("/Users/pushpita/Documents/ML_Projects/AI_Safety/emotion-mechanisms-llm/results/baseline/all_layers/activations_32b.h5")
additional_activations_path = Path("/Users/pushpita/Documents/ML_Projects/AI_Safety/emotion-mechanisms-llm/results/baseline/all_layers/activations_additionalemotions_32b.h5")
probes_dir = Path("/Users/pushpita/Documents/ML_Projects/AI_Safety/emotion-mechanisms-llm/results/probes/all_layers/v2")
probes_dir.mkdir(parents=True, exist_ok=True)

layer_indices = list(range(64))

# --- Base 12 emotions ---
base_emotion_vectors = vectors.EmotionVectorExtractor(activations_path=base_activations_path,
                                                    layer_indices=layer_indices)
base_emotion_vectors.load_activations()
base_metrics = base_emotion_vectors.layer_sweep(layer_indices)
best_layer = max(layer_indices, key=lambda layer: np.mean([m["balanced_accuracy"] for m in base_metrics[layer].values()]))

print(f"Best layer: {best_layer}")
print(f"Best layer metrics: {base_metrics[best_layer]}")

base_emotion_vectors.save_metrics(base_metrics, best_layer, root=probes_dir)

for layer in layer_indices:
    base_emotion_vectors.save_vectors(base_emotion_vectors.mean_diff(layer),        subdir=f"layer_{layer}/mean_diff",    root=probes_dir)
    base_emotion_vectors.save_vectors(base_emotion_vectors.probe_directions(layer),  subdir=f"layer_{layer}/probe",        root=probes_dir)
    base_emotion_vectors.save_vectors(base_emotion_vectors.pca_denoise(layer),       subdir=f"layer_{layer}/pca_denoised", root=probes_dir)

best_link = probes_dir / "best_layer"
best_link.unlink(missing_ok=True)
best_link.symlink_to(probes_dir / f"layer_{best_layer}", target_is_directory=True)
print(f"Saved base vectors for all {len(layer_indices)} layers. Best layer symlink -> layer_{best_layer}")


# --- Additional (conflict-avoidance) emotions ---
additional_probes_dir = Path("/Users/pushpita/Documents/ML_Projects/AI_Safety/emotion-mechanisms-llm/results/probes/all_layers_additional")
additional_probes_dir.mkdir(parents=True, exist_ok=True)

additional_emotion_vectors = vectors.EmotionVectorExtractor(activations_path=additional_activations_path,
                                                            layer_indices=layer_indices,
                                                            emotions=cfg.ALL_EMOTIONS)
additional_emotion_vectors.load_activations()
additional_metrics = additional_emotion_vectors.layer_sweep(layer_indices)
additional_best_layer = max(layer_indices, key=lambda layer: np.mean([m["balanced_accuracy"] for m in additional_metrics[layer].values()]))

print(f"Best layer (additional): {additional_best_layer}")
print(f"Best layer metrics (additional): {additional_metrics[additional_best_layer]}")

additional_emotion_vectors.save_metrics(additional_metrics, additional_best_layer, root=additional_probes_dir)

for layer in layer_indices:
    additional_emotion_vectors.save_vectors(additional_emotion_vectors.mean_diff(layer),        subdir=f"layer_{layer}/mean_diff",    root=additional_probes_dir)
    additional_emotion_vectors.save_vectors(additional_emotion_vectors.probe_directions(layer),  subdir=f"layer_{layer}/probe",        root=additional_probes_dir)
    additional_emotion_vectors.save_vectors(additional_emotion_vectors.pca_denoise(layer),       subdir=f"layer_{layer}/pca_denoised", root=additional_probes_dir)

additional_best_link = additional_probes_dir / "best_layer"
additional_best_link.unlink(missing_ok=True)
additional_best_link.symlink_to(additional_probes_dir / f"layer_{additional_best_layer}", target_is_directory=True)
print(f"Saved additional vectors for all {len(layer_indices)} layers. Best layer symlink -> layer_{additional_best_layer}")


