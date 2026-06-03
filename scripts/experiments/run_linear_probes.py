from __future__ import annotations
import numpy as np
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import emotion_mechanisms.config as cfg
import emotion_mechanisms.vectors as vectors


base_activations_path = PROJECT_ROOT / "results/gemma/activations_32b.h5"
additional_activations_path = PROJECT_ROOT / "results/gemma/activations_additionalemotions_32b.h5"
probes_dir = PROJECT_ROOT / "results/gemma/probes"


layer_indices = [x for x in range(32, 62, 1)] # Focus on later layers where emotion info is stronger, but can be adjusted as needed

def get_args():
    parser = argparse.ArgumentParser(description="Linear Probes CLI")
    parser.add_argument("--base_activations_path", type=str, default=str(base_activations_path), help="Path to base activations .h5 file")
    parser.add_argument("--additional_activations_path", type=str, default=str(additional_activations_path), help="Path to additional emotions activations .h5 file")
    parser.add_argument("--probes_dir", type=str, default=str(probes_dir), help="Directory to save probe results")
    parser.add_argument("--layer_indices", type=int, nargs='+', default=layer_indices, help="List of layer indices to analyze (e.g. --layer_indices 32 33 34 ...)")
    parser.add_argument("--core_emotions_list", type=str, nargs='+', default=cfg.BASE_EMOTIONS, help="List of core emotions to analyze (default: base emotions)")
    parser.add_argument("--additional_emotions_list", type=str, nargs='+', default=cfg.ADDITIONAL_EMOTIONS, help="List of additional emotions to analyze (default: base emotions)")
    args = parser.parse_args()
    return args

def load_and_save_metrics(emotion_vectors, layer_indices, probes_dir):
    emotion_vectors.load_activations()
    base_metrics = emotion_vectors.layer_sweep(layer_indices)
    best_layer = max(layer_indices, key=lambda layer: np.mean([m["balanced_accuracy"] for m in base_metrics[layer].values()]))

    for layer in layer_indices:
        emotion_vectors.save_vectors(emotion_vectors.mean_diff(layer),        subdir=f"layer_{layer}/mean_diff",    root=probes_dir)
        emotion_vectors.save_vectors(emotion_vectors.probe_directions(layer),  subdir=f"layer_{layer}/probe",        root=probes_dir)
        emotion_vectors.save_vectors(emotion_vectors.pca_denoise(layer),       subdir=f"layer_{layer}/pca_denoised", root=probes_dir)
        
    emotion_vectors.save_metrics(base_metrics, best_layer, root=probes_dir)
    print(f"Best layer: {best_layer}")
    best_link = probes_dir / "best_layer"
    best_link.unlink(missing_ok=True)
    best_link.symlink_to(probes_dir / f"layer_{best_layer}", target_is_directory=True)
    print(f"Saved base vectors for all {len(layer_indices)} layers. Best layer symlink -> layer_{best_layer}")
    return base_metrics, best_layer

def main():
    args = get_args()
    print("Running linear probes with the following arguments:")
    print(args)

    base_activations = Path(args.base_activations_path)
    additional_activations = Path(args.additional_activations_path)
    out_dir = Path(args.probes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = args.layer_indices

    # --- Base emotions ---
    base_emotion_vectors = vectors.EmotionVectorExtractor( activations_path=base_activations,
                            layer_indices=layers, emotions=args.core_emotions_list)
    load_and_save_metrics(base_emotion_vectors, layers, out_dir)

    # --- Additional emotions ---
    additional_out_dir = out_dir / "all_layers_additional"
    additional_out_dir.mkdir(parents=True, exist_ok=True)
    additional_emotion_vectors = vectors.EmotionVectorExtractor(activations_path=additional_activations,
                                layer_indices=layers, emotions=args.additional_emotions_list)
    load_and_save_metrics(additional_emotion_vectors, layers, additional_out_dir)

if __name__ == "__main__":
    main()