"""
Compute per-layer average L2 norm of residual-stream activations from saved HDF5 files.
Pools over all samples (neutral + all emotions) from both baseline HDF5 files.

Output: results/baseline/residual_norms_32b.json  {layer_idx: avg_norm}
"""
import json
import argparse
from pathlib import Path
import h5py
import numpy as np


def compute_norms(h5_paths: list[Path]) -> dict[int, float]:
    norm_sum = {}
    count = {}

    for path in h5_paths:
        with h5py.File(path, "r") as f:
            # neutral group: keys are layer_0, layer_1, ...
            if "neutral" in f:
                for key in f["neutral"].keys():
                    layer = int(key.split("_")[1])
                    v = f["neutral"][key][:]  # (N, hidden_dim)
                    norms = np.linalg.norm(v, axis=1)
                    norm_sum[layer] = norm_sum.get(layer, 0.0) + norms.sum()
                    count[layer] = count.get(layer, 0) + len(norms)

            # emotional group: keys are emotion names, each contains layer_* datasets
            if "emotional" in f:
                for emotion in f["emotional"]:
                    for key in f["emotional"][emotion].keys():
                        layer = int(key.split("_")[1])
                        v = f["emotional"][emotion][key][:]
                        norms = np.linalg.norm(v, axis=1)
                        norm_sum[layer] = norm_sum.get(layer, 0.0) + norms.sum()
                        count[layer] = count.get(layer, 0) + len(norms)

    return {l: float(norm_sum[l] / count[l]) for l in sorted(norm_sum)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_paths", nargs="+", required=True,
                        help="Paths to HDF5 activation files")
    parser.add_argument("--output", type=str,
                        default="results/baseline/residual_norms_32b.json",
                        help="Output JSON path")
    args = parser.parse_args()

    h5_paths = [Path(p) for p in args.h5_paths]
    for p in h5_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    avg_norms = compute_norms(h5_paths)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(avg_norms, f, indent=2)

    print(f"Saved {len(avg_norms)} layer norms to {out_path}")
    for l, n in avg_norms.items():
        print(f"  layer {l:>2}: {n:.4f}")


if __name__ == "__main__":
    main()
