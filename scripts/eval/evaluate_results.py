import sys, os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    

positive_emotions = ["happy", "inspired", "loving", "proud", "calm"]
negative_emotions = ["afraid", "angry", "desperate", "guilty", "nervous", "sad"]

# Split taxonomy based on observed geometry
compliance_emotions = [
    "approval_seeking", "validation_seeking",
    "people_pleasing", "deferential", "obsequious",
]
distress_emotions = [
    "ashamed", "socially_anxious",
    "conflict_avoidant", "submissive",
]
conflict_avoidance_all = compliance_emotions + distress_emotions

LAYER = 48

data_path = PROJECT_ROOT / "results" / "gemma" / "probes"

core_dir     = data_path / "all_layers" / f"layer_{LAYER}" / "pca_denoised"
conflict_dir = data_path / "all_layers_additional" / f"layer_{LAYER}" / "pca_denoised"

result_path = PROJECT_ROOT / "results" / "gemma" / "cluster_data"

def normalize(v, eps=1.e-8):
    return v / (np.linalg.norm(v) + eps)

def load_vec(probe_dir, emotion):
    path = probe_dir / f"{emotion}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing probe file: {path}")
    return normalize(np.load(path))

def cos(a, b):
    return float(np.dot(a, b))

def _section(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    
def main():
    layer_dir = result_path / f"layer_{LAYER}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    # ── Load valence means ────────────────────────────────────────
    positive_mean = normalize(np.stack([load_vec(core_dir, e) for e in positive_emotions]).mean(axis=0))
    negative_mean = normalize(np.stack([load_vec(core_dir, e) for e in negative_emotions]).mean(axis=0))

    # ── Load sub-cluster means ────────────────────────────────────
    compliance_mean = normalize(np.stack([load_vec(conflict_dir, e) for e in compliance_emotions]).mean(axis=0))
    distress_mean   = normalize(np.stack([load_vec(conflict_dir, e) for e in distress_emotions]).mean(axis=0))
    conflict_mean   = normalize(np.stack([load_vec(conflict_dir, e) for e in conflict_avoidance_all]).mean(axis=0))

    # ── Section 1: Full-mean baseline (replicates original output) ─
    print(f"\n{'='*60}")
    print(f"RESULTS FOR LAYER {LAYER}")
    print('='*60)
    _section("FULL TAXONOMY MEAN (baseline)")
    print(f"conflict_mean vs positive : {cos(conflict_mean, positive_mean):+.4f}")
    print(f"conflict_mean vs negative : {cos(conflict_mean, negative_mean):+.4f}")

    # sub-cluster geometry
    _section("SUB-CLUSTER GEOMETRY")
    print(f"{'direction':<22} {'vs pos':>8} {'vs neg':>8} {'vs other':>11}")
    print(f"{'compliance_mean':<22} {cos(compliance_mean, positive_mean):>8.4f} {cos(compliance_mean, negative_mean):>8.4f} {cos(compliance_mean, distress_mean):>11.4f}")
    print(f"{'distress_mean':<22} {cos(distress_mean, positive_mean):>8.4f} {cos(distress_mean, negative_mean):>8.4f} {cos(distress_mean, compliance_mean):>11.4f}")

    # per-emotion
    _section("per-emotion breakdown")
    print(f"{'emotion':<23} {'vs pos':>8} {'vs neg':>8} {'vs compliance':>13} {'vs distress':>11}")
    for emotion in conflict_avoidance_all:
        vec = load_vec(conflict_dir, emotion)
        tag = "*" if emotion in distress_emotions else ""
        print(f"{emotion:<23} {cos(vec, positive_mean):>8.4f} {cos(vec, negative_mean):>8.4f} {cos(vec, compliance_mean):>13.4f} {cos(vec, distress_mean):>11.4f}  {tag}")
    print("(* = distress cluster)")

    # steering checks
    _section("steering viability")
    c2d = cos(compliance_mean, distress_mean)
    c2p = cos(compliance_mean, positive_mean)
    c2n = cos(compliance_mean, negative_mean)
    print(f"compliance vs distress : {c2d:.4f}")
    print(f"compliance vs positive : {c2p:.4f}")
    print(f"compliance vs negative : {c2n:.4f}")

    if c2d < 0.15:
        print("  -> subspaces look separable")
    elif c2d < 0.35:
        print(" -> some overlap, worth checking distress probes after steering")
    else:
        print("-> too much overlap, reconsider the split")

    if c2p >= 0.55:
        print("-> compliance sits close to positive valence, collateral risk")

    _section("PURE COMPLIANCE DIRECTION (positive-valence projected out)")
    pure_compliance = compliance_mean - np.dot(compliance_mean, positive_mean) * positive_mean
    pure_compliance = normalize(pure_compliance)

    print(f"pure_compliance vs positive : {cos(pure_compliance, positive_mean):+.4f}  (target: ~0.0)")
    print(f"pure_compliance vs negative : {cos(pure_compliance, negative_mean):+.4f}")
    print(f"pure_compliance vs distress : {cos(pure_compliance, distress_mean):+.4f}")

    # Per-emotion alignment with pure direction
    print(f"\n  {'emotion':<23}  {'vs pure_compliance':>18}")
    print(f"{'-'*23}  {'-'*18}")
    for emotion in compliance_emotions:
        vec = load_vec(conflict_dir, emotion)
        print(f"{emotion:<23}  {cos(vec, pure_compliance):>+18.4f}")

    # save compliance direction
    np.save(layer_dir / "steering_direction_compliance.npy", compliance_mean)
    print(f"saved -> {layer_dir / 'steering_direction_compliance.npy'}")

    _section("PURE WARMTH DIRECTION (compliance-valence projected out)")
    pure_warmth = positive_mean - np.dot(compliance_mean, positive_mean) * compliance_mean
    pure_warmth = normalize(pure_warmth)

    print(f"pure_warmth vs positive : {cos(pure_warmth, positive_mean):+.4f}  (target: ~0.0)")
    print(f"pure_warmth vs negative : {cos(pure_warmth, negative_mean):+.4f}")
    print(f"pure_warmth vs distress : {cos(pure_warmth, distress_mean):+.4f}")

    # Per-emotion alignment with pure direction
    print(f"\n  {'emotion':<23}  {'vs pure_warmth':>18}")
    print(f"{'-'*23}  {'-'*18}")
    for emotion in compliance_emotions:
        vec = load_vec(conflict_dir, emotion)
        print(f"{emotion:<23}  {cos(vec, pure_warmth):>+18.4f}")

    # save warmth direction
    np.save(layer_dir / "steering_direction_pure_warmth.npy", pure_warmth)
    print(f"saved -> {layer_dir / 'steering_direction_pure_warmth.npy'}")

    # ── Positive valence direction ────────────────────────────────
    _section("POSITIVE VALENCE DIRECTION")
    valence_dir = normalize(positive_mean - negative_mean)

    print(f"valence_dir vs positive_mean : {cos(valence_dir, positive_mean):+.4f}")
    print(f"valence_dir vs negative_mean : {cos(valence_dir, negative_mean):+.4f}")
    print(f"valence_dir vs compliance    : {cos(valence_dir, compliance_mean):+.4f}")
    print(f"valence_dir vs distress      : {cos(valence_dir, distress_mean):+.4f}")

    print(f"\n  {'emotion':<23}  {'vs valence_dir':>14}")
    print(f"  {'-'*23}  {'-'*14}")
    for e in positive_emotions:
        print(f"  {e:<23}  {cos(load_vec(core_dir, e), valence_dir):>+14.4f}")
    for e in negative_emotions:
        print(f"  {e:<23}  {cos(load_vec(core_dir, e), valence_dir):>+14.4f}")

    np.save(layer_dir / "steering_direction_positive_valence.npy", valence_dir)
    print(f"\nsaved -> {layer_dir / 'steering_direction_positive_valence.npy'}")

    np.save(layer_dir / "steering_direction_pure_compliance.npy", pure_compliance)
    print(f"saved -> {layer_dir / 'steering_direction_pure_compliance.npy'}")


if __name__ == "__main__":
    main()
