import sys, os
import json
import numpy as np
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    

positive_emotions = ["happy", "inspired", "loving", "proud", "calm"]
negative_emotions = ["afraid", "angry", "desperate", "nervous", "sad"]
compliance_emotions = ["approval_seeking", "validation_seeking", "people_pleasing"]
distress_emotions = ["ashamed", "socially_anxious", "conflict_avoidant"]

conflict_avoidance_all = compliance_emotions + distress_emotions

LAYER = 40

data_path = PROJECT_ROOT / "results" / "probes" / "v2"

core_dir     = data_path / f"layer_{LAYER}" / "pca_denoised"
conflict_dir = data_path / "all_layers_additional" / f"layer_{LAYER}" / "pca_denoised"

result_path = PROJECT_ROOT / "results" / "qwen_v2" / "cluster_data"

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate emotion vector geometry and save steering directions")
    parser.add_argument("--positive_emotions", type=str, nargs='+', default=positive_emotions, help="List of positive emotions")
    parser.add_argument("--negative_emotions", type=str, nargs='+', default=negative_emotions, help="List of negative emotions")
    parser.add_argument("--compliance_emotions", type=str, nargs='+', default=compliance_emotions, help="List of compliance-related emotions")
    parser.add_argument("--distress_emotions", type=str, nargs='+', default=distress_emotions, help="List of distress-related emotions")
    parser.add_argument("--layer", type=int, default=LAYER, help="Layer index to analyze")
    parser.add_argument("--core_dir", type=str, default=core_dir, help="Directory containing core emotion vectors")
    parser.add_argument("--conflict_dir", type=str, default=conflict_dir, help="Directory containing conflict-avoidance emotion vectors")
    parser.add_argument("--result_path", type=str, default=result_path, help="Directory to save evaluation results and steering directions")
    args = parser.parse_args()
    return args

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
    args = get_args()

    core_dir     = Path(args.core_dir)
    conflict_dir = Path(args.conflict_dir)
    result_path  = Path(args.result_path)
    positive_emotions      = args.positive_emotions
    negative_emotions      = args.negative_emotions
    compliance_emotions    = args.compliance_emotions
    distress_emotions      = args.distress_emotions
    conflict_avoidance_all = compliance_emotions + distress_emotions

    layer_dir = result_path / f"layer_{args.layer}"
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
    print(f"RESULTS FOR LAYER {args.layer}")
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

    print(f"pure_warmth vs compliance : {cos(pure_warmth, compliance_mean):+.4f}  (target: ~0.0)")
    print(f"pure_warmth vs positive   : {cos(pure_warmth, positive_mean):+.4f}")
    print(f"pure_warmth vs negative   : {cos(pure_warmth, negative_mean):+.4f}")
    print(f"pure_warmth vs distress   : {cos(pure_warmth, distress_mean):+.4f}")

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

    np.save(layer_dir / "steering_direction_positive_centroid.npy", positive_mean)
    print(f"saved -> {layer_dir / 'steering_direction_positive_centroid.npy'}")

    np.save(layer_dir / "steering_direction_pure_compliance.npy", pure_compliance)
    print(f"saved -> {layer_dir / 'steering_direction_pure_compliance.npy'}")

    # -- direction 5: distress ⊥ positive -----------------
    _section("DIRECTION 5: DISTRESS ⊥ POSITIVE")
    distress_proj = distress_mean - np.dot(distress_mean, positive_mean) * positive_mean
    distress_proj = normalize(distress_proj)

    print(f"distress_proj vs positive : {cos(distress_proj, positive_mean):+.4f}  (target: ~0.0)")
    print(f"distress_proj vs negative : {cos(distress_proj, negative_mean):+.4f}")
    print(f"distress_proj vs distress_mean : {cos(distress_proj, distress_mean):+.4f}")

    print(f"\n  {'emotion':<23}  {'vs distress_proj':>18}")
    print(f"{'-'*23}  {'-'*18}")
    for emotion in distress_emotions:
        vec = load_vec(conflict_dir, emotion)
        print(f"{emotion:<23}  {cos(vec, distress_proj):>+18.4f}")

    np.save(layer_dir / "steering_direction_distress_proj.npy", distress_proj)
    print(f"saved -> {layer_dir / 'steering_direction_distress_proj.npy'}")

    # -- direction 6: approval ⊥ {positive, distress} -----------------
    # Gram-Schmidt: first orthogonalize distress to positive, then project both out of approval
    _section("DIRECTION 6: APPROVAL ⊥ {POSITIVE, DISTRESS}")
    distress_orth = normalize(distress_mean - np.dot(distress_mean, positive_mean) * positive_mean)
    approval_proj = compliance_mean - np.dot(compliance_mean, positive_mean) * positive_mean
    approval_proj = approval_proj - np.dot(approval_proj, distress_orth) * distress_orth
    approval_proj = normalize(approval_proj)

    print(f"approval_proj vs positive      : {cos(approval_proj, positive_mean):+.4f}  (target: ~0.0)")
    print(f"approval_proj vs distress_orth : {cos(approval_proj, distress_orth):+.4f}  (target: ~0.0)")
    print(f"approval_proj vs negative      : {cos(approval_proj, negative_mean):+.4f}")
    print(f"approval_proj vs distress_mean : {cos(approval_proj, distress_mean):+.4f}")

    print(f"\n  {'emotion':<23}  {'vs approval_proj':>18}")
    print(f"{'-'*23}  {'-'*18}")
    for emotion in compliance_emotions:
        vec = load_vec(conflict_dir, emotion)
        print(f"{emotion:<23}  {cos(vec, approval_proj):>+18.4f}")

    np.save(layer_dir / "steering_direction_approval_proj.npy", approval_proj)
    print(f"saved -> {layer_dir / 'steering_direction_approval_proj.npy'}")

    # -- direction 7: distress ⊥ {positive, approval} -----------------
    # Gram-Schmidt: first orthogonalize approval to positive, then project both out of distress
    _section("DIRECTION 7: DISTRESS ⊥ {POSITIVE, APPROVAL}")
    compliance_orth = normalize(compliance_mean - np.dot(compliance_mean, positive_mean) * positive_mean)
    distress_proj_2 = distress_mean - np.dot(distress_mean, positive_mean) * positive_mean
    distress_proj_2 = distress_proj_2 - np.dot(distress_proj_2, compliance_orth) * compliance_orth
    distress_proj_2 = normalize(distress_proj_2)

    print(f"distress_proj_2 vs positive        : {cos(distress_proj_2, positive_mean):+.4f}  (target: ~0.0)")
    print(f"distress_proj_2 vs compliance_orth : {cos(distress_proj_2, compliance_orth):+.4f}  (target: ~0.0)")
    print(f"distress_proj_2 vs negative        : {cos(distress_proj_2, negative_mean):+.4f}")
    print(f"distress_proj_2 vs distress_mean   : {cos(distress_proj_2, distress_mean):+.4f}")

    print(f"\n  {'emotion':<23}  {'vs distress_proj_2':>18}")
    print(f"{'-'*23}  {'-'*18}")
    for emotion in distress_emotions:
        vec = load_vec(conflict_dir, emotion)
        print(f"{emotion:<23}  {cos(vec, distress_proj_2):>+18.4f}")

    np.save(layer_dir / "steering_direction_distress_proj_2.npy", distress_proj_2)
    print(f"saved -> {layer_dir / 'steering_direction_distress_proj_2.npy'}")

    # ── Metadata JSON ─────────────────────────────────────────────
    metadata = {
        "layer": args.layer,
        "positive_emotions": positive_emotions,
        "negative_emotions": negative_emotions,
        "approval_emotions": compliance_emotions,
        "distress_emotions": distress_emotions,
        "cosines": {
            "approval_vs_positive":               cos(compliance_mean, positive_mean),
            "approval_vs_negative":               cos(compliance_mean, negative_mean),
            "approval_vs_distress":               cos(compliance_mean, distress_mean),
            "distress_vs_positive":               cos(distress_mean, positive_mean),
            "distress_vs_negative":               cos(distress_mean, negative_mean),
            "pure_compliance_vs_positive":        cos(pure_compliance, positive_mean),
            "pure_compliance_vs_distress":        cos(pure_compliance, distress_mean),
            "pure_warmth_vs_compliance":          cos(pure_warmth, compliance_mean),
            "pure_warmth_vs_positive":            cos(pure_warmth, positive_mean),
            "valence_contrast_vs_positive":       cos(valence_dir, positive_mean),
            "valence_contrast_vs_negative":       cos(valence_dir, negative_mean),
            "approval_proj_vs_positive":          cos(approval_proj, positive_mean),
            "approval_proj_vs_distress_orth":     cos(approval_proj, distress_orth),
            "distress_proj_2_vs_positive":        cos(distress_proj_2, positive_mean),
            "distress_proj_2_vs_compliance_orth": cos(distress_proj_2, compliance_orth),
        }
    }
    metadata_path = layer_dir / "direction_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"saved -> {metadata_path}")


if __name__ == "__main__":
    main()
