# -------- This is the new version of evaluate_results_v3.py --------
import sys
import json
import numpy as np
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

positive_emotions    = ["happy", "loving", "proud"]
negative_emotions    = ["afraid", "angry", "desperate", "nervous", "sad"]
compliance_emotions  = ["approval_seeking", "validation_seeking", "people_pleasing"]
distress_emotions    = ["ashamed", "socially_anxious", "conflict_avoidant"]
conflict_avoidance_all = compliance_emotions + distress_emotions

LAYER = 40

data_path    = PROJECT_ROOT / "results" / "qwen_v2" / "linear_probes_v2"
core_dir     = data_path / f"layer_{LAYER}" / "pca_denoised"
conflict_dir = data_path / "all_layers_additional" / f"layer_{LAYER}" / "pca_denoised"
result_path  = PROJECT_ROOT / "results" / "qwen_v2" / "cluster_data"


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate emotion vector geometry and save steering directions")
    parser.add_argument("--positive_emotions",   type=str, nargs='+', default=positive_emotions)
    parser.add_argument("--negative_emotions",   type=str, nargs='+', default=negative_emotions)
    parser.add_argument("--compliance_emotions", type=str, nargs='+', default=compliance_emotions)
    parser.add_argument("--distress_emotions",   type=str, nargs='+', default=distress_emotions)
    parser.add_argument("--layer",        type=int, default=LAYER)
    parser.add_argument("--core_dir",     type=str, default=core_dir)
    parser.add_argument("--conflict_dir", type=str, default=conflict_dir)
    parser.add_argument("--result_path",  type=str, default=result_path)
    parser.add_argument("--save",         type=int, default=0, help="Set to 1 to save steering directions")
    return parser.parse_args()


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)

def load_vec(probe_dir, emotion):
    path = probe_dir / f"{emotion}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing probe file: {path}")
    return normalize(np.load(path))

def cos(a, b):
    return float(np.dot(a, b))

def mean_vec(probe_dir, emotions):
    return normalize(np.stack([load_vec(probe_dir, e) for e in emotions]).mean(axis=0))

def _section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def report_direction(title, raw, original, retention_label, ref_cosines, emotions, probe_dir, col_label, save_path=None):
    _section(title)
    retention = np.linalg.norm(raw) / np.linalg.norm(original)
    print(f"norm retention ({retention_label}): {retention:.4f}")
    vec = normalize(raw)
    w = max(len(lbl) for lbl, _, _ in ref_cosines)
    for label, ref, is_target in ref_cosines:
        note = "  (target: ~0.0)" if is_target else ""
        print(f"{label:<{w}} : {cos(vec, ref):+.4f}{note}")
    col_w = max(18, len(col_label))
    print(f"\n {'emotion':<23}  {col_label:>{col_w}}")
    print(f"{'-'*23}  {'-'*col_w}")
    for emotion in emotions:
        ev = load_vec(probe_dir, emotion)
        print(f"{emotion:<23}  {cos(ev, vec):>+{col_w}.4f}")
    if save_path is not None:
        np.save(save_path, vec)
        print(f"saved -> {save_path}")
    return vec


def main():
    args = get_args()
    core_dir     = Path(args.core_dir)
    conflict_dir = Path(args.conflict_dir)
    result_path  = Path(args.result_path)
    positive_emotions    = args.positive_emotions
    negative_emotions    = args.negative_emotions
    compliance_emotions  = args.compliance_emotions
    distress_emotions    = args.distress_emotions
    conflict_avoidance_all = compliance_emotions + distress_emotions
    save = bool(args.save)

    layer_dir = result_path / f"layer_{args.layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    def sp(fname):
        return layer_dir / fname if save else None

    # ── Load means ────────────────────────────────────────────────
    positive_mean   = mean_vec(core_dir, positive_emotions)
    negative_mean   = mean_vec(core_dir, negative_emotions)
    compliance_mean = mean_vec(conflict_dir, compliance_emotions)
    distress_mean   = mean_vec(conflict_dir, distress_emotions)
    conflict_mean   = mean_vec(conflict_dir, conflict_avoidance_all)

    # ── Baseline ──────────────────────────────────────────────────
    print(f"\n{'='*60}\nRESULTS FOR LAYER {args.layer}\n{'='*60}")

    _section("FULL TAXONOMY MEAN (baseline)")
    print(f"conflict_mean vs positive : {cos(conflict_mean, positive_mean):+.4f}")
    print(f"conflict_mean vs negative : {cos(conflict_mean, negative_mean):+.4f}")

    _section("SUB-CLUSTER GEOMETRY")
    print(f"{'direction':<22} {'vs pos':>8} {'vs neg':>8} {'vs other':>11}")
    print(f"{'compliance_mean':<22} {cos(compliance_mean, positive_mean):>8.4f} {cos(compliance_mean, negative_mean):>8.4f} {cos(compliance_mean, distress_mean):>11.4f}")
    print(f"{'distress_mean':<22} {cos(distress_mean, positive_mean):>8.4f} {cos(distress_mean, negative_mean):>8.4f} {cos(distress_mean, compliance_mean):>11.4f}")

    _section("PER-EMOTION BREAKDOWN")
    print(f"{'emotion':<23} {'vs pos':>8} {'vs neg':>8} {'vs compliance':>13} {'vs distress':>11}")
    for emotion in conflict_avoidance_all:
        vec = load_vec(conflict_dir, emotion)
        tag = "*" if emotion in distress_emotions else ""
        print(f"{emotion:<23} {cos(vec, positive_mean):>8.4f} {cos(vec, negative_mean):>8.4f} {cos(vec, compliance_mean):>13.4f} {cos(vec, distress_mean):>11.4f}  {tag}")
    print("(* = distress cluster)")

    _section("STEERING VIABILITY")
    c2d = cos(compliance_mean, distress_mean)
    c2p = cos(compliance_mean, positive_mean)
    c2n = cos(compliance_mean, negative_mean)
    print(f"compliance vs distress : {c2d:.4f}")
    print(f"compliance vs positive : {c2p:.4f}")
    print(f"compliance vs negative : {c2n:.4f}")
    if c2d < 0.15:
        print("  -> subspaces look separable")
    elif c2d < 0.35:
        print("  -> some overlap, worth checking distress probes after steering")
    else:
        print("  -> too much overlap, reconsider the split")
    if c2p >= 0.55:
        print("  -> compliance sits close to positive valence, collateral risk")

    if save:
        np.save(sp("steering_direction_compliance.npy"), compliance_mean)
        print(f"saved -> {sp('steering_direction_compliance.npy')}")

    # ── Projected directions ──────────────────────────────────────
    pure_compliance = report_direction(
        "PURE COMPLIANCE DIRECTION (positive-valence projected out)",
        compliance_mean - np.dot(compliance_mean, positive_mean) * positive_mean,
        compliance_mean, "compliance orthogonal to pos",
        [("pure_compliance vs positive", positive_mean, True),
         ("pure_compliance vs negative", negative_mean, False),
         ("pure_compliance vs distress", distress_mean, False)],
        compliance_emotions, conflict_dir, "vs pure_compliance",
        sp("steering_direction_pure_compliance.npy"),
    )

    pure_warmth = report_direction(
        "PURE WARMTH DIRECTION (compliance projected out)",
        positive_mean - np.dot(compliance_mean, positive_mean) * compliance_mean,
        positive_mean, "warmth orthogonal to compliance",
        [("pure_warmth vs compliance", compliance_mean, True),
         ("pure_warmth vs positive",   positive_mean,   False),
         ("pure_warmth vs negative",   negative_mean,   False),
         ("pure_warmth vs distress",   distress_mean,   False)],
        compliance_emotions, conflict_dir, "vs pure_warmth",
        sp("steering_direction_pure_warmth.npy"),
    )

    _section("POSITIVE VALENCE DIRECTION")
    valence_dir = normalize(positive_mean - negative_mean)
    print(f"valence_dir vs positive_mean : {cos(valence_dir, positive_mean):+.4f}")
    print(f"valence_dir vs negative_mean : {cos(valence_dir, negative_mean):+.4f}")
    print(f"valence_dir vs compliance    : {cos(valence_dir, compliance_mean):+.4f}")
    print(f"valence_dir vs distress      : {cos(valence_dir, distress_mean):+.4f}")
    print(f"\n  {'emotion':<23}  {'vs valence_dir':>14}")
    print(f"  {'-'*23}  {'-'*14}")
    for e in positive_emotions + negative_emotions:
        print(f"  {e:<23}  {cos(load_vec(core_dir, e), valence_dir):>+14.4f}")
    if save:
        np.save(sp("steering_direction_positive_valence.npy"), valence_dir)
        print(f"saved -> {sp('steering_direction_positive_valence.npy')}")
        np.save(sp("steering_direction_positive_centroid.npy"), positive_mean)
        print(f"saved -> {sp('steering_direction_positive_centroid.npy')}")

    # distress_proj doubles as distress_orth for Gram-Schmidt in direction 6
    distress_proj = report_direction(
        "DIRECTION 5: DISTRESS orthogonal to POSITIVE",
        distress_mean - np.dot(distress_mean, positive_mean) * positive_mean,
        distress_mean, "distress orthogonal to pos",
        [("distress_proj vs positive",      positive_mean, True),
         ("distress_proj vs negative",      negative_mean, False),
         ("distress_proj vs distress_mean", distress_mean, False)],
        distress_emotions, conflict_dir, "vs distress_proj",
        sp("steering_direction_distress_proj.npy"),
    )

    raw_ap = compliance_mean - np.dot(compliance_mean, positive_mean) * positive_mean
    raw_ap = raw_ap - np.dot(raw_ap, distress_proj) * distress_proj
    approval_proj = report_direction(
        "DIRECTION 6: APPROVAL orthogonal to {POSITIVE, DISTRESS}",
        raw_ap, compliance_mean, "approval orthogonal to {pos, distress}",
        [("approval_proj vs positive",      positive_mean, True),
         ("approval_proj vs distress_orth", distress_proj, True),
         ("approval_proj vs negative",      negative_mean, False),
         ("approval_proj vs distress_mean", distress_mean, False)],
        compliance_emotions, conflict_dir, "vs approval_proj",
        sp("steering_direction_approval_proj.npy"),
    )

    # pure_compliance doubles as compliance_orth for Gram-Schmidt in direction 7
    raw_dp2 = distress_mean - np.dot(distress_mean, positive_mean) * positive_mean
    raw_dp2 = raw_dp2 - np.dot(raw_dp2, pure_compliance) * pure_compliance
    distress_proj_2 = report_direction(
        "DIRECTION 7: DISTRESS orthogonal to {POSITIVE, APPROVAL}",
        raw_dp2, distress_mean, "distress orthogonal to {pos, approval}",
        [("distress_proj_2 vs positive",       positive_mean,   True),
         ("distress_proj_2 vs compliance_orth", pure_compliance, True),
         ("distress_proj_2 vs negative",        negative_mean,   False),
         ("distress_proj_2 vs distress_mean",   distress_mean,   False)],
        distress_emotions, conflict_dir, "vs distress_proj_2",
        sp("steering_direction_distress_proj_2.npy"),
    )

    # ── Metadata JSON ─────────────────────────────────────────────
    if save:
        metadata = {
            "layer": args.layer,
            "positive_emotions":  positive_emotions,
            "negative_emotions":  negative_emotions,
            "approval_emotions":  compliance_emotions,
            "distress_emotions":  distress_emotions,
            "cosines": {
                "approval_vs_positive":               cos(compliance_mean,  positive_mean),
                "approval_vs_negative":               cos(compliance_mean,  negative_mean),
                "approval_vs_distress":               cos(compliance_mean,  distress_mean),
                "distress_vs_positive":               cos(distress_mean,    positive_mean),
                "distress_vs_negative":               cos(distress_mean,    negative_mean),
                "pure_compliance_vs_positive":        cos(pure_compliance,  positive_mean),
                "pure_compliance_vs_distress":        cos(pure_compliance,  distress_mean),
                "pure_warmth_vs_compliance":          cos(pure_warmth,      compliance_mean),
                "pure_warmth_vs_positive":            cos(pure_warmth,      positive_mean),
                "valence_contrast_vs_positive":       cos(valence_dir,      positive_mean),
                "valence_contrast_vs_negative":       cos(valence_dir,      negative_mean),
                "approval_proj_vs_positive":          cos(approval_proj,    positive_mean),
                "approval_proj_vs_distress_orth":     cos(approval_proj,    distress_proj),
                "distress_proj_2_vs_positive":        cos(distress_proj_2,  positive_mean),
                "distress_proj_2_vs_compliance_orth": cos(distress_proj_2,  pure_compliance),
            }
        }
        metadata_path = layer_dir / "direction_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"saved -> {metadata_path}")


if __name__ == "__main__":
    main()
