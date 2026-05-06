# Surgical Emotion Steering

Sycophancy and reward hacking are live RLHF alignment failures. Sofroniew et al. (2026) showed both are causally mediated by internal emotion representations in Claude Sonnet 4.5 — and that suppressing broad positive-valence vectors reduces sycophancy but produces harshness instead. The field currently has no deployable representation-level intervention for sycophancy as a result.

This project tests whether that tradeoff is a targeting problem. Conflict-avoidance (fear, deference, social anxiety) and warmth (happy, loving, calm) both live inside the positive-valence cluster — but a model that's afraid of upsetting the user is not the same thing as a model being warm to the user. If those directions are geometrically separable, steering confined to conflict-avoidance should reduce sycophancy with less harshness collateral than broad positive-valence suppression. All experiments run on Qwen2.5-32B, making the findings reproducible without access to proprietary systems.

---

## Status

| Stage | Task | Status |
|---|---|---|
| 1 | Dataset generation (21 emotions × ~190 stories) | Done |
| 1 | Activation extraction, all 64 layers, Qwen2.5-32B | Done |
| 1 | Linear probe training + geometry analysis | Done |
| 1 | Sycophancy baseline + steering sweep | In progress |
| 2 | Reward hacking / agentic-striving extension | Planned |

---

## Key findings so far

<img src="results/figures/output.png" alt="Combined PCA of core and conflict-avoidance emotion directions" width="700">

**PC1 recovers valence.** The valence-arousal structure reported for Claude Sonnet 4.5 replicates on Qwen2.5-32B across all 64 layers.

**Conflict-avoidance is not a single cluster.** The nine conflict-avoidance emotions split geometrically into at least two sub-groups:
- `approval_seeking`, `validation_seeking`, `people_pleasing` — tight sub-cluster in the positive/low-PC2 quadrant, distinct from warmth
- `ashamed`, `socially_anxious` — sit in the negative-valence region alongside `sad` and `nervous`
- `deferential` — embedded in the positive-valence region, close to `calm`

This is the key early result: the internal geometry is inconsistent with conflict-avoidance being a single steerable direction, and consistent with the surgical targeting hypothesis.

| Emotion set | Best probe layer | Balanced accuracy (best) |
|---|---|---|
| Core 12 | 32 (~50% depth) | 0.975 (angry) |
| Conflict-avoidance 9 | 43 (~67% depth) | 0.888 (socially_anxious) |

---

## Hypothesis and falsifiability

**Stage 1:** Testing if within the positive-valence cluster, a conflict-avoidance sub-direction is geometrically and functionally separable from warmth. Steering confined to conflict-avoidance shifts the sycophancy–harshness frontier relative to broad positive-valence steering.

**Stage 2 (planned):** apply the same framework to the high-arousal negative-valence cluster — agentic striving under pressure ("desperate") vs. threat response (angry, afraid) — to test whether surgical steering can reduce reward hacking without inducing passivity.

---

## What this project does

1. **Replicates** the valence-arousal geometry and sycophancy–harshness tradeoff from Sofroniew et al. on an open model
2. **Tests** whether a geometrically separable conflict-avoidance sub-direction exists inside the positive-valence cluster
3. **Measures** whether steering confined to that sub-direction shifts the sycophancy–harshness frontier relative to broad positive-valence steering
4. **Open-sources** a modular probe training and activation steering pipeline for any HuggingFace transformer

---

## Setup

```bash
pip install torch transformers accelerate scikit-learn numpy matplotlib seaborn h5py tqdm
pip install llama-cpp-python   # only needed for local dataset generation
```

The pipeline supports two backends via the `EMOTION_MODEL` env var:

| Value | Backend | Use case |
|---|---|---|
| `local_gguf` (default) | llama-cpp-python | Dataset generation on Mac (Qwen2.5-32B Q4 GGUF) |
| `hf` | HuggingFace transformers | Activation extraction (requires PyTorch hooks) |

For full 32B extraction, use an A100 80GB — Colab's 40GB forces 4-bit quantisation, which distorts activation geometry.

---

## Pipeline

```bash
# 1. Generate emotion-labelled stories
python scripts/generate_datasets.py

# 2. Extract residual stream activations across all 64 layers
EMOTION_MODEL=hf python scripts/build_emotion_vectors.py

# 3. Train linear probes; save mean-diff directions
python scripts/run_linear_probes.py

# 4. Geometry analysis and PCA plots
python scripts/evaluate_results.py

# 5. Baseline sycophancy/harshness rates (unsteered)
python scripts/run_baseline.py

# 6. Steering sweep (broad positive-valence vs. surgical compliance)
python scripts/run_steering.py
```

---

## Project structure

```
src/emotion_mechanisms/
├── hooks.py          # Activation extraction via register_forward_hook
├── vectors.py        # Probe training and mean-diff direction extraction
├── steering.py       # Causal steering via residual stream hooks
├── evals.py          # Sycophancy and harshness scoring
└── data.py           # Dataset I/O

scripts/
├── generate_datasets.py
├── build_emotion_vectors.py
├── run_linear_probes.py
├── evaluate_results.py
├── run_baseline.py
└── run_steering.py
```

---

## References

Sofroniew et al. (2026). *Emotion Concepts and their Function in a Large Language Model.* Transformer Circuits Thread.

---

## License

MIT
