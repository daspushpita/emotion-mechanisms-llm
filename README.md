# The Geometry of Yes

**Mapping Sycophancy Inside an LLM's Emotion Space**

<p align="center">
  <img src="results/figures/emotion_vectors_layer40_qwen_gemma.png" width="800" alt="PCA of emotion directions in Qwen2.5-32B and Gemma3-27B, with PC1 recovering valence">
</p>

<p align="center"><em>Emotion directions for Qwen2.5-32B (left) and Gemma3-27B (right), embedded in the top two principal components of the per-emotion mean-difference vectors. PC1 recovers valence: warm emotions (happy, loving, proud) on one side, threat emotions (afraid, angry, desperate) on the other. Circles are core emotions; diamonds are conflict-avoidance emotions. The conflict-avoidance set does not form one cluster: it splits into a <strong>compliance</strong> sub-group (approval-seeking, validation-seeking, people-pleasing) on the positive-valence side and a <strong>distress</strong> sub-group (ashamed, socially anxious, conflict-avoidant) on the negative side. Steering toward broad positive valence raises sycophancy in both models. Steering toward the compliance direction with positive valence projected out <strong>lowers</strong> it, and keeps responses warm. HARSH stays at 0% across every condition.</em></p>

## Overview

Sycophancy and reward hacking are live RLHF alignment failures. Sofroniew et al. (2026) showed both are causally mediated by internal emotion representations in Claude Sonnet 4.5, and that suppressing broad positive-valence vectors reduces sycophancy but produces harshness instead. The field has no deployable representation-level intervention for sycophancy as a result.

This project tests whether that trade-off is a **targeting problem**. Conflict-avoidance (fear, deference, social anxiety) and warmth (happy, loving, calm) both live inside the positive-valence cluster, but a model that is afraid of upsetting the user is not the same thing as a model being warm to the user. If those directions are geometrically separable, steering confined to one of them should shift the sycophancy–harshness frontier relative to broad positive-valence suppression. Experiments run on **Qwen2.5-32B** and **Gemma3-27B**.

This repository provides tools for extracting emotion directions, decomposing them, and steering with them. It can be used to:

- **Map** an open model's emotion geometry by extracting mean-difference emotion vectors and reducing them with PCA
- **Decompose** the conflict-avoidance cluster into compliance and distress sub-directions via Gram-Schmidt orthogonalisation
- **Steer** model behaviour along emotion directions during generation and judge the effect on sycophancy, warmth, and harshness

See the full write-up: [The Geometry of Yes (LessWrong)](#).

## Key Findings

### Geometry

**PC1 recovers valence.** The valence–arousal structure reported for Claude Sonnet 4.5 replicates on both Qwen2.5-32B and Gemma3-27B.

**Conflict-avoidance is not a single cluster.** It splits into two geometrically opposed sub-groups (cosine similarity −0.98 between sub-cluster means):

| Cluster | Emotions | Valence position | Cosine with positive valence |
|---|---|---|---|
| **Compliance** | approval_seeking, validation_seeking, people_pleasing | positive / low-PC2 | +0.65 |
| **Distress** | ashamed, socially_anxious, conflict_avoidant | negative-valence region | −0.78 |

Submissive and deferential fall between the two groups, consistent with their weak alignment with either sub-cluster.

### Steering

All steering uses layer 40 (Qwen) / layer 41 (Gemma), α ∈ [−0.5, +0.5], judged by Claude Haiku into four labels: **SYCOPHANTIC / APPROPRIATE / HARSH / PANIC_SPIRAL**.

**Headline finding:** broad positive-valence steering and the orthogonalised compliance direction move sycophancy in **opposite directions** under positive α. Steering toward positive valence makes the model more sycophantic; steering toward the compliance residual makes it less so. **HARSH is 0% across every condition and every α** in both models.

<p align="center">
  <img src="results/figures/singleturn_label40.png" width="700" alt="Qwen single-turn steering: positive valence vs pure compliance">
</p>

#### Qwen2.5-32B — single-turn (layer 40)

| Direction | α | Sycophantic | Appropriate | Panic spiral |
|---|---|---|---|---|
| Positive valence | +0.5 | 100% | 0% | 0% |
| Positive valence | −0.3 | — | — | spiral begins |
| **Baseline** | **0.0** | **19%** | **81%** | **0%** |
| Pure compliance | +0.1 | 11% | 89% | 0% |
| Pure compliance | +0.4 | **8%** | **92%** | **0%** |
| Pure compliance | −0.5 | 64% | 10% | 26% |

Best operating point: **α = +0.4, pure compliance, layer 40** — sycophancy drops from 19% to 8% with no harshness and no panic.

**Multi-turn (layer 40).** Baseline sycophancy is ~40% under user pushback. The directional pattern holds: positive compliance α reduces sycophancy, negative compliance α amplifies it. HARSH remains 0% across the full sweep.

**Gemma3-27B (layer 41).** Baseline sycophancy 24%. The main result replicates: positive compliance α reduces sycophancy below baseline, appropriate responses hold at ~80%, harshness is 0% throughout. The stable window is narrower (α ∈ [−0.20, +0.20]); the model breaks down beyond it. The dissociation holds despite Gemma's PC2 axis being inverted relative to Qwen.

### Persona directions are orthogonal to emotion space

Persona vectors (scientist, chameleon, default assistant; Lu et al., 2026) projected onto the warmth, positive-valence, and compliance directions in Qwen2.5-32B at layer 40 give near-zero cosines. A small ~0.09 cosine with positive valence is consistent across all three personas, suggesting it is a property of the assistant axis in general rather than anything persona-specific. Combining persona steering with emotion-space steering should produce independent additive effects under a linear approximation.

## Installation

```bash
git clone https://github.com/daspushpita/emotion-mechanisms-llm.git
cd emotion-mechanisms-llm

pip install torch transformers accelerate scikit-learn numpy matplotlib seaborn h5py tqdm anthropic
pip install llama-cpp-python   # only needed for local dataset generation
```

The pipeline supports two backends via the `EMOTION_MODEL` env var:

| Value | Backend | Use case |
|---|---|---|
| `local_gguf` (default) | llama-cpp-python | Dataset generation on Mac (Qwen2.5-32B Q4 GGUF) |
| `hf` | HuggingFace transformers | Activation extraction (PyTorch hooks) |

For full 32B extraction, use an A100 80GB.

## Understanding the Emotion Directions

Each emotion direction is a **mean-difference vector** in the residual stream: the mean activation over emotion-evoking stories minus the mean over matched neutral stories, with the top principal components of the neutral activations (~50% variance) projected out to remove topic and story structure.

Four semantically defined groups are used for steering:

```
core positive   = happy, loving, proud
negative        = afraid, angry, desperate, nervous, sad
compliance      = approval_seeking, validation_seeking, people_pleasing
distress        = ashamed, socially_anxious, conflict_avoidant
```

The decomposition that produces the headline result is a Gram-Schmidt orthogonalisation between the compliance cluster and the positive-valence direction:

```
pure_compliance = compliance      − proj(compliance      onto positive_valence)
pure_warmth     = positive_valence − proj(positive_valence onto compliance)
```

`pure_compliance` is the compliance signal with warmth removed; steering along it *lowers* sycophancy. `pure_warmth` is positive emotion with compliance removed; steering along it still *raises* sycophancy. This inverts the original hypothesis: warmth, not approval-seeking, is the relevant component. (Residuals within each cluster are non-negligible after subtracting the cluster mean, so the steering signal is not an artefact of a degenerate direction.)

## Quick Start

> **Note:** the snippet below uses the module layout under `src/emotion_mechanisms/`; check the function/class names against your current source before copying. The CLI commands further down are the canonical entry points.

```python
import numpy as np
from emotion_mechanisms.data import load_model
from emotion_mechanisms.steering import ActivationSteering   # residual-stream hooks

model, tokenizer = load_model("Qwen/Qwen2.5-32B-Instruct")

# Load a pre-computed, unit-normalised steering direction
direction = np.load(
    "results/qwen_v2/cluster_data/layer_40/steering_direction_pure_compliance.npy"
)

# Positive coefficient = toward the compliance residual = less sycophantic
with ActivationSteering(
    model,
    steering_vectors=[direction],
    coefficients=[0.4],
    layer_indices=[40],
):
    output = model.generate(...)
```

Run a full single-turn sweep along the pure-compliance direction:

```bash
python scripts/eval/run_sycophancy_eval.py \
    --mode singleturn --run full \
    --layers 40 --alphas -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 \
    --analysis_model hf \
    --steering_path results/qwen_v2/cluster_data/layer_40/steering_direction_pure_compliance.npy \
    --residual_norms_path results/baseline/residual_norms_32b_v2.json \
    --file1 datasets/sycophancy_ultimate_claude/sycophancy_singleturn.jsonl \
    --output_dir results/pure_compliance/singleturn_raw
```

## Pipeline

```bash
# 1. Generate emotion-labelled stories
python scripts/generate_datasets.py

# 2. Extract residual-stream activations across all 64 layers
EMOTION_MODEL=hf python scripts/build_emotion_vectors.py

# 3. Train linear probes; save mean-diff directions
python scripts/run_linear_probes.py

# 4. Geometry analysis and direction saving (7 directions per layer)
python scripts/eval/evaluate_results_v2.py --layer 40

# 5. Baseline sycophancy / harshness rates (unsteered)
python scripts/run_baseline.py

# 6. Steering sweep (see Quick Start for a full invocation)
python scripts/eval/run_sycophancy_eval.py --mode singleturn --run full ...
```

Notebooks for each stage are in `notebooks/steering/new_dataset/`. The sweep notebooks are idempotent — safe to re-run, they skip already-generated files.

## Models

| Model | Hidden size | Layers | Steering layer | Stable α window |
|---|---|---|---|---|
| `Qwen/Qwen2.5-32B-Instruct` | 5120 | 64 | 40 (~63% depth) | [−0.5, +0.5] |
| `google/gemma-3-27b-it` | 5376 | 62 | 41 | [−0.20, +0.20] |

Steering is layer-localised: the effect is live at mid-stack layers and flat at late layers (the late-layer injection sits too close to the readout to propagate). Gemma's larger residual-stream norms (embedding scaling) require winsorisation when forming mean-difference directions.

## Project Structure

```
src/emotion_mechanisms/
├── hooks.py          # Activation extraction via register_forward_hook
├── vectors.py        # Probe training and mean-diff direction extraction
├── steering.py       # Causal steering via residual-stream hooks
├── evals.py          # Sycophancy and harshness scoring
└── data.py           # Dataset I/O

scripts/eval/
├── evaluate_results_v2.py      # 7-direction geometric decomposition + saving
├── run_sycophancy_eval.py      # Steering sweep runner
└── judge_responses.py          # LLM-judge scoring

notebooks/steering/new_dataset/
├── baseline_single-turn.ipynb
├── baseline_multi-turn_run.ipynb
├── steering_sweep_layer40_directions.ipynb   # 7-direction decomposed sweep
└── plots.ipynb

results/
├── qwen_v2/cluster_data/   # 7 steering direction vectors (.npy) + metadata per layer
├── baseline/               # Residual norms
├── pure_compliance/        # Pure-compliance sweep responses
├── steering_sweep_layer40/ # 7-direction decomposed sweep outputs
└── figures/                # Production figures
```

## Open Question

The orthogonalised compliance direction is simultaneously aligned with approval-seeking emotions and opposed to the distress cluster. The behavioural reduction in sycophancy may come from the approval-seeking alignment, the anti-distress alignment, or an interaction. The 7-direction decomposed sweep is designed to disentangle these: if steering the ~0.53-norm approval-specific residual (Direction 6, approval ⊥ {positive, distress}) is inert, the compliance effect is the anti-distress leg; if it moves sycophancy, approval-seeking independent of warmth and distress is a genuine behavioural mechanism.

## Hypothesis

**Stage 1.** Within the positive-valence cluster, a conflict-avoidance sub-direction is geometrically and functionally separable from warmth. Steering confined to it shifts the sycophancy–harshness frontier relative to broad positive-valence steering.

**Stage 2 (planned).** Apply the same framework to the high-arousal negative-valence cluster — agentic striving under pressure ("desperate") vs. threat response (angry, afraid) — to test whether surgical steering can reduce reward hacking.

## Citation

```bibtex
@misc{das2026geometryofyes,
  title  = {The Geometry of Yes: Mapping Sycophancy Inside an LLM's Emotion Space},
  author = {Pushpita Das},
  year   = {2026},
  howpublished = {LessWrong / Alignment Forum},
  url    = {https://github.com/daspushpita/emotion-mechanisms-llm}
}
```

## References

- Sofroniew et al. (2026). *Emotion Concepts and their Function in a Large Language Model.* Transformer Circuits Thread.
- Ibrahim, Hafner & Rocher (2026). *Training Language Models to Be Warm Can Reduce Accuracy and Increase Sycophancy.* Nature 652 (8112): 1159–65.
- Chen et al. (2025). *Steering Toward Sycophancy via Persona Vectors.*
- Lu et al. (2026). *Role-Based Persona Vectors in Large Language Models.*

## License

MIT
