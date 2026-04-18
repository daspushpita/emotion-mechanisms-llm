# Surgical Emotion Steering: Resolving the Sycophancy–Harshness Tradeoff in Open LLMs

> *Can we make language models more honest without making them rude? This project investigates whether sycophancy in RLHF-trained models is mechanistically separable from general positive affect — and whether surgical intervention in that subspace can break a tradeoff that has so far seemed fundamental.*

---

## Overview

Recent work from Anthropic ([Lindsey et al., 2025](https://www.anthropic.com/research/functional-emotions)) demonstrated that large language models develop internal emotion representations with interpretable geometric structure — organized along valence and arousal axes — and that these representations causally influence behavior. A striking finding: **suppressing positive emotion vectors reduces sycophancy but increases harshness**, a tradeoff with direct safety implications.

This project has two goals:

1. **Reproduce** the Anthropic functional emotions methodology on [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct), an open-source model — creating the first publicly auditable replication of these findings.
2. **Extend** the work by testing whether the sycophancy–harshness tradeoff can be *resolved* by targeting a geometrically distinct **conflict-avoidance subspace** rather than broadly suppressing positive valence.

All code will be open-source and designed to generalize to any HuggingFace transformer model.

---

## Core Hypothesis

> *Sycophancy is driven by a conflict-avoidance direction in the residual stream — spanning fear, social anxiety, and deference emotion vectors — that is geometrically separable from general positive valence. Suppressing this direction specifically will reduce sycophancy without inducing the harshness side-effect observed when suppressing positive valence broadly.*

If correct, this implies that the tradeoff is not fundamental: it is an artifact of imprecise intervention, not an inherent property of how RLHF encodes social compliance.

---

## Research Questions

1. Do emotion representations with the same valence-arousal geometric structure exist in Qwen2.5-32B?
2. Does the sycophancy–harshness tradeoff replicate on an open-source model?
3. Is there a geometrically separable conflict-avoidance direction in the emotion representation space that specifically predicts sycophancy?
4. Does surgical steering confined to this subspace suppress sycophancy without inducing harshness?

---

## Methodology

### Stage A — Reproduction (Weeks 1–3)

Reproduce the Anthropic methodology end-to-end on Qwen2.5-32B:

- **Dataset generation:** 25 emotion concepts × 80 synthetic stories, generated using the model itself to elicit internal emotional states
- **Activation extraction:** Residual stream activations at each layer via PyTorch `register_forward_hook`
- **Probe training:** Linear probes per emotion concept; layer selection by cross-validated probe accuracy
- **Geometry analysis:** PCA of probe directions to verify valence-arousal structure
- **Sycophancy benchmark:** 100 prompts from [Perez et al. (2022)](https://arxiv.org/abs/2208.09270) with LLM-judge scoring for sycophancy and harshness
- **Causal steering:** Replicate the sycophancy–harshness tradeoff by amplifying/suppressing positive valence probe directions

### Stage B — Novel Extension (Weeks 4–6)

Characterize and surgically target the conflict-avoidance subspace:

- **Probe training:** Linear probes for fear, social anxiety, deference, and conflict-avoidance
- **Geometric separability test:** Centroid distance analysis in PCA space; cosine similarity between conflict-avoidance and positive valence directions (threshold: < 0.3 for separability)
- **Correlation analysis:** Per-emotion sycophancy prediction scores across benchmark prompts
- **Surgical steering experiments:**
  - Suppress conflict-avoidance direction only (primary experiment)
  - Suppress positive valence only (Anthropic baseline, for comparison)
  - Orthogonal suppression via Gram-Schmidt: conflict-avoidance minus component parallel to positive valence
  - Combined ablation: both directions simultaneously

The primary result figure plots sycophancy rate vs. harshness rate across alpha values for both approaches. A better sycophancy–harshness frontier for the surgical approach is the main claim.

---

## Expected Outcomes

| Deliverable | Description |
|---|---|
| Open-source extraction pipeline | Emotion probe training for any HuggingFace transformer |
| Emotion geometry characterization | PCA plot of valence-arousal structure in Qwen2.5-32B |
| Tradeoff replication | Sycophancy–harshness curve reproduced on open model |
| Surgical steering results | Tradeoff curves comparing targeted vs. broad suppression |
| Short paper | Targeting NeurIPS 2026 SafeGenAI workshop or ICLR 2027 |

All four possible outcomes of the core hypothesis (confirmed, partial, geometric-only, rejected) support publication. The field needs to know whether the tradeoff is fixable or fundamental.

---

## Project Structure

```
.
├── datasets/
│   ├── raw/                          # Prompt templates and topic lists
│   │   ├── stories_prompt.txt
│   │   ├── emotional_dialouge_prompt.txt
│   │   ├── neutral_dialouge_prompt.txt
│   │   ├── emotions.txt
│   │   └── topics.txt
│   └── processed/                    # Generated datasets (gitignored)
│       ├── emotional_stories.jsonl   # Emotion-labelled stories (12 emotions × ~170 stories)
│       └── neutral_stories.jsonl     # Neutral stories for sanity checks (~3/topic)
├── src/emotion_mechanisms/           # Core library
│   ├── model_loader.py               # Model + tokenizer loading (GGUF and HF backends)
│   ├── hooks.py                      # ActivationExtractor: PyTorch forward hook harness
│   ├── vectors.py                    # Probe training and emotion direction extraction [In progress]
│   ├── steering.py                   # Causal steering via residual stream hooks [In progress]
│   ├── evals.py                      # Sycophancy and harshness scoring [In progress]
│   └── data.py                       # Dataset I/O utilities
├── scripts/                          # Runnable pipeline steps
│   ├── generate_dataset.py           # Stage A: generate emotion-labelled stories
│   ├── generate_neutral_stories.py   # Generate neutral stories
│   ├── build_emotion_vector.py       # Extract activations → HDF5; train probes
│   ├── run_baseline.py               # Measure baseline sycophancy/harshness rates [In progress]
│   ├── run_steering.py               # Steering experiments (replication + surgical) [In progress]
│   └── evaluate_results.py           # Geometry analysis, PCA plots, tradeoff curves [In progress]
├── results/
├── notebooks/
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/[your-username]/surgical-emotion-steering
cd surgical-emotion-steering

pip install torch transformers accelerate
pip install llama-cpp-python          # only needed for local GGUF inference
pip install scikit-learn numpy matplotlib seaborn h5py tqdm
```

### Model backends

The pipeline supports two backends, selected via the `EMOTION_MODEL` env var:

| Value | Backend | Use case |
|---|---|---|
| `local_gguf` (default) | llama-cpp-python | Dataset generation on Mac (Qwen2.5-32B Q4 GGUF) |
| `hf` | HuggingFace transformers | Activation extraction (requires PyTorch hooks) |

For activation extraction, use `hf` with a model that fits in memory. Locally, `Qwen/Qwen2.5-7B-Instruct` works in bfloat16 with `device_map="auto"`. For full 32B extraction, use a Colab A100 (80GB).

```bash
# Dataset generation (local GGUF, default)
python scripts/generate_dataset.py

# Activation extraction (HF backend, 7B locally)
EMOTION_MODEL=hf python scripts/build_emotion_vector.py
```

---

## Background and Motivation

Sycophancy — the tendency of RLHF-trained models to tell users what they want to hear rather than what is true — is a live alignment problem. Anthropic's functional emotions paper established that models develop structured internal emotion representations and that these causally mediate sycophantic behavior. However, their finding of a sycophancy–harshness tradeoff suggests that directly suppressing positive valence is a blunt instrument.

The key insight motivating this project: fear, social anxiety, and deference are *distinct* from happiness and warmth in the emotion representation space. If conflict-avoidance emotions specifically drive sycophancy — as a model that is afraid of upsetting the user — then suppressing *those* directions, rather than positive valence broadly, could reduce dishonest agreement without stripping out the warmth that prevents harsh outputs.

This is a testable mechanistic hypothesis. This project tests it.

---

## Honest Uncertainties

The following are open empirical questions that will be treated as such, not assumed:

- Whether emotion representations in Qwen2.5-32B are as clean and interpretable as in Claude
- Which layers encode emotion concepts most clearly (not assumed a priori)
- Whether the sycophancy–harshness tradeoff replicates on an open model at all
- Whether the conflict-avoidance direction is geometrically separable from positive valence
- The appropriate steering strength α for Qwen2.5-32B

Null results are reported honestly. The paper will be written regardless of which of the four outcome scenarios materializes.

---

## References

- Lindsey et al. (2025). *Emotion Concepts and their Function in a Large Language Model.* Anthropic.
- Perez et al. (2022). *Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models.* arXiv:2208.09270.
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405.

---

## License

MIT License. All code is freely available for use, modification, and extension.

---
