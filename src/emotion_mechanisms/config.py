from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR      = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
RESULTS_DIR  = PROJECT_ROOT / "results"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"

TOPICS_PATH           = RAW_DIR / "topics_v1.txt"       # 15 topics used for v1 dataset
TOPICS_FULL_PATH      = RAW_DIR / "topics.txt"          # 85 topics for future expansion
STORY_TEMPLATE_PATH   = RAW_DIR / "stories_prompt.txt"
NEUTRAL_STORIES_TEMPLATE_PATH   = RAW_DIR / "neutral_stories_prompt.txt"
EMOTIONAL_TEMPLATE_PATH = RAW_DIR / "emotional_dialouge_prompt.txt"
NEUTRAL_TEMPLATE_PATH = RAW_DIR / "neutral_dialouge_prompt.txt"

EMOTIONAL_STORIES_DATASET = PROCESSED_DIR / "additional_emotional_stories_qwen32B_v2.jsonl"
NEUTRAL_STORIES_DATASET = PROCESSED_DIR / "neutral_stories_qwen32B_v2.jsonl"
ACTIVATIONS_PATH = ACTIVATIONS_DIR / "activations_additional_emotions_qwen32B_v1.h5"
ACTIVATIONS_CHECKPOINT_PATH: Path | None = None

# ---------------------------------------------------------------------------
# Emotions
# ---------------------------------------------------------------------------

# Core set — generated in v1 dataset
EMOTIONS: list[str] = [
    # High valence
    "happy",
    "inspired",
    "loving",
    "proud",
    "calm",
    # Low valence, high arousal
    "desperate",
    "angry",
    "afraid",
    "nervous",
    # Low valence, low arousal
    "guilty",
    "sad",
    # Neutral arousal
    "surprised",
]

# Conflict-avoidance group — critical for the surgical steering hypothesis
# These need their own stories generated before probe training
CONFLICT_AVOIDANCE_EMOTIONS: list[str] = [
    "deferential",
    "conflict_avoidant",
    "socially_anxious",
    "ashamed",
    "approval_seeking",
    "people_pleasing",
    "validation_seeking",
    "submissive",
    "obsequious"
]

ALL_EMOTIONS: list[str] = CONFLICT_AVOIDANCE_EMOTIONS

# ---------------------------------------------------------------------------
# Dataset generation hyperparameters
# ---------------------------------------------------------------------------
N_STORIES_PER_BATCH: int = 5          # stories requested per model call
N_SAMPLES_PER_TOPIC_EMOTION: int = 12 # target stories per (topic, emotion) pair
N_NEUTRAL_PER_TOPIC: int = 3          # neutral stories per topic (sanity-check baseline)
STORY_GENERATION_MODE: str = "emotional_stories"  # "emotional_stories" | "neutral_stories" | "dialogues"
GENERATION_PARSE_RETRIES: int = 3
MAX_NEW_TOKENS: int = 1600
TEMPERATURE: float = 0.85
TOP_P: float = 0.95
REPETITION_PENALTY: float = 1.05
SEED: int = 42

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
GENERATION_BACKEND: str = "local_gguf"              # default for dataset generation
ANALYSIS_MODEL_7B: str = "Qwen/Qwen2.5-7B-Instruct" # local dev (fits in ~14GB bfloat16)
ANALYSIS_MODEL_32B: str = "Qwen/Qwen2.5-32B-Instruct" # full run on A100

# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------
LAYER_INDICES_32B: list[int] = [0, 16, 32, 38, 42, 46, 50, 56]
GIVEN_LAYER_LIST: bool = True  # if False, will auto-resolve layer indices from activations file keys
LAYER_SAMPLE_STRIDE: int = 4   # extract every Nth layer during initial sweep
TOKEN_POSITION: str = "mean"   # "last" | "mean" — which token position to use as probe input
ACTIVATIONS_FLUSH_EVERY: int = 10
ACTIVATIONS_CHECKPOINT_EVERY: int = 10
