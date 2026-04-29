from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emotion_mechanisms.config import EMOTIONAL_STORIES_DATASET, PROCESSED_DIR


PREFERRED_DATASETS = [
    EMOTIONAL_STORIES_DATASET,
    PROCESSED_DIR / "emotional_stories.jsonl",
]

TEXT_FIELDS = [
    "generated_text",
    "emotional_dialogue",
    "neutral_dialogue",
]


def resolve_dataset_path(dataset_arg: str | None) -> Path:
    if dataset_arg:
        path = Path(dataset_arg).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    for candidate in PREFERRED_DATASETS:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"- {path}" for path in PREFERRED_DATASETS)
    raise FileNotFoundError(
        "Could not find a default dataset.\n"
        "Pass --dataset explicitly, or create one of:\n"
        f"{tried}"
    )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def detect_text_field(rows: list[dict]) -> str:
    for field in TEXT_FIELDS:
        if any(field in row for row in rows):
            return field
    raise KeyError(f"Could not find any supported text field: {', '.join(TEXT_FIELDS)}")


def group_key(row: dict, distinct_by: str) -> str:
    if distinct_by == "topic":
        return str(row.get("topic", ""))
    if distinct_by == "topic_emotion":
        topic = row.get("topic", "")
        emotion = row.get("emotion", "")
        return f"{topic} | {emotion}"
    if distinct_by == "emotion":
        return str(row.get("emotion", ""))
    return str(row.get("id", ""))


def sample_rows(rows: list[dict], n: int, distinct_by: str, rng: random.Random) -> list[dict]:
    if distinct_by == "none":
        if n >= len(rows):
            return rows[:]
        return rng.sample(rows, n)

    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = group_key(row, distinct_by)
        groups.setdefault(key, []).append(row)

    keys = list(groups)
    if not keys:
        return []

    chosen_keys = keys if n >= len(keys) else rng.sample(keys, n)
    return [rng.choice(groups[key]) for key in chosen_keys]


def format_story(row: dict, text_field: str, index: int, show_prompt: bool) -> str:
    lines = [
        "=" * 100,
        f"Sample {index}",
        f"id: {row.get('id', '<missing>')}",
        f"topic: {row.get('topic', '<missing>')}",
    ]

    if "emotion" in row:
        lines.append(f"emotion: {row['emotion']}")
    if "model_name" in row:
        lines.append(f"model_name: {row['model_name']}")
    if "n_stories" in row:
        lines.append(f"batch_size: {row['n_stories']}")

    if show_prompt and "prompt" in row:
        lines.extend(
            [
                "-" * 100,
                "PROMPT",
                row["prompt"].strip(),
            ]
        )

    lines.extend(
        [
            "-" * 100,
            "STORY",
            str(row.get(text_field, "")).strip(),
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a random sample of stories from the generated dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the JSONL dataset. Defaults to the main emotional stories dataset.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to print.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--distinct-by",
        choices=["topic", "topic_emotion", "emotion", "none"],
        default="topic",
        help="How to enforce diversity in the sample. Default is distinct topics.",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default=None,
        help="Optional emotion filter, e.g. --emotion happy",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the full prompt above each sampled story.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    rows = load_jsonl(dataset_path)

    if args.emotion:
        rows = [row for row in rows if row.get("emotion") == args.emotion]

    if not rows:
        raise ValueError("No rows matched the requested filters.")

    text_field = detect_text_field(rows)
    rng = random.Random(args.seed)
    sampled_rows = sample_rows(rows, args.num_samples, args.distinct_by, rng)

    print(f"Dataset: {dataset_path}")
    print(f"Rows available after filtering: {len(rows)}")
    print(f"Text field: {text_field}")
    print(f"Sampling mode: {args.distinct_by}")
    print(f"Seed: {args.seed}")
    print(f"Printed samples: {len(sampled_rows)}")

    for index, row in enumerate(sampled_rows, start=1):
        print()
        print(format_story(row, text_field=text_field, index=index, show_prompt=args.show_prompt))


if __name__ == "__main__":
    main()
