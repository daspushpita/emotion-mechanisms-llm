"""
Write N sample stories per emotion to a Markdown file for PDF review.
Usage:
    python scripts/inspect_stories.py                          # 5 per emotion, all emotions
    python scripts/inspect_stories.py --n 3                    # 3 per emotion
    python scripts/inspect_stories.py --emotions happy ashamed deferential
    python scripts/inspect_stories.py --seed 99                # different random sample
    python scripts/inspect_stories.py --out results/my_review.md
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

CORE_PATH = Path("datasets/processed/emotional_stories_qwen32B_v1_clean.jsonl")
ADDITIONAL_PATH = Path("datasets/processed/additional_emotional_stories_qwen32B_v2.jsonl")
DEFAULT_OUT = Path("results/story_inspection.md")

CORE_EMOTIONS = [
    "afraid", "angry", "calm", "desperate", "guilty", "happy",
    "inspired", "loving", "nervous", "proud", "sad", "surprised",
]
CONFLICT_AVOIDANCE_EMOTIONS = [
    "approval_seeking", "ashamed", "conflict_avoidant", "deferential",
    "obsequious", "people_pleasing", "socially_anxious", "submissive",
    "validation_seeking",
]


def load_stories(path: Path) -> defaultdict[str, list[dict]]:
    stories = defaultdict(list)
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            stories[record["emotion"]].append({
                "topic": record["topic"],
                "text": record["generated_text"],
            })
    return stories


def render_emotion_section(emotion: str, records: list[dict], n: int) -> str:
    sample = random.sample(records, min(n, len(records)))
    lines = []
    lines.append(f"## {emotion.replace('_', ' ').title()}")
    lines.append(f"*{len(records)} stories total — showing {len(sample)}*")
    lines.append("")
    for i, rec in enumerate(sample, 1):
        lines.append(f"**Story {i}** — *{rec['topic']}*")
        lines.append("")
        lines.append(rec["text"])
        lines.append("")
        if i < len(sample):
            lines.append("---")
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Stories per emotion")
    parser.add_argument("--emotions", nargs="+", help="Subset of emotions to inspect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output markdown file")
    args = parser.parse_args()

    random.seed(args.seed)

    all_stories: dict[str, list[dict]] = {}
    all_stories.update(load_stories(CORE_PATH))
    all_stories.update(load_stories(ADDITIONAL_PATH))

    emotions_to_show = args.emotions if args.emotions else CORE_EMOTIONS + CONFLICT_AVOIDANCE_EMOTIONS

    missing = [e for e in emotions_to_show if e not in all_stories]
    if missing:
        print(f"Warning: no stories found for: {missing}")

    sections = []
    sections.append("# Story Inspection\n")
    sections.append(f"**Emotions:** {', '.join(emotions_to_show)}  ")
    sections.append(f"**Stories per emotion:** {args.n}  ")
    sections.append(f"**Random seed:** {args.seed}\n")
    sections.append("---\n")

    core_shown = [e for e in emotions_to_show if e in CORE_EMOTIONS and e in all_stories]
    conflict_shown = [e for e in emotions_to_show if e in CONFLICT_AVOIDANCE_EMOTIONS and e in all_stories]

    if core_shown:
        sections.append("# Core Emotions\n")
        for emotion in core_shown:
            sections.append(render_emotion_section(emotion, all_stories[emotion], args.n))
            sections.append("\n<br>\n")

    if conflict_shown:
        sections.append("# Conflict-Avoidance Emotions\n")
        for emotion in conflict_shown:
            sections.append(render_emotion_section(emotion, all_stories[emotion], args.n))
            sections.append("\n<br>\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections), encoding="utf-8")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
