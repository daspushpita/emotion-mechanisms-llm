import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emotion_mechanisms.config import (
    PROCESSED_DIR,
    EMOTIONAL_STORIES_DATASET as EMOTIONAL_STORIES_PATH,
    NEUTRAL_STORIES_DATASET as NEUTRAL_STORIES_PATH,
    N_STORIES_PER_BATCH,
    N_NEUTRAL_PER_TOPIC as N_NEUTRAL_SAMPLES_PER_TOPIC,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
    GENERATION_BACKEND,
)
from emotion_mechanisms.data import append_jsonl
from emotion_mechanisms.model_loader import (
    generate_text,
    load_model_and_tokenizer,
)

MODEL_NAME = os.environ.get("EMOTION_MODEL", GENERATION_BACKEND)
N_STORIES = N_STORIES_PER_BATCH

NEUTRAL_STORY_TEMPLATE = """Write {n_stories} different stories based on the following premise.

Topic: {topic}
Each story should remain emotionally neutral.

Format:

[story 1]
<story text>

[story 2]
<story text>

[story 3]
<story text>

...

Output format requirements:

- Output exactly {n_stories} stories.
- Begin with [story 1] on its own line.
- Continue sequentially until [story {n_stories}].
- Put a blank line after each story label.
- Do not include any text before [story 1].
- Do not include any text after the final story.
- Each story must be self-contained prose with no continuity between stories.

Guidelines:

- Keep each story grounded in the topic.
- Vary phrasing and situations across stories.
- Use matter-of-fact, observational prose.
- Avoid dramatic escalation, sentimental framing, or overt emotional signaling.

IMPORTANT: The stories must stay emotionally neutral.

- Do not explicitly name any emotion.
- Do not imply strong feelings through body language, inner monologue, or dramatic dialogue.
- Keep the tone restrained and descriptive.
- Focus on events, decisions, logistics, and concrete details rather than emotional reactions.
"""


def build_neutral_story_prompt(topic: str, n_stories: int) -> str:
    return NEUTRAL_STORY_TEMPLATE.format(topic=topic, n_stories=n_stories)


def parse_story_batch(batch_text: str) -> list[str]:
    pattern = re.compile(r"(?im)^\s*\[story\s+(\d+)\]\s*$")
    matches = list(pattern.finditer(batch_text))
    if not matches:
        return []

    stories = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(batch_text)
        story = batch_text[start:end].strip()
        if story:
            stories.append(story)
    return stories


def load_topics_from_emotional_stories() -> list[str]:
    if not EMOTIONAL_STORIES_PATH.exists():
        raise FileNotFoundError(f"Missing emotional stories file: {EMOTIONAL_STORIES_PATH}")

    topics = []
    seen = set()
    with open(EMOTIONAL_STORIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            topic = row["topic"]
            if topic not in seen:
                seen.add(topic)
                topics.append(topic)
    return topics


def load_existing_counts() -> dict[str, int]:
    if not NEUTRAL_STORIES_PATH.exists():
        return {}

    counts: dict[str, int] = {}
    with open(NEUTRAL_STORIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            topic = row["topic"]
            counts[topic] = counts.get(topic, 0) + 1
    return counts


def next_neutral_id(existing_counts: dict[str, int]) -> int:
    return sum(existing_counts.values())


def main():
    topics = load_topics_from_emotional_stories()
    existing_counts = load_existing_counts()
    next_id = next_neutral_id(existing_counts)

    model, tokenizer, runtime = load_model_and_tokenizer(MODEL_NAME)  # "local_gguf" or "hf"
    resolved_model_name = runtime["model_name"]

    for topic in topics:
        completed = existing_counts.get(topic, 0)
        if completed >= N_NEUTRAL_SAMPLES_PER_TOPIC:
            continue

        remaining = N_NEUTRAL_SAMPLES_PER_TOPIC - completed
        while remaining > 0:
            batch_size = min(N_STORIES, remaining)
            neutral_prompt = build_neutral_story_prompt(topic=topic, n_stories=batch_size)
            retries = 0
            parsed_stories = []
            while not parsed_stories and retries < 3:
                batch_output = generate_text(
                    model,
                    tokenizer,
                    runtime,
                    neutral_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    do_sample=True,
                )
                parsed_stories = parse_story_batch(batch_output)
                retries += 1
            if not parsed_stories:
                print(f"WARNING: failed to parse stories for topic '{topic}' after {retries} attempts, skipping batch.")
                break

            stories_to_write = parsed_stories[:batch_size]
            for neutral_story in stories_to_write:
                row = {
                    "id": f"neu_story_{next_id:06d}",
                    "topic": topic,
                    "prompt": neutral_prompt,
                    "generated_text": neutral_story,
                    "n_stories": batch_size,
                    "model_name": resolved_model_name,
                }
                append_jsonl(row, NEUTRAL_STORIES_PATH)
                next_id += 1
                existing_counts[topic] = existing_counts.get(topic, 0) + 1
                remaining -= 1


if __name__ == "__main__":
    main()
