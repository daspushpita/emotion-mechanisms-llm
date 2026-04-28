import json
import os
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from emotion_mechanisms.config import (
    EMOTIONS,
    EMOTIONAL_TEMPLATE_PATH,
    NEUTRAL_TEMPLATE_PATH,
    STORY_TEMPLATE_PATH,
    NEUTRAL_STORIES_TEMPLATE_PATH,
    EMOTIONAL_STORIES_DATASET,
    NEUTRAL_STORIES_DATASET,
    TOPICS_PATH,
    PROCESSED_DIR,
    N_STORIES_PER_BATCH,
    N_SAMPLES_PER_TOPIC_EMOTION,
    N_NEUTRAL_PER_TOPIC as N_NEUTRAL_SAMPLES_PER_TOPIC,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
    SEED,
    GENERATION_BACKEND,
    STORY_GENERATION_MODE,
    GENERATION_PARSE_RETRIES,
)
from emotion_mechanisms.data import (
    append_jsonl,
    build_emotional_prompt,
    build_neutral_prompt,
    build_story_prompt,
    build_neutral_story_prompt,
    init_jsonl,
    load_template,
    load_topics,
)
from emotion_mechanisms.model_loader import (
    generate_text,
    load_model_and_tokenizer,
)

MODEL_NAME = os.environ.get("EMOTION_MODEL", GENERATION_BACKEND)

OUT_DIR = PROCESSED_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STORIES = N_STORIES_PER_BATCH
story_flag = STORY_GENERATION_MODE

random.seed(SEED)
torch.manual_seed(SEED)


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


def load_existing_story_counts(path: Path, has_emotion: bool) -> tuple[dict[tuple[str, str | None], int], int]:
    if not path.exists():
        return {}, 0

    counts: dict[tuple[str, str | None], int] = {}
    n_rows = 0
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            topic = row["topic"]
            emotion = row.get("emotion") if has_emotion else None
            counts[(topic, emotion)] = counts.get((topic, emotion), 0) + 1
            n_rows += 1
    return counts, n_rows


def generate_parsed_stories(model, tokenizer, runtime, prompt: str, batch_size: int, label: str) -> list[str]:
    for attempt in range(1, GENERATION_PARSE_RETRIES + 1):
        batch_output = generate_text(model, tokenizer, runtime, prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE, top_p=TOP_P,
                        repetition_penalty=REPETITION_PENALTY, do_sample=True)

        parsed_stories = parse_story_batch(batch_output)
        if parsed_stories:
            if len(parsed_stories) < batch_size:
                print(
                    f"WARNING: parsed {len(parsed_stories)} of {batch_size} requested stories "
                    f"for {label} on attempt {attempt}."
                )
            return parsed_stories[:batch_size]

    raise RuntimeError(
        f"Failed to parse generated stories for {label} after "
        f"{GENERATION_PARSE_RETRIES} attempts."
    )


def main():
    topics = load_topics(TOPICS_PATH)
    model, tokenizer, runtime = load_model_and_tokenizer(MODEL_NAME)
    resolved_model_name = runtime["model_name"]

    if story_flag == "emotional_stories":
        story_template = load_template(STORY_TEMPLATE_PATH)
        story_out_path = EMOTIONAL_STORIES_DATASET
        story_out_path.parent.mkdir(parents=True, exist_ok=True)
        existing_counts, emo_id = load_existing_story_counts(story_out_path, has_emotion=True)
        for topic in topics:
            for emotion in EMOTIONS:
                key = (topic, emotion)
                remaining = N_SAMPLES_PER_TOPIC_EMOTION - existing_counts.get(key, 0)
                while remaining > 0:
                    batch_size = min(N_STORIES, remaining)
                    emotional_prompt = build_story_prompt(story_template,
                                        topic=topic, emotion=emotion, n_stories=batch_size)

                    stories_to_write = generate_parsed_stories(
                        model, tokenizer, runtime, emotional_prompt, batch_size,
                        label=f"topic={topic!r}, emotion={emotion!r}",
                    )

                    for emotional_story in stories_to_write:
                        row = {
                            "id": f"emo_{emo_id:06d}",
                            "topic": topic,
                            "emotion": emotion,
                            "prompt": emotional_prompt,
                            "generated_text": emotional_story,
                            "n_stories": batch_size,
                            "model_name": resolved_model_name,
                        }
                        append_jsonl(row, story_out_path)
                        emo_id += 1
                        existing_counts[key] = existing_counts.get(key, 0) + 1
                        remaining -= 1
                        
    elif story_flag == "neutral_stories":
        story_template = load_template(NEUTRAL_STORIES_TEMPLATE_PATH)
        story_out_path = NEUTRAL_STORIES_DATASET
        story_out_path.parent.mkdir(parents=True, exist_ok=True)
        existing_counts, next_id = load_existing_story_counts(story_out_path, has_emotion=False)
        for topic in topics:
            key = (topic, None)
            remaining = N_NEUTRAL_SAMPLES_PER_TOPIC - existing_counts.get(key, 0)
            while remaining > 0:
                batch_size = min(N_STORIES, remaining)
                neutral_prompt = build_neutral_story_prompt(story_template,
                                    topic=topic, n_stories=batch_size)

                stories_to_write = generate_parsed_stories(
                    model, tokenizer, runtime, neutral_prompt, batch_size,
                    label=f"topic={topic!r}, condition='neutral'")
                for neutral_story in stories_to_write:
                    row = {
                        "id": f"neu_story_{next_id:06d}",
                        "topic": topic,
                        "prompt": neutral_prompt,
                        "generated_text": neutral_story,
                        "n_stories": batch_size,
                        "model_name": resolved_model_name,
                    }
                    append_jsonl(row, story_out_path)
                    next_id += 1
                    existing_counts[key] = existing_counts.get(key, 0) + 1
                    remaining -= 1


    elif story_flag == "dialogues":
        emotional_template = load_template(EMOTIONAL_TEMPLATE_PATH)
        neutral_template = load_template(NEUTRAL_TEMPLATE_PATH)
        emo_id = 0
        neu_id = 0
        emotional_out_path = OUT_DIR / "emotional_dialogues.jsonl"
        neutral_out_path = OUT_DIR / "neutral_dialogues.jsonl"
        init_jsonl(emotional_out_path)
        init_jsonl(neutral_out_path)
        for topic in topics:
            neutral_prompt = build_neutral_prompt(neutral_template, topic=topic, n_stories=N_STORIES)
            for _ in range(N_NEUTRAL_SAMPLES_PER_TOPIC):
                neutral_dialogue = generate_text(model,
                                    tokenizer, runtime, neutral_prompt,
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    temperature=TEMPERATURE, top_p=TOP_P,
                                    repetition_penalty=REPETITION_PENALTY, do_sample=True)
                row = {
                    "id": f"neu_{neu_id:06d}",
                    "topic": topic,
                    "neutral_prompt": neutral_prompt,
                    "neutral_dialogue": neutral_dialogue,
                    "n_stories": N_STORIES,
                    "model_name": resolved_model_name,
                }
                append_jsonl(row, neutral_out_path)
                neu_id += 1

            for emotion in EMOTIONS:
                emotional_prompt = build_emotional_prompt(emotional_template, topic=topic, emotion=emotion, n_stories=N_STORIES)
                for _ in range(N_SAMPLES_PER_TOPIC_EMOTION):
                    emotional_dialogue = generate_text(model, 
                                        tokenizer, runtime, emotional_prompt,
                                        max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                        top_p=0.95, repetition_penalty=1.05, do_sample=True)
                    row = {
                        "id": f"emo_{emo_id:06d}",
                        "topic": topic,
                        "emotion": emotion,
                        "emotional_prompt": emotional_prompt,
                        "emotional_dialogue": emotional_dialogue,
                        "n_stories": N_STORIES,
                        "model_name": resolved_model_name,
                    }
                    append_jsonl(row, emotional_out_path)
                    emo_id += 1
    else:
        raise ValueError("Story flag not recognized or not implemented...")

if __name__ == "__main__":
    main()
