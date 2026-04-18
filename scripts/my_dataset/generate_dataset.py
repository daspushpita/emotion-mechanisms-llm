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
)
from emotion_mechanisms.data import (
    append_jsonl,
    build_emotional_prompt,
    build_neutral_prompt,
    build_story_prompt,
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
story_flag = True

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


def main():
    topics = load_topics(TOPICS_PATH)
    model, tokenizer, runtime = load_model_and_tokenizer(MODEL_NAME)
    resolved_model_name = runtime["model_name"]

    if story_flag:
        story_template = load_template(STORY_TEMPLATE_PATH)
        story_out_path = OUT_DIR / "emotional_stories.jsonl"
        init_jsonl(story_out_path)
        emo_id = 0
        for topic in topics:
            for emotion in EMOTIONS:
                for batch_start in range(0, N_SAMPLES_PER_TOPIC_EMOTION, N_STORIES):
                    batch_size = min(N_STORIES, N_SAMPLES_PER_TOPIC_EMOTION - batch_start)
                    emotional_prompt = build_story_prompt(story_template,
                                        topic=topic, emotion=emotion, n_stories=batch_size)
                    
                    batch_output = generate_text(model, tokenizer, runtime, emotional_prompt,
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    temperature=TEMPERATURE, top_p=0.95,
                                    repetition_penalty=1.05, do_sample=True)
                    
                    parsed_stories = parse_story_batch(batch_output)
                    if not parsed_stories:
                        continue

                    stories_to_write = parsed_stories[:batch_size]

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

    else:
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
                neutral_dialogue = generate_text(
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
                emotional_prompt = build_emotional_prompt(
                    emotional_template,
                    topic=topic,
                    emotion=emotion,
                    n_stories=N_STORIES,
                )
                for _ in range(N_SAMPLES_PER_TOPIC_EMOTION):
                    emotional_dialogue = generate_text(
                        model,
                        tokenizer,
                        runtime,
                        emotional_prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=0.95,
                        repetition_penalty=1.05,
                        do_sample=True,
                    )
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


if __name__ == "__main__":
    main()
