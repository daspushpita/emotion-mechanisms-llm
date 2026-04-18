from __future__ import annotations

import json
from pathlib import Path

def load_template(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        my_template = f.read()
    return my_template

def load_topics(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        topics = []
        for line in f:
            if topic := line.strip():
                topics.append(topic)
    return topics
                

def build_emotional_prompt(template: str, topic: str, emotion: str, n_stories: int) -> str:
    return template.format(
        topic=topic,
        emotion=emotion,
        n_stories=n_stories,
    )

def build_neutral_prompt(template: str, topic: str, n_stories: int) -> str:
    return template.format(
        topic=topic,
        n_stories=n_stories,
    )

def build_story_prompt(template: str, topic: str, emotion: str, n_stories: int) -> str:
    return template.format(
        topic=topic,
        emotion=emotion,
        n_stories=n_stories,
    )

def save_jsonl(rows: list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            json_string = json.dumps(row, ensure_ascii=False)
            fout.write(json_string + "\n")

def init_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass

def append_jsonl(row: dict, path: Path) -> None:
    with open(path, "a", encoding="utf-8") as fout:
        json_string = json.dumps(row, ensure_ascii=False)
        fout.write(json_string + "\n")
