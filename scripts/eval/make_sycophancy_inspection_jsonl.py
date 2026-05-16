"""
Sample steering check JSONL files into human-readable Markdown for review.
Each entry shows the agree/disagree pair for the same question side by side,
so you can see at a glance whether the model flipped its answer.

Examples:
    python scripts/make_sycophancy_inspection_jsonl.py
    python scripts/make_sycophancy_inspection_jsonl.py --n 6 --seed 7
    python scripts/make_sycophancy_inspection_jsonl.py --format jsonl
    python scripts/make_sycophancy_inspection_jsonl.py --input-dir results/steering/checks --out-dir results/steering/checks/readable
"""

import argparse
import json
import re
import random
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("../results/steering/pilot_runs")
DEFAULT_OUT_DIR = DEFAULT_INPUT_DIR / "readable"

_CHOICE_RE = re.compile(r"\(([AB])\)")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows


def load_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """Split flat results list (first half agree, second half disagree) into pairs."""
    n = len(rows) // 2
    return [(rows[i], rows[n + i]) for i in range(n)]


def sample_pairs(
    pairs: list[tuple[dict, dict]], n: int, seed: int, source_name: str
) -> list[tuple[dict, dict]]:
    if len(pairs) <= n:
        return pairs
    rng = random.Random(f"{seed}:{source_name}")
    indices = sorted(rng.sample(range(len(pairs)), n))
    return [pairs[i] for i in indices]


def parse_choice(response: str) -> str | None:
    m = _CHOICE_RE.search(response)
    return f"({m.group(1)})" if m else None


def compact_prompt(prompt: str) -> tuple[str, str]:
    if "\n\nChoices:" not in prompt:
        return prompt, ""
    setup, choices = prompt.split("\n\nChoices:", 1)
    return setup, "Choices:" + choices


def make_pair_review_record(
    agree_row: dict[str, Any],
    disagree_row: dict[str, Any],
    source_file: str,
    pair_no: int,
) -> dict[str, Any]:
    a_choice = parse_choice(str(agree_row.get("response", "")))
    d_choice = parse_choice(str(disagree_row.get("response", "")))
    flipped = (a_choice != d_choice) if (a_choice is not None and d_choice is not None) else None

    meta = agree_row.get("_meta") or {}
    pair_id = meta.get("pair_id") if isinstance(meta, dict) else None

    a_scores = {k: agree_row.get(k) for k in ("sycophantic", "harsh", "distressed") if k in agree_row}
    d_scores = {k: disagree_row.get(k) for k in ("sycophantic", "harsh", "distressed") if k in disagree_row}

    a_setup, a_choices = compact_prompt(str(agree_row.get("prompt", "")))
    d_setup, d_choices = compact_prompt(str(disagree_row.get("prompt", "")))

    flip_label = {True: "YES", False: "NO", None: "UNPARSEABLE"}[flipped]

    readable = "\n".join([
        f"FILE: {source_file}",
        f"PAIR: {pair_no}  |  pair_id: {pair_id}",
        f"ALPHA: {agree_row.get('alpha')} | LAYER: {agree_row.get('layer_idx')} | STEERING: {agree_row.get('use_steering')}",
        f"FLIPPED: {flip_label}  (agree→{a_choice or '?'}  disagree→{d_choice or '?'})",
        "",
        "── AGREE PROMPT ──────────────────────────────────────────",
        a_setup,
        "",
        a_choices,
        "",
        "AGREE RESPONSE:",
        str(agree_row.get("response", "")),
        "",
        "AGREE SCORES: " + (", ".join(f"{k}={v}" for k, v in a_scores.items()) if a_scores else "<not present>"),
        "AGREE JUDGE:",
        str(agree_row.get("judge_raw", "<not present>")),
        "",
        "── DISAGREE PROMPT ───────────────────────────────────────",
        d_setup,
        "",
        d_choices,
        "",
        "DISAGREE RESPONSE:",
        str(disagree_row.get("response", "")),
        "",
        "DISAGREE SCORES: " + (", ".join(f"{k}={v}" for k, v in d_scores.items()) if d_scores else "<not present>"),
        "DISAGREE JUDGE:",
        str(disagree_row.get("judge_raw", "<not present>")),
    ])

    return {
        "source_file": source_file,
        "pair_no": pair_no,
        "pair_id": pair_id,
        "alpha": agree_row.get("alpha"),
        "layer_idx": agree_row.get("layer_idx"),
        "use_steering": agree_row.get("use_steering"),
        "flipped": flipped,
        "agree_choice": a_choice,
        "disagree_choice": d_choice,
        "agree": {
            "prompt": agree_row.get("prompt"),
            "response": agree_row.get("response"),
            "judge_raw": agree_row.get("judge_raw"),
            **a_scores,
        },
        "disagree": {
            "prompt": disagree_row.get("prompt"),
            "response": disagree_row.get("response"),
            "judge_raw": disagree_row.get("judge_raw"),
            **d_scores,
        },
        "readable": readable,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, rows: list[dict[str, Any]], source_file: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        f"# Sycophancy Response Inspection: {source_file}",
        "",
        f"Showing {len(rows)} sampled pairs.",
        "",
    ]

    for row in rows:
        flip_label = {True: "**YES**", False: "NO", None: "unparseable"}.get(row.get("flipped"), "?")
        a = row["agree"]
        d = row["disagree"]

        a_scores = ", ".join(f"{k}: `{a[k]}`" for k in ("sycophantic", "harsh", "distressed") if k in a)
        d_scores = ", ".join(f"{k}: `{d[k]}`" for k in ("sycophantic", "harsh", "distressed") if k in d)

        sections.extend([
            f"## Pair {row['pair_no']} | pair_id {row.get('pair_id')}",
            "",
            f"- Source: `{row['source_file']}`",
            f"- Alpha: `{row.get('alpha')}` | Layer: `{row.get('layer_idx')}` | Steering: `{row.get('use_steering')}`",
            f"- Flipped: {flip_label} — agree→`{row.get('agree_choice') or '?'}` / disagree→`{row.get('disagree_choice') or '?'}`",
            "",
            "### Agree prompt",
            "",
            str(a.get("prompt", "")),
            "",
            "### Agree response",
            "",
            str(a.get("response", "")),
            "",
            f"Scores: {a_scores or '`not present`'}",
            "",
            "```text",
            str(a.get("judge_raw", "<not present>")),
            "```",
            "",
            "### Disagree prompt",
            "",
            str(d.get("prompt", "")),
            "",
            "### Disagree response",
            "",
            str(d.get("response", "")),
            "",
            f"Scores: {d_scores or '`not present`'}",
            "",
            "```text",
            str(d.get("judge_raw", "<not present>")),
            "```",
            "",
            "---",
            "",
        ])

    path.write_text("\n".join(sections), encoding="utf-8")


def iter_input_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.glob("*.jsonl")
        if not path.name.endswith("_readable.jsonl")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create small human-readable samples from sycophancy steering checks."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=6, help="Pairs to sample from each file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable samples.")
    parser.add_argument(
        "--format",
        choices=("md", "jsonl", "both"),
        default="md",
        help="Output format. Default is Markdown because it is easiest to inspect by eye.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Write every pair instead of sampling.",
    )
    args = parser.parse_args()

    input_files = iter_input_files(args.input_dir)
    if not input_files:
        raise SystemExit(f"No JSONL files found in {args.input_dir}")

    print(f"Reading {len(input_files)} files from {args.input_dir}")
    print(f"Writing readable samples to {args.out_dir}")

    for input_path in input_files:
        rows = load_jsonl(input_path)
        pairs = load_pairs(rows)
        selected = pairs if args.all else sample_pairs(pairs, args.n, args.seed, input_path.name)
        review_rows = [
            make_pair_review_record(a, d, input_path.name, pair_no)
            for pair_no, (a, d) in enumerate(selected, 1)
        ]

        suffix = "all_pairs" if args.all else f"{len(review_rows)}_pairs"
        out_paths = []

        if args.format in ("jsonl", "both"):
            out_path = args.out_dir / f"{input_path.stem}_{suffix}.jsonl"
            write_jsonl(out_path, review_rows)
            out_paths.append(str(out_path))

        if args.format in ("md", "both"):
            out_path = args.out_dir / f"{input_path.stem}_{suffix}.md"
            write_markdown(out_path, review_rows, input_path.name)
            out_paths.append(str(out_path))

        print(f"{input_path.name}: {len(review_rows)}/{len(pairs)} pairs -> {', '.join(out_paths)}")


if __name__ == "__main__":
    main()
