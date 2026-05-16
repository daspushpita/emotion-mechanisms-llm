import os, sys
import importlib
import argparse
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import emotion_mechanisms.evals as evals
importlib.reload(evals)


def get_args():
    parser = argparse.ArgumentParser(description="Judge Responses CLI")
    parser.add_argument("--api_key_path", type=str, default=None, help="Path to file containing Anthropic API key")
    parser.add_argument("--singleturn_dir", type=str, required=True, help="Path to singleturn raw dir")
    parser.add_argument("--multiturn_dir",  type=str, required=True, help="Path to multiturn raw dir")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5")
    return parser.parse_args()


def main():
    args = get_args()

    import anthropic
    if args.api_key_path:
        api_key = open(args.api_key_path).read().strip()
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    client  = anthropic.Anthropic(api_key=api_key)
    judge   = evals.Sycophancy_conversation(client, args.model)

    raw_files = (
        glob.glob(f"{args.singleturn_dir}/*.jsonl") +
        glob.glob(f"{args.multiturn_dir}/*.jsonl")
    )
    raw_files = [p for p in raw_files if not p.endswith("_judged.jsonl")]

    for in_path in raw_files:
        out_path = Path(in_path).with_suffix("").name + "_judged.jsonl"
        out_path = Path(in_path).parent / out_path
        if out_path.exists():
            print(f"Skipping {out_path.name} — already exists")
            continue
        print(f"Judging {Path(in_path).name} ...")
        judge.judge_file(Path(in_path), out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()