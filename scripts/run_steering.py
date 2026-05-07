#should run broad positive-valence and surgical conflict-avoidance steering sweeps.
import os, sys
import importlib
import argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    
import emotion_mechanisms.steering as steering
import emotion_mechanisms.evals as eval
importlib.reload(steering)
importlib.reload(eval)

PILOT_ALPHAS = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]
PILOT_N      = 10


def get_args():
    parser = argparse.ArgumentParser(description="Steering Sweep CLI")

    # Experiment Hyperparameters
    parser.add_argument("--layers", type=int, nargs='+', required=True, help="Layers to steer (e.g. 10 12 14)")
    parser.add_argument("--alphas", type=float, nargs='+', default=[-0.1, 0.0, 0.1], help="Steering strengths")
    parser.add_argument("--n_samples", type=int, default=300, help="Number of prompts to test")
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--mode", type=str, choices=["pilot", "full"], default="pilot", help="pilot: sweep layers/alphas; full: run single layer/alpha")

    parser.add_argument("--final_layer", type=int, default=43, help="Final Layer to steer")
    parser.add_argument("--final_alpha", type=float, default=0.0, help="Steering strength for full run")
    parser.add_argument("--batch_size", type=int, default=32, help="Generation batch size")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save full run results (required for --mode full)")


    # Model & Path Configs
    parser.add_argument("--analysis_model", type=str, required=True, help="ID of the model being steered")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model used for evaluation")
    parser.add_argument("--steering_path", type=str, required=True, help="Path to .npy direction")
    parser.add_argument("--file1", type=str, required=True, help="Path to dataset file 1")
    parser.add_argument("--file2", type=str, required=True, help="Path to dataset file 2")

    args = parser.parse_args()
    return args

def sample_balanced(file1, file2, n):
    half = n // 2
    f1 = eval.run_eval.load_jsonl(Path(file1))
    f2 = eval.run_eval.load_jsonl(Path(file2))
    return f1[:half] + f2[:n - half]


def pilot_steering_sweep(args):

    # 1. Initialize the eval setup
    pilot_eval = eval.run_eval(model_id=args.analysis_model,
                                judge_model=args.judge_model,
                                steering_direction_path=args.steering_path,
                                file1_path=args.file1,
                                file2_path=args.file2)

    # 2. Get your prompt subset — balanced across both files
    pilot_prompts = [d["question"] for d in sample_balanced(args.file1, args.file2, args.n_samples)]
    direction = np.load(args.steering_path)

    # 3. The Nested Sweep
    for layer in args.layers:
        steer = eval.steering.ActivationSteer(model=pilot_eval.model,
                                            tokenizer=pilot_eval.tokenizer,
                                            layer_idx=layer,
                                            direction=direction)

        for alpha in args.alphas:
            print(f"\n🚀 [Layer {layer} | Alpha {alpha}] Running...")
            responses = steer.generate_batch(pilot_prompts, alpha=alpha, max_new_tokens=args.max_tokens)
            for resp in responses:
                print(f"  > {resp[:120]}...")
                


def steering_run(args):
    assert args.output_dir is not None, "--output_dir is required for --mode full"

    # 1. Initialize the eval setup
    run = eval.run_eval(model_id=args.analysis_model,
                        judge_model=args.judge_model,
                        steering_direction_path=args.steering_path,
                        file1_path=args.file1,
                        file2_path=args.file2)

    # 2. Replace dataset with balanced subset so generate_modified_responses uses it
    run.dataset = sample_balanced(args.file1, args.file2, args.n_samples)

    out_path = Path(args.output_dir) / f"steered_L{args.final_layer}_a{args.final_alpha}.jsonl"
    print(f"\n🚀 [Layer {args.final_layer} | Alpha {args.final_alpha}] -> {out_path}")
    
    rows = run.generate_modified_responses(use_steering=True,
                                            output_path=out_path,
                                            alpha=args.final_alpha,
                                            layer_idx=args.final_layer,
                                            batch_size=args.batch_size)
    
    print(f"{len(rows)} rows saved.")


if __name__ == "__main__":

    args = get_args()
    if args.mode == "pilot":
        pilot_steering_sweep(args)
    else:
        steering_run(args)