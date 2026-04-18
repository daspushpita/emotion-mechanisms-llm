"""
Diagnostic script: Qwen2.5-32B on M1 Pro 32GB
Tests TransformerLens compatibility, MPS, model loading, and activation extraction.
Run with: python diagnose_qwen.py
"""

import torch
import psutil
import os

def mb(bytes): return round(bytes / 1024**2)
def gb(bytes): return round(bytes / 1024**3, 2)

def print_header(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def print_result(label, passed, detail=""):
    icon = "PASS" if passed else "FAIL"
    print(f"  [{icon}] {label}")
    if detail:
        print(f"        {detail}")

# ── Stage 1: TransformerLens compatibility ────────────────────
print_header("Stage 1: TransformerLens compatibility")

tl_available = False
try:
    import transformer_lens
    from transformer_lens import HookedTransformer
    # Check if Qwen2.5 is in supported model list
    supported = transformer_lens.utilities.official_model_names.get_official_model_list()
    qwen_supported = any("qwen" in m.lower() for m in supported)
    print_result("TransformerLens installed", True, f"version {transformer_lens.__version__}")
    print_result("Qwen2.5 in supported models", qwen_supported,
                 "Will use TL directly" if qwen_supported else "Will use HuggingFace hooks instead")
    tl_available = qwen_supported
except ImportError:
    print_result("TransformerLens installed", False, "Run: pip install transformer_lens")
except Exception as e:
    print_result("TransformerLens check", False, str(e))

# ── Stage 2: MPS device ───────────────────────────────────────
print_header("Stage 2: MPS (Apple Silicon) device")

mps_ok = False
try:
    mps_ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    print_result("MPS available", mps_ok,
                 "Device: mps" if mps_ok else "Falling back to CPU — will be slow")
    if mps_ok:
        t = torch.ones(3, device="mps")
        print_result("MPS tensor creation", True, f"tensor {t.tolist()} on mps")
except Exception as e:
    print_result("MPS check", False, str(e))

device = "mps" if mps_ok else "cpu"
print(f"\n  Using device: {device}")

# ── Stage 3: System memory ────────────────────────────────────
print_header("Stage 3: System RAM check")

ram = psutil.virtual_memory()
total_gb = gb(ram.total)
available_gb = gb(ram.available)
used_pct = round(ram.percent)

print(f"  Total RAM    : {total_gb} GB")
print(f"  Available    : {available_gb} GB")
print(f"  Used         : {used_pct}%")

# Qwen2.5-32B at 4-bit needs ~18-20GB
needed_gb = 20
headroom_gb = round(available_gb - needed_gb, 1)
fits = available_gb >= needed_gb

print_result(f"Enough RAM for 32B at 4-bit (~{needed_gb}GB)", fits,
             f"{headroom_gb}GB headroom after load" if fits else
             f"Short by {abs(headroom_gb)}GB — use Colab A100 or try 8B model")

# ── Stage 4: Model load + activation extraction ───────────────
print_header("Stage 4: Model load + activation extraction")

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

if not fits:
    print(f"  Skipping model load — insufficient RAM.")
    print(f"  Try: MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct' to test pipeline on smaller model.")
else:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import numpy as np

        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print_result("Tokenizer loaded", True)

        print(f"  Loading model at 4-bit (this takes 2-5 mins)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print_result("Model loaded at 4-bit", True)

        # Hook into residual stream at middle layers
        # Qwen2.5-32B has 64 layers — we target layers 20-40
        TARGET_LAYERS = [20, 28, 36]
        activations = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output is typically (hidden_states, ...) — grab hidden states
                hidden = output[0] if isinstance(output, tuple) else output
                activations[f"layer_{layer_idx}"] = hidden.detach().cpu().float()
            return hook

        hooks = []
        for layer_idx in TARGET_LAYERS:
            layer = model.model.layers[layer_idx]
            h = layer.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        # Test prompt — emotional content to verify probes will activate
        test_prompt = "She felt a wave of anxiety wash over her as she realized she would have to disagree with her boss."
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        print(f"  Running forward pass on test prompt...")
        with torch.no_grad():
            _ = model(**inputs)

        for h in hooks:
            h.remove()

        # Verify extractions
        all_ok = True
        for layer_idx in TARGET_LAYERS:
            key = f"layer_{layer_idx}"
            if key in activations:
                shape = activations[key].shape
                # Expected: [batch=1, seq_len, hidden_dim=5120 for 32B]
                ok = len(shape) == 3 and shape[2] > 0
                print_result(f"Layer {layer_idx} activations", ok,
                             f"shape {tuple(shape)} | hidden_dim={shape[2]}")
                if not ok:
                    all_ok = False
            else:
                print_result(f"Layer {layer_idx} activations", False, "Hook didn't fire")
                all_ok = False

        print_result("Residual stream extraction", all_ok,
                     "Pipeline is ready!" if all_ok else "Check hook logic")

    except ImportError as e:
        print_result("Required package", False, f"Missing: {e}\nRun: pip install transformers bitsandbytes accelerate")
    except Exception as e:
        print_result("Model load/extraction", False, str(e))

# ── Stage 5: RAM headroom after load ─────────────────────────
print_header("Stage 5: Memory headroom after load")

ram_after = psutil.virtual_memory()
available_after = gb(ram_after.available)
print(f"  Available RAM now : {available_after} GB")
print_result("Enough headroom for probe training (need 4GB+)", available_after >= 4,
             f"{available_after}GB free" )

# ── Summary ───────────────────────────────────────────────────
print_header("Summary & next steps")
if tl_available:
    print("  TransformerLens works → use HookedTransformer directly")
else:
    print("  TransformerLens doesn't support Qwen2.5 natively")
    print("  → Use HuggingFace forward hooks (as in Stage 4 above)")
    print("  → This is fine — hooks give same activations, slightly more code")

print(f"\n  Device       : {device}")
print(f"  RAM available: {available_gb}GB / {total_gb}GB total")
print("\n  If Stage 4 failed due to RAM, options:")
print("  1. Use Qwen/Qwen2.5-7B-Instruct to develop pipeline locally")
print("  2. Run full 32B experiments on Colab A100 (~$2/hr)")
print("  3. Use mlx-lm on M1 for inference, extract via hooks\n")
