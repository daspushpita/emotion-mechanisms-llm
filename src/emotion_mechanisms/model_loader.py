from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_MODEL = "Qwen/Qwen2.5-32B-Instruct"
GGUF_MODEL = Path("~/models/qwen25_32b/qwen2.5-32b-instruct-q4_k_m.gguf").expanduser()



def select_model(model_name: str, analysis: bool = False, analysis_model: str = "") -> str:
        
    if model_name == "hf":
        if analysis:
            if not analysis_model:
                raise ValueError("Analysis model is not set")
            return analysis_model
        if not HF_MODEL:
            raise ValueError("HF_MODEL is not set")
        return HF_MODEL
    elif model_name == "local_gguf":
        if not GGUF_MODEL.exists():
            raise FileNotFoundError(f"GGUF model not found at {GGUF_MODEL}")
        return str(GGUF_MODEL)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_dtype(device: str):
    return torch.float16 if device == "mps" else torch.float32


def load_model_and_tokenizer(model_name: str, *, n_ctx: int = 4096, n_gpu_layers: int = -1,
                             analysis: bool = False, analysis_model: str = ""):

    resolved_model = select_model(model_name, analysis=analysis, analysis_model=analysis_model)

    if resolved_model.endswith(".gguf"):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError("Loading a GGUF model requires llama-cpp-python to be installed.") from exc

        model = Llama(model_path=resolved_model, n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers, n_batch=512,
                        n_threads=8,flash_attn=True,
                        seed=42, default_temperature=0.2,
                        chat_format="chatml", verbose=False)

        runtime = {"backend": "llama_cpp",
                    "device": "metal" if n_gpu_layers != 0 else "cpu",
                    "model_name": resolved_model}

        return model, None, runtime

    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    model = AutoModelForCausalLM.from_pretrained(resolved_model, torch_dtype=torch.bfloat16, device_map="auto")

    input_device = next(model.parameters()).device

    model.eval()
    runtime = {"backend": "transformers", "device": "auto",
                "input_device": input_device, "model_name": resolved_model}
    return model, tokenizer, runtime


def generate_text(model,
        tokenizer,
        runtime: dict[str, Any],
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
    ) -> str:
    backend = runtime["backend"]

    if backend == "llama_cpp":
        response = model.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 0.0,
                    top_p=top_p,
                    repeat_penalty=repetition_penalty,
        )
        return response["choices"][0]["message"]["content"].strip()

    device = runtime.get("input_device", runtime["device"])
    if getattr(tokenizer, "chat_template", None):
        chat_out = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
        # older transformers returns a raw tensor; newer returns BatchEncoding
        input_ids = (chat_out if isinstance(chat_out, torch.Tensor) else chat_out["input_ids"]).to(device)
        input_length = input_ids.shape[-1]
        generation_inputs = {"input_ids": input_ids}
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[-1]
        generation_inputs = inputs

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(
            **generation_inputs,
            **generation_kwargs,
        )

    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
