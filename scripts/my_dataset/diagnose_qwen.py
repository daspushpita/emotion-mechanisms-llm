import torch
from src.emotion_mechanisms.model_loader import load_model_and_tokenizer
from src.emotion_mechanisms.hooks import ActivationExtractor

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def main():

    model, tokenizer, runtime = load_model_and_tokenizer(model_name="hf", analysis=True, analysis_model=MODEL_NAME)
    text = "Alex stared at the email, heart racing, unsure what to do next."
    inputs = tokenizer(text, return_tensors="pt")

    device = runtime.get("input_device", next(model.parameters()).device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    extractor = ActivationExtractor(model, [0, 8, 16, 24], token_position="last")
    activations = extractor.extract(**inputs)

    for layer_idx, vector in activations.items():
        print(f"Layer {layer_idx}: vector shape = {vector.shape}")


if __name__ == "__main__":
    main()
