import numpy as np
import torch

class ActivationSteer:
    def __init__(self, model, tokenizer, 
                layer_idx: int, 
                direction: np.ndarray):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self._alpha = 0.0
        self.direction = torch.tensor(direction, dtype=torch.float32)
        
    def _make_hook(self):
        def hook_fn(_module, _input, output):
            hidden_vector = output[0] if isinstance(output, tuple) else output
            steering_direction = self.direction.to(hidden_vector.device, dtype=hidden_vector.dtype)
            hidden_vector = hidden_vector + self._alpha * steering_direction
            return (hidden_vector,) + output[1:] if isinstance(output, tuple) else hidden_vector
        
        return hook_fn

    def generate(self, prompt: str, alpha: float, max_new_tokens: int = 300) -> str:
        return self.generate_batch([prompt], alpha=alpha, max_new_tokens=max_new_tokens)[0]

    def generate_batch(self, prompts: list[str], alpha: float,
                        max_new_tokens: int = 300, batch_size: int = 32) -> list[str]:
        results = []
        for i in range(0, len(prompts), batch_size):
            results.extend(self._generate_chunk(prompts[i : i + batch_size], alpha, max_new_tokens))
        return results

    def _generate_chunk(self, prompts: list[str], alpha: float, max_new_tokens: int) -> list[str]:
        self._alpha = alpha
        target_layer = self.model.model.layers[self.layer_idx]
        hook_handle = target_layer.register_forward_hook(self._make_hook())

        try:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            token_ids = []
            for p in prompts:
                ids = self.tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
                if hasattr(ids, "input_ids"):
                    ids = ids.input_ids
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                token_ids.append(ids)

            inputs = self.tokenizer.pad({"input_ids": token_ids}, return_tensors="pt").to(self.model.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            return [self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True) for out in outputs]
        finally:
            hook_handle.remove()
