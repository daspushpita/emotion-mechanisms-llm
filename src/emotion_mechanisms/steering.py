import numpy as np
import torch

class ActivationSteer:
    def __init__(self, model, tokenizer,
                layer_idx: int,
                direction: np.ndarray,
                residual_norm: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self._alpha = 0.0
        self.residual_norm = residual_norm
        self.direction = torch.tensor(direction, dtype=torch.float32)

    def _make_hook(self):
        def hook_fn(_module, _input, output):
            hidden_vector = output[0] if isinstance(output, tuple) else output
            steering_direction = self.direction.to(hidden_vector.device, dtype=hidden_vector.dtype)
            hidden_vector = hidden_vector + self._alpha * self.residual_norm * steering_direction
            return (hidden_vector,) + output[1:] if isinstance(output, tuple) else hidden_vector

        return hook_fn

    def generate(self, prompt: str, alpha: float, max_new_tokens: int = 300) -> str:
        return self.generate_batch([prompt], alpha=alpha, max_new_tokens=max_new_tokens)[0]

    def generate_batch(self, prompts: list[str], alpha: float,
                        max_new_tokens: int = 300, batch_size: int = 32,
                        system_prompt: str = None) -> list[str]:
        results = []
        for i in range(0, len(prompts), batch_size):
            results.extend(self._generate_chunk(prompts[i : i + batch_size], alpha, max_new_tokens, system_prompt))
        return results

    def _generate_chunk(self, prompts: list[str], alpha: float, max_new_tokens: int,
                        system_prompt: str = None) -> list[str]:
        self._alpha = alpha
        target_layer = self.model.model.layers[self.layer_idx]
        hook_handle = target_layer.register_forward_hook(self._make_hook())

        try:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            token_ids = []
            for p in prompts:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                    
                messages.append({"role": "user", "content": p})
                ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                
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
    def generate_from_messages(self, messages_list: list[list[dict]], alpha: float, 
                            max_new_tokens: int = 300) -> str:
        self._alpha = alpha
        target_layer = self.model.model.layers[self.layer_idx]
        hook_handle = target_layer.register_forward_hook(self._make_hook())
        
        try:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
            token_ids = []
            for messages in messages_list:
                ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                if hasattr(ids, "input_ids"):
                    ids = ids.input_ids
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                token_ids.append(ids)
                
            inputs = self.tokenizer.pad({"input_ids": token_ids}, return_tensors="pt").to(self.model.device)
            prompt_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                            do_sample=False)
            return [self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True) for out in outputs]
        finally:
            hook_handle.remove()
