import numpy as np
import torch
from emotion_mechanisms.hooks import _get_layers

class ActivationSteer:
    def __init__(self, model, tokenizer,
                layer_idx: int,
                direction: np.ndarray,
                residual_norm: float = 1.0,
                persona_direction=None, beta=0.0):
        
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self._alpha = 0.0
        self.residual_norm = residual_norm
        self.direction = torch.tensor(direction, dtype=torch.float32)

        #For injecting a secondary steering direction (e.g. persona) alongside the primary (e.g. emotion), we use a simple convex combination of the two directions, where beta is the mixing coefficient for the persona direction.
        self.beta = beta
        self.persona_direction = (torch.tensor(persona_direction, dtype=torch.float32) 
            if persona_direction is not None else None)


    def _make_hook(self):
        def hook_fn(_module, _input, output):
            hidden_vector = output[0] if isinstance(output, tuple) else output

            # Only steer newly generated tokens, not the prompt: during the
            # prefill pass hidden_vector covers the whole prompt (seq_len > 1);
            # during each cached decode step it's just the new token (seq_len == 1).
            # Steering the prompt would corrupt the model's own reading of the
            # user's message, confounding sycophancy measured on the response.
            if hidden_vector.shape[1] > 1:
                return output

            steering_direction = self.direction.to(hidden_vector.device, dtype=hidden_vector.dtype)
            hidden_vector = hidden_vector + self._alpha * self.residual_norm * steering_direction
            
            if self.persona_direction is not None and self.beta != 0.0:
                persona = self.persona_direction.to(hidden_vector.device, dtype=hidden_vector.dtype)
                hidden_vector = hidden_vector + self.beta * self.residual_norm * persona
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
        target_layer = _get_layers(self.model)[self.layer_idx]
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
        target_layer = _get_layers(self.model)[self.layer_idx]
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
