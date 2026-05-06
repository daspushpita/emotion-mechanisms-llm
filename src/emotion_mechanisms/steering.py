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
        def hook_fn(module, input, output):
            hidden_vector = output[0] if isinstance(output, tuple) else output
            steering_direction = self.direction.to(hidden_vector.device)
            hidden_vector = hidden_vector + self._alpha * steering_direction
            return (hidden_vector,) + output[1:] if isinstance(output, tuple) else hidden_vector
        
        return hook_fn

    def generate(self, prompt: str, alpha: float,
                max_new_tokens: int = 300) -> str:
        
        self._alpha = alpha
        #Attach the hook
        target_layer = self.model.model.layers[self.layer_idx]
        hook_handle = target_layer.register_forward_hook(self._make_hook)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                            do_sample = False)
                return self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        finally:
            hook_handle.remove()
