import numpy as np
import torch

class ActivationExtractor:
    def __init__(self, model, layer_indices: list[int], token_position: str = "last"):
        if token_position not in {"last", "mean"}:
            raise ValueError(f"Unknown token position: {token_position}")

        self.token_position = token_position
        n_layers = len(model.model.layers)
        if bad := [i for i in layer_indices if not (0 <= i < n_layers)]:
            raise ValueError(f"Layer indices {bad} out of range for model with {n_layers} layers")

        self.model = model
        self.layer_indices = layer_indices
        self.handles = []
        self.activations = {}

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if self.token_position == "last":
                selected_vec = hidden_states[0, -1]
            else:
                selected_vec = hidden_states[0].mean(dim=0)
            self.activations[layer_idx] = selected_vec.detach().float().cpu().numpy()
        return hook_fn

    def _register_hooks(self):
        if self.handles:
            raise RuntimeError("Hooks are already registered — call remove() first")
        for layer_idx in self.layer_indices:
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def extract(self, **model_inputs) -> dict[int, np.ndarray]:
        self.activations = {}
        self._register_hooks()
        try:
            with torch.no_grad():
                _ = self.model(**model_inputs)
        finally:
            self.remove()
        return self.activations
