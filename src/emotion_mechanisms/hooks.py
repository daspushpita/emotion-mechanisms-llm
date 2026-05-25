import numpy as np
import torch


def _get_layers(model):
    """Return the transformer decoder layers, handling different model architectures.

    Standard HF models (Qwen, Llama, Mistral, …): model.model.layers
    Gemma3 multimodal:                              model.model.language_model.layers
    """
    inner = model.model
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return inner.language_model.layers
    raise AttributeError(
        f"Cannot locate decoder layers on {type(inner).__name__}. "
        "Expected '.layers' or '.language_model.layers'."
    )


class ActivationExtractor:
    def __init__(self, model, layer_indices: list[int], token_position: str = "last"):
        if token_position not in {"last", "mean"}:
            raise ValueError(f"Unknown token position: {token_position}")

        self.token_position = token_position
        n_layers = len(_get_layers(model))
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
            layer = _get_layers(self.model)[layer_idx]
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
