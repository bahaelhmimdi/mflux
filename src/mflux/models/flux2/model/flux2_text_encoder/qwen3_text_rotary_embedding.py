import mlx.core as mx
from mlx import nn


class Qwen3TextRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 40960,
        base: float = 1000000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Precompute inverse frequencies
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        if position_ids.ndim == 1:
            position_ids = position_ids[None, :]
            
        # Broadcast over batch and sequence lengths
        inv_freq = self.inv_freq[None, None, :]
        
        # Scale the position domain directly (Linear Scaling strategy)
        pos = position_ids.astype(mx.float32)[..., None] / self.scaling_factor
        freqs = pos * inv_freq
        
        # Duplicate for the hidden dimension mapping
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        
        return cos.astype(x.dtype), sin.astype(x.dtype)
