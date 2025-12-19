import math
import jax.numpy as jnp
from flax import linen as nn

from models.torch_models import TorchLinear, TorchEmbedding


class TimestepEmbedder(nn.Module):
    """Embeds a scalar timestep (or scalar conditioning) into a vector."""

    hidden_size: int
    frequency_embedding_size: int = 256
    weight_init: str = "kaiming"
    init_constant: float = 1.0

    def setup(self):
        init_kwargs = dict(
            out_features=self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
            bias_init="zeros",
        )
        self.mlp = nn.Sequential(
            [
                TorchLinear(self.frequency_embedding_size, **init_kwargs),
                nn.silu,
                TorchLinear(self.hidden_size, **init_kwargs),
            ]
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period)
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / half
        )
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations with token dropout."""

    num_classes: int
    hidden_size: int
    weight_init: str = "kaiming"
    init_constant: float = 1.0

    def setup(self):
        self.embedding_table = TorchEmbedding(
            self.num_classes + 1,
            self.hidden_size,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
        )

    def __call__(self, labels):
        return self.embedding_table(labels)


class PatchEmbedder(nn.Module):
    """Image to Patch Embedding."""

    input_size: int
    initial_patch_size: int
    in_channels: int
    hidden_size: int
    bias: bool = True

    def setup(self):
        self.patch_size = (self.initial_patch_size, self.initial_patch_size)
        self.img_size = self.input_size
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(
            self.img_size
        )

        self.flatten = True
        self.proj = nn.Conv(
            self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(in_axis=(0, 1, 2), out_axis=-1),
            bias_init=nn.initializers.zeros,
        )

    def _init_img_size(self, img_size: int):
        img_size = (img_size, img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def __call__(self, x):
        B, H, W, C = x.shape  # (2, 32, 32, 4)
        x = self.proj(x)  # (B, H/p, W/p, hidden_c)
        x = x.reshape(B, -1, x.shape[3])  # NHWC -> NLC
        return x
