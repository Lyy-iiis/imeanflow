import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.torch_models import TorchLinear, TorchEmbedding


class TimestepEmbedder(nn.Module):
    """Embeds a scalar timestep (or scalar conditioning) into a vector."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256, weight_init="kaiming", init_constant=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.weight_init = weight_init
        self.init_constant = init_constant

        init_kwargs = dict(
            out_features=self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
            bias_init="zeros",
        )
        self.mlp = nn.Sequential(
            TorchLinear(self.frequency_embedding_size, **init_kwargs),
            nn.SiLU(),
            TorchLinear(self.hidden_size, **init_kwargs),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        args = t[:, None].to(torch.float32) * freqs[None].to(t.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], axis=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations with token dropout."""

    def __init__(self, num_classes, hidden_size, weight_init="kaiming", init_constant=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.weight_init = weight_init
        self.init_constant = init_constant

        self.embedding_table = TorchEmbedding(
            self.num_classes + 1,
            self.hidden_size,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
        )

    def forward(self, labels):
        return self.embedding_table(labels)


class PatchEmbedder(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, input_size, initial_patch_size, in_channels, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.initial_patch_size = initial_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.bias = bias

        self.patch_size = (self.initial_patch_size, self.initial_patch_size)
        self.img_size = self.input_size
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(
            self.img_size
        )

        self.flatten = True
        self.proj = nn.Conv2d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=self.bias,
        )
        
        # init proj weights like nn.Linear, instead of nn.Conv2d
        kh = kw = self.patch_size[0]
        fan_in = kh * kw * self.in_channels
        fan_out = self.hidden_size
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.proj.weight, -limit, limit)
        if self.bias:
            nn.init.zeros_(self.proj.bias)

    def _init_img_size(self, img_size: int):
        img_size = (img_size, img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def forward(self, x):
        B, C, H, W = x.shape  # (2, 32, 32, 4)
        assert H == W, f'{x.shape}'
        x = self.proj(x)  # (B, H/p, W/p, hidden_c)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, x.shape[1])  # NCHW -> NLC
        return x
