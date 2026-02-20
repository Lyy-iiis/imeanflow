import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedder import PatchEmbedder, TimestepEmbedder, LabelEmbedder
from models.torch_models import TorchLinear, RMSNorm, SwiGLUMlp


def unsqueeze(t, dim):
    """Adds a new axis to a tensor at the given position."""
    return t.unsqueeze(dim)


#################################################################################
#                   Modern Transformer Components with Vec Gates               #
#################################################################################


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE and QK RMS norm."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        init_kwargs = dict(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

        self.q_proj = TorchLinear(**init_kwargs)
        self.k_proj = TorchLinear(**init_kwargs)
        self.v_proj = TorchLinear(**init_kwargs)
        self.out_proj = TorchLinear(**init_kwargs)

        self.head_dim = self.hidden_size // self.num_heads

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_freqs):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)

        # manually implement attention to match JAX implementation
        query = q / math.sqrt(self.head_dim)
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, k)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)

        attn = attn.reshape(batch, seq_len, self.hidden_size)

        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Transformer block with zero-initialized vector gates on residuals."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        self.norm1 = RMSNorm(self.hidden_size)
        self.attn = RoPEAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )
        self.norm2 = RMSNorm(self.hidden_size)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.mlp = SwiGLUMlp(
            self.hidden_size,
            mlp_hidden_dim,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )

        self.attn_scale = nn.Parameter(torch.zeros(self.hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, x, rope_freqs):
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """Final projection layer with RMSNorm and zero init weights."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.norm = RMSNorm(self.hidden_size)
        self.linear = TorchLinear(
            self.hidden_size,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )

    def __call__(self, x):
        return self.linear(self.norm(x))


#################################################################################
#                improved MeanFlow DiT with In-context Conditioning             #
#################################################################################


class imfDiT(nn.Module):
    """
    MeanFlow improved Transformer (imfDiT).
    A shared backbone processes the first (depth - aux_head_depth) layers.
    Two heads of equal depth (aux_head_depth) branch off afterwards.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 8 / 3,
        num_classes: int = 1000,
        aux_head_depth: int = 8,
        num_class_tokens: int = 8,
        num_time_tokens: int = 4,
        num_cfg_tokens: int = 4,
        num_interval_tokens: int = 2,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
        eval_mode: bool = False,
    ):
        """
        Set up the imfDiT model components.
         - Patch embedder for input images.
         - Embedders for time, omega, cfg intervals, and class labels.
         - Learnable tokens for conditioning.
         - Transformer blocks with shared backbone and dual heads.
         - Final projection layers for u and v outputs.
        """
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        self.aux_head_depth = aux_head_depth

        self.num_class_tokens = num_class_tokens
        self.num_time_tokens = num_time_tokens
        self.num_cfg_tokens = num_cfg_tokens
        self.num_interval_tokens = num_interval_tokens

        self.token_init_constant = token_init_constant
        self.embedding_init_constant = embedding_init_constant
        self.weight_init_constant = weight_init_constant

        self.eval_mode = eval_mode

        self.out_channels = self.in_channels

        self.x_embedder = PatchEmbedder(
            self.input_size,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            bias=True,
        )

        embed_kwargs = dict(
            hidden_size=self.hidden_size,
            weight_init="scaled_variance",
            init_constant=self.embedding_init_constant,
        )

        self.h_embedder = TimestepEmbedder(**embed_kwargs)
        self.omega_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_start_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_end_embedder = TimestepEmbedder(**embed_kwargs)
        self.y_embedder = LabelEmbedder(self.num_classes, **embed_kwargs)

        token_initializer = partial(
            nn.init.normal_, std=self.token_init_constant / math.sqrt(self.hidden_size)
        )
        self.time_tokens = nn.Parameter(
            token_initializer(torch.empty(self.num_time_tokens, self.hidden_size))
        )
        self.class_tokens = nn.Parameter(
            token_initializer(torch.empty(self.num_class_tokens, self.hidden_size))
        )
        self.omega_tokens = nn.Parameter(
            token_initializer(torch.empty(self.num_cfg_tokens, self.hidden_size))
        )
        self.t_min_tokens = nn.Parameter(
            token_initializer(torch.empty(self.num_interval_tokens, self.hidden_size))
        )
        self.t_max_tokens = nn.Parameter(
            token_initializer(torch.empty(self.num_interval_tokens, self.hidden_size))
        )

        total_tokens = (
            self.x_embedder.num_patches
            + self.num_class_tokens
            + self.num_cfg_tokens
            + 2 * self.num_interval_tokens
            + self.num_time_tokens
        )
        self.prefix_tokens = (
            self.num_class_tokens
            + self.num_cfg_tokens
            + 2 * self.num_interval_tokens
            + self.num_time_tokens
        )
        self.head_dim = self.hidden_size // self.num_heads
        self.rope_freqs = precompute_rope_freqs(self.head_dim, total_tokens)

        head_depth = self.aux_head_depth
        shared_depth = self.depth - head_depth

        block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=self.weight_init_constant,
        )

        self.shared_blocks = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(shared_depth)]
        )
        self.u_heads = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(head_depth)]
        )

        # We don't need the v heads during evaluation
        self.v_heads = nn.ModuleList(
            [
                TransformerBlock(**block_kwargs)
                for _ in range(head_depth if not self.eval_mode else 0)
            ]
        )

        self.u_final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        )
        self.v_final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        )

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        images = x.reshape((x.shape[0], c, h * p, w * p))
        return images

    def _build_sequence(self, x, h, w, t_min, t_max, y):
        """
        Build the input token sequence for the transformer.
        1. Embed the input image patches.
        2. Embed the conditioning information (time, omega, cfg, class labels).
        3. Prepend the conditioning tokens to the patch embeddings.

        Args:
            x: Input images
            h: timestep
            w: CFG scale
            t_min, t_max: CFG interval
            y: Class labels

        Returns:
            seq: Token sequence for the transformer
        """

        x_embed = self.x_embedder(x)
        h_embed = self.h_embedder(h)
        omega_embed = self.omega_embedder(1 - 1 / w)
        t_min_embed = self.cfg_t_start_embedder(t_min)
        t_max_embed = self.cfg_t_end_embedder(t_max)
        y_embed = self.y_embedder(y)

        time_tokens = self.time_tokens + unsqueeze(h_embed, 1)
        omega_tokens = self.omega_tokens + unsqueeze(omega_embed, 1)
        t_min_tokens = self.t_min_tokens + unsqueeze(t_min_embed, 1)
        t_max_tokens = self.t_max_tokens + unsqueeze(t_max_embed, 1)
        class_tokens = self.class_tokens + unsqueeze(y_embed, 1)

        seq = torch.cat(
            [
                class_tokens,
                omega_tokens,
                t_min_tokens,
                t_max_tokens,
                time_tokens,
                x_embed,
            ],
            axis=1,
        )

        return seq

    def forward(self, x, t, h, w, t_min, t_max, y):
        """
        Forward pass of the imfDiT model.
        Returns the predicted u and v components.

        Args:
            x: Input images
            t, h: time steps
            w: CFG scale
            t_min, t_max: CFG interval
            y: Class labels

        Returns:
            u: Average velocity field
            v: Instantaneous velocity field
        """

        # We don't explicitly condition on time t, only on h = t - r
        # following https://arxiv.org/abs/2502.13129
        seq = self._build_sequence(x, h, w, t_min, t_max, y)

        for block in self.shared_blocks:
            seq = block(seq, self.rope_freqs)

        u_seq = v_seq = seq
        for block in self.u_heads:
            u_seq = block(u_seq, self.rope_freqs)

        for block in self.v_heads:
            v_seq = block(v_seq, self.rope_freqs)

        u_tokens = u_seq[:, self.prefix_tokens :]
        v_tokens = v_seq[:, self.prefix_tokens :]

        u = self.unpatchify(self.u_final_layer(u_tokens))
        v = self.unpatchify(self.v_final_layer(v_tokens))

        return u, v


#################################################################################
#                           Rotary Position Helpers                             #
#################################################################################


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs_cis = torch.outer(positions, freqs)
    real = torch.cos(freqs_cis)
    imag = torch.sin(freqs_cis)
    return torch.complex(real, imag)


def apply_rotary_pos_emb(x, freqs_cis):
    x_complex = x.to(torch.float32).view(torch.complex64)
    x_complex = x_complex.reshape(x.shape[:-1] + (-1,))
    freqs_cis = unsqueeze(unsqueeze(freqs_cis, 0), 2)
    x_rotated = x_complex * freqs_cis.to(x.device)
    x_out = x_rotated.to(x_complex.dtype).view(x.dtype)
    return x_out.reshape(x.shape)


#################################################################################
#                                iMF DiT Configs                                #
#################################################################################


imfDiT_B_2 = partial(
    imfDiT,
    depth=12,
    hidden_size=768,
    patch_size=2,
    num_heads=12,
    aux_head_depth=8,
)

imfDiT_M_2 = partial(
    imfDiT,
    depth=24,
    hidden_size=768,
    patch_size=2,
    num_heads=12,
    aux_head_depth=8,
)

imfDiT_L_2 = partial(
    imfDiT,
    depth=32,
    hidden_size=1024,
    patch_size=2,
    num_heads=16,
    aux_head_depth=8,
)

imfDiT_XL_2 = partial(
    imfDiT,
    depth=48,
    hidden_size=1024,
    patch_size=2,
    num_heads=16,
    aux_head_depth=8,
)
