from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class TorchLinear(nn.Module):
    """A linear layer similar to torch.nn.Linear."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_init="scaled_variance",
        init_constant=1.0,
        bias_init="zeros",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_init = weight_init
        self.init_constant = init_constant
        self.bias_init = bias_init

        if self.weight_init == "scaled_variance":
            std = self.init_constant / sqrt(self.in_features)
            weight_initializer = partial(nn.init.normal_, std=std)
        elif self.weight_init == "zeros":
            weight_initializer = nn.init.zeros_
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        if self.bias_init == "zeros":
            bias_initializer = nn.init.zeros_
        else:
            raise ValueError(f"Invalid bias_init: {self.bias_init}")

        self._flax_linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
        )
        weight_initializer(self._flax_linear.weight)
        if self.bias:
            bias_initializer(self._flax_linear.bias)

    def forward(self, x):
        return self._flax_linear(x)


class TorchEmbedding(nn.Module):
    """A embedding layer similar to torch.nn.Embedding."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        weight_init="scaled_variance",
        init_constant=1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_init = weight_init
        self.init_constant = init_constant

        if self.weight_init is None:
            std = 0.02
        elif self.weight_init == "scaled_variance":
            std = self.init_constant / sqrt(self.embedding_dim)
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        self._flax_embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
        )
        nn.init.normal_(self._flax_embedding.weight, std=std)

    def forward(self, x):
        return self._flax_embedding(x)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(self.dim))

    def _norm(self, x):
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps)

    def forward(self, x):
        output = self._norm(x).to(x.dtype)
        return output * self.weight


class SwiGLUMlp(nn.Module):
    """Swish-Gated Linear Unit MLP."""

    def __init__(
        self,
        in_features,
        hidden_features,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        init_kwargs = dict(
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

        self.w1 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w3 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w2 = TorchLinear(self.hidden_features, self.in_features, **init_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
