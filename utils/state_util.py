# The code is adapted and modified from https://github.com/facebookresearch/flip
# LICENSE: https://github.com/facebookresearch/flip/blob/main/LICENSE

# --------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# --------------------------------------------------------
# References:
# The code is adapted and modified from https://github.com/google-research/t5x/tree/main/t5x
# LICENSE: https://github.com/google-research/t5x/blob/2a62e14fd2806a28c8b24c7674fdd5423aa95e3d/LICENSE
# --------------------------------------------------------

"""Utilities for manipulating training state dictionaries."""

from flax import traverse_util
from utils.logging_util import log_for_0


def tensorstore_leaf(_, value):
    """Detect if the node is a serialized tensorstore spec.

    Args:
      _: The unused name of the current item.
      value: The value of the possible leaf.

    Returns:
      True if the value represents a tensorstore spec, False otherwise.
    """
    # It is a tensorstore leaf if it at least has `driver`, `kvstore` and
    # `metadata` in its keys, sometime they have additional ones like `dtype` or
    # `transform`.
    return set(value.keys()) >= {"driver", "kvstore", "metadata"}


def flatten_state_dict(state_dict, keep_empty_nodes: bool = False):
    """Flatten a dictionary until an array or tensorstore is reached.

    Args:
      state_dict: Optimizer state as nested dictionary.
      keep_empty_nodes: Whether to keep empty node, for example, empty param
        states from simple optimizers or non-touched parameter states in a
        multioptimizer.

    Returns:
      Flattened dictionary, though keeping tensor store state unflattened.
    """
    return traverse_util.flatten_dict(
        state_dict, is_leaf=tensorstore_leaf, keep_empty_nodes=keep_empty_nodes, sep="/"
    )


def print_params(params):
    """Print all parameters in the model."""
    params_flatten = flatten_state_dict(params)

    total_params = 0
    max_length = max(len(k) for k in params_flatten.keys())
    max_shape = max(len(f"{p.shape}") for p in params_flatten.values())
    max_digits = max(len(f"{p.size:,}") for p in params_flatten.values())
    log_for_0("-" * (max_length + max_digits + max_shape + 8))

    for name, param in params_flatten.items():
        layer_params = param.size
        str_layer_shape = f"{param.shape}".rjust(max_shape)
        str_layer_params = f"{layer_params:,}".rjust(max_digits)
        log_for_0(
            f" {name.ljust(max_length)} | {str_layer_shape} | {str_layer_params} "
        )
        total_params += layer_params

    log_for_0("-" * (max_length + max_digits + max_shape + 8))
    log_for_0(f"Total parameters: {total_params:,}")
