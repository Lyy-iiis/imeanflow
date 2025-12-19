import jax
import jax.numpy as jnp
from jax import random
from typing import Any
from functools import partial
from copy import deepcopy
import ml_collections
import optax

from flax.training import train_state
from utils.logging_util import log_for_0
from utils.state_util import print_params
from utils.ema_util import update_ema


#######################################################
#                    Initialize                       #
#######################################################


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 4)
    x = jnp.ones(input_shape)
    t = jnp.ones((1,), dtype=int)
    y = jnp.ones((1,), dtype=int)

    @jax.jit
    def init(*args):
        return model.init(*args)

    log_for_0("Initializing params...")
    variables = init({"params": key}, x, t, y)
    log_for_0("Initializing params done.")

    param_count = sum(x.size for x in jax.tree_leaves(variables["params"]))
    log_for_0("Total trainable parameters: " + str(param_count))
    return variables, variables["params"]


#######################################################
#                     Train State                     #
#######################################################


class TrainState(train_state.TrainState):
    ema_params: Any


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, lr_fn
):
    """
    Create initial training state.
    ---
    apply_fn: output a dict, with key 'loss', 'mse'
    """

    rng, rng_init = random.split(rng)

    _, params = initialized(rng_init, image_size, model)
    ema_params = deepcopy(params)
    ema_params = update_ema(ema_params, params, 0)
    print_params(params["net"])

    tx = optax.adamw(
        learning_rate=lr_fn,
        weight_decay=0,
        b2=config.training.adam_b2,
    )
    state = TrainState.create(
        apply_fn=partial(model.apply, method=model.forward),
        params=params,
        ema_params=ema_params,
        tx=tx,
    )
    return state
