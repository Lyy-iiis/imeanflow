from functools import partial

import jax

# Register more EMA types here if needed
supported_ema = ["const"]


def const_schedule(step, ema_value):
    return ema_value


def ema_schedules(config):
    ema_type = config.training.get("ema_type", "const")
    assert ema_type in supported_ema

    if ema_type == "const":
        ema_value = config.training.get("ema_val", 0.9999)
        return partial(const_schedule, ema_value=ema_value)
    else:
        raise ValueError("Unknown EMA!")


def update_ema(ema_params, params, alpha):
    return jax.tree_map(lambda e, p: alpha * e + (1 - alpha) * p, ema_params, params)
