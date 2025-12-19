"""Learning rate utilities for training."""

import optax


def make_warmup_const_schedule(
    base_lr: float, warmup_epochs: int, steps_per_epoch: int
):
    """Alias of create_warmup_schedule; kept for clarity."""
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    if warmup_steps <= 0:
        return optax.constant_schedule(base_lr)

    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )
    const = optax.constant_schedule(base_lr)
    # After 'warmup_steps' steps, switch to constant
    return optax.join_schedules([warmup, const], boundaries=[warmup_steps])


def make_warmup_cosine_schedule(
    base_lr: float,
    warmup_epochs: int,
    steps_per_epoch: int,
    total_epochs: int,
    lr_min_factor: float = 0.0,
):
    """
    Linear warmup to base_lr, then cosine decay down to base_lr * lr_min_factor.
    Returns an optax schedule callable.
    """
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    total_steps = int(total_epochs * steps_per_epoch)
    decay_steps = max(total_steps - warmup_steps, 1)

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=base_lr * lr_min_factor,
    )


def lr_schedules(config, steps_per_epoch):
    """
    Build LR schedule from config.

    Expected config.training fields:
      - learning_rate: float
      - warmup_epochs: int (default 0)
      - num_epochs: int (for cosine)
      - lr_schedule: str in {"warmup_const", "warmup_cosine"}
      - lr_min_factor: float in [0,1], only used by warmup_cosine (default 0.0)

    Returns:
      optax.Schedule (callable step -> lr)
    """
    base_lr = float(config.training.learning_rate)
    warmup_epochs = int(config.training.get("warmup_epochs", 0))
    schedule_kind = config.training.get("lr_schedule", "warmup_const")

    if schedule_kind == "warmup_const":
        return make_warmup_const_schedule(base_lr, warmup_epochs, steps_per_epoch)

    elif schedule_kind == "warmup_cosine":
        total_epochs = int(config.training.num_epochs)
        lr_min_factor = float(config.training.get("lr_min_factor", 0.0))
        return make_warmup_cosine_schedule(
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            steps_per_epoch=steps_per_epoch,
            total_epochs=total_epochs,
            lr_min_factor=lr_min_factor,
        )

    else:
        raise ValueError(
            f"Unknown lr_schedule '{schedule_kind}'. "
            "Supported: 'warmup_const', 'warmup_cosine'."
        )
