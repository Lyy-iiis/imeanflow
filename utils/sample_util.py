import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from functools import partial
from utils import fid_util
from utils.logging_util import log_for_0


def run_p_sample_step(
    p_sample_step, state, sample_idx, latent_manager, ema=True, **kwargs
):
    """
    Run one p_sample_step to get samples from the model.
    """
    params = state.ema_params if ema else state.params

    variable = {"params": params}
    latent = p_sample_step(variable, sample_idx=sample_idx, **kwargs)
    latent = latent.reshape(-1, *latent.shape[2:])

    samples = latent_manager.decode(latent)
    assert not jnp.any(
        jnp.isnan(samples)
    ), f"There is nan in decoded samples! Latent range: {latent.min()}, {latent.max()}. nan in latent: {jnp.any(jnp.isnan(latent))}"

    samples = samples.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    samples = 127.5 * samples + 128.0
    samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)

    jax.random.normal(random.key(0), ()).block_until_ready()  # dist sync
    return samples


def generate_fid_samples(
    state, config, p_sample_step, run_p_sample_step, ema=True, **kwargs
):
    """
    Generate samples for FID evaluation.
    """
    num_steps = np.ceil(
        config.fid.num_samples / config.fid.device_batch_size / jax.device_count()
    ).astype(int)

    samples_all = []

    log_for_0("Note: the first sample may be significant slower")
    for step in range(num_steps):
        sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(
            jax.local_device_count()
        )
        sample_idx = jax.device_count() * step + sample_idx
        log_for_0(f"Sampling step {step} / {num_steps}...")
        samples = run_p_sample_step(
            p_sample_step, state, sample_idx=sample_idx, ema=ema, **kwargs
        )
        samples = jax.device_get(samples)
        samples_all.append(samples)

    samples_all = np.concatenate(samples_all, axis=0)

    return samples_all


def get_fid_evaluator(config, writer, latent_manager):
    """
    Create FID evaluator function.
    """
    inception_net = fid_util.build_jax_inception()
    stats_ref = fid_util.get_reference(config.fid.cache_ref)
    run_p_sample_step_inner = partial(run_p_sample_step, latent_manager=latent_manager)

    def _evaluate_one_mode(state, p_sample_step, ema, **kwargs):
        # 1) Sampling
        samples_all = generate_fid_samples(
            state, config, p_sample_step, run_p_sample_step_inner, ema, **kwargs
        )
        # 2) Stats
        stats = fid_util.compute_stats(samples_all, inception_net)
        # 3) Metrics
        metric = {}

        mode_str = "ema" if ema else "online"

        omega = kwargs.get("omega", None)[0]
        t_min = kwargs.get("t_min", None)[0]
        t_max = kwargs.get("t_max", None)[0]
        log_for_0(
            f"Computing FID and Inception Score at omega={omega:.2f}, t_min={t_min:.2f}, t_max={t_max:.2f}, mode={mode_str}..."
        )
        descriptor = f"omega_{omega:.2f}_tmin_{t_min:.2f}_tmax_{t_max:.2f}_{mode_str}"

        fid = fid_util.compute_fid(
            stats_ref["mu"], stats["mu"], stats_ref["sigma"], stats["sigma"]
        )
        is_score, _ = fid_util.compute_inception_score(stats["logits"])

        metric[f"FID_{descriptor}"] = fid
        metric[f"IS_{descriptor}"] = is_score

        return metric, fid, is_score

    def evaluator(state, p_sample_step, step, ema_only=False, **kwargs):
        metric_dict = {}
        metric, fid, is_score = _evaluate_one_mode(state, p_sample_step, True, **kwargs)
        metric_dict.update(metric)
        if not ema_only:
            metric, _, _ = _evaluate_one_mode(state, p_sample_step, False, **kwargs)
            metric_dict.update(metric)

        writer.write_scalars(step + 1, metric_dict)
        return fid, is_score

    return evaluator
