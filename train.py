"""
Training and evaluation for improved MeanFlow.
"""

import jax
import jax.numpy as jnp
import ml_collections
from flax import jax_utils
from jax import lax, random
from flax.training import checkpoints
from functools import partial
from optax._src.alias import *

from imf import iMeanFlow, generate

import utils.input_pipeline as input_pipeline
from utils.ckpt_util import save_checkpoint
from utils.ema_util import ema_schedules, update_ema
from utils.logging_util import MetricsTracker, Timer, log_for_0, Writer
from utils.vae_util import LatentManager
from utils.vis_util import make_grid_visualization
from utils.lr_utils import lr_schedules
from utils.sample_util import get_fid_evaluator, run_p_sample_step
from utils.trainstate_util import create_train_state, TrainState

#######################################################
#                    Train Step                       #
#######################################################


def compute_metrics(dict_losses):
    metrics = {k: jnp.mean(v) for k, v in dict_losses.items()}
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def train_step_with_vae(state, batch, rng_init, ema_fn, lr_fn, latent_manager):
    """
    Perform a single training step.
    """
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))

    images = batch["image"]  # [B, H, W, C]
    labels = batch["label"]

    rng_base, rng_vae = random.split(rng_base)
    images = latent_manager.cached_encode(images, rng_vae)  # [B, H, W, C] sample latent

    def loss_fn(params):
        """loss function used for training."""
        outputs = state.apply_fn(
            {"params": params},
            images=images,
            labels=labels,
            rngs=dict(
                gen=rng_base,
            ),
        )
        return outputs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)

    lr_value = lr_fn(state.step)
    ema_value = ema_fn(state.step)

    dict_losses = aux[1]
    metrics = compute_metrics(dict_losses)
    metrics["lr"] = lr_value
    new_ema = update_ema(new_state.ema_params, new_state.params, ema_value)
    new_state = new_state.replace(ema_params=new_ema)

    return new_state, metrics


#######################################################
#               Sampling and Metrics                  #
#######################################################


def sample_step(variable, sample_idx, model, rng_init, device_batch_size, 
                config, num_steps, omega, t_min, t_max):
    """
    sample_idx: each random sampled image corrresponds to a seed
    """
    rng_sample = random.fold_in(rng_init, sample_idx)  # fold

    images = generate(variable, model, rng_sample, device_batch_size,
                      config, num_steps, omega, t_min, t_max, sample_idx=sample_idx)

    images = images.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    return images


#######################################################
#                       Main                          #
#######################################################


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    ########### Initialize ###########
    writer = Writer(config, workdir)

    rng = random.key(config.training.seed)
    image_size = config.dataset.image_size
    device_bsz = config.fid.device_batch_size

    log_for_0("config.training.batch_size: {}".format(config.training.batch_size))
    local_batch_size = config.training.batch_size // jax.process_count()
    log_for_0("local_batch_size: {}".format(local_batch_size))
    log_for_0("jax.local_device_count: {}".format(jax.local_device_count()))

    ########### Create DataLoaders ###########
    train_loader, steps_per_epoch = input_pipeline.create_latent_split(
        config.dataset,
        local_batch_size,
        split="train",
    )
    log_for_0("Steps per Epoch: {}".format(steps_per_epoch))

    ########### Create Model ###########
    model_config = config.model.to_dict()
    model = iMeanFlow(**model_config)

    ########### Create Train State ###########
    lr_fn = lr_schedules(config, steps_per_epoch)
    ema_fn = ema_schedules(config)
    state = create_train_state(rng, config, model, image_size, lr_fn)

    if config.load_from != "":
        state = checkpoints.restore_checkpoint(config.load_from, state)

    step = int(state.step)
    epoch_offset = step // steps_per_epoch

    state = jax_utils.replicate(state)

    ########### Create Latent Manager ###########

    latent_manager = LatentManager(config.dataset.vae, device_bsz, image_size)

    ########### Create train and sample pmap ###########

    p_train_step = jax.pmap(
        partial(
            train_step_with_vae,
            rng_init=rng,
            ema_fn=ema_fn,
            lr_fn=lr_fn,
            latent_manager=latent_manager,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )

    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=random.PRNGKey(99),
            config=config,
            device_batch_size=device_bsz,
            num_steps=config.sampling.num_steps,
        ),
        axis_name="batch",
    )

    sample_kwargs = {
        "omega": config.sampling.omega,
        "t_min": config.sampling.t_min,
        "t_max": config.sampling.t_max,
    }
    sample_kwargs = jax_utils.replicate(sample_kwargs)

    fid_evaluator = get_fid_evaluator(config, writer, latent_manager)

    ########### Training Loop ###########
    metrics_tracker = MetricsTracker()
    log_for_0("Initial compilation, this might take some minutes...")
    vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(
        jax.local_device_count()
    )

    for epoch in range(epoch_offset, config.training.num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        log_for_0("epoch {}...".format(epoch))

        ########### Sampling ###########
        if (epoch + 1) % config.training.sample_per_epoch == 0:
            log_for_0(f"Samples at epoch {epoch}...")
            vis_sample = run_p_sample_step(p_sample_step, state, vis_sample_idx, 
                                           latent_manager, True, **sample_kwargs)
            vis_sample = make_grid_visualization(vis_sample, grid=4)
            vis_sample = jax.device_get(vis_sample)[0]
            writer.write_images(step + 1, {"vis_sample": vis_sample})

        ########### Train ###########
        timer = Timer()
        log_for_0("epoch {}...".format(epoch))
        timer.reset()
        for n_batch, batch in enumerate(train_loader):
            step = epoch * steps_per_epoch + n_batch

            batch = input_pipeline.prepare_batch_data(batch)
            state, metrics = p_train_step(state, batch)

            if epoch == epoch_offset and n_batch == 0:
                log_for_0("Initial compilation completed. Reset timer.")
                compilation_time = timer.elapse_with_reset()
                log_for_0("p_train_step compiled in {:.2f}s".format(compilation_time))

            ########### Metrics ###########
            metrics_tracker.update(metrics)  # stream one step in
            if (step + 1) % config.training.log_per_step == 0:
                summary = metrics_tracker.finalize()
                summary["steps_per_second"] = (
                    config.training.log_per_step / timer.elapse_with_reset()
                )
                writer.write_scalars(step + 1, summary)

        ########### Save Checkpoint ###########
        if (epoch + 1) % config.training.checkpoint_per_epoch == 0 \
            or (epoch + 1) == config.training.num_epochs:
            save_checkpoint(state, workdir)

        ########### FID ###########
        if (epoch + 1) % config.training.fid_per_epoch == 0 \
            or (epoch + 1) == config.training.num_epochs:
            fid_evaluator(state, p_sample_step, step, **sample_kwargs)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state

########################################################
#                    Evaluation                        #
########################################################

def just_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:

    assert config.eval_only, "config.eval_only must be True for just_evaluate"
    assert (
        config.load_from != ""
    ), "config.load_from must be specified for just_evaluate"

    ########### Initialize ###########
    writer = Writer(config, workdir)

    rng = random.key(0)
    image_size = config.dataset.image_size
    device_bsz = config.fid.device_batch_size
    lr_fn = lr_schedules(config, 1000)  # dummy steps_per_epoch

    ########### Create Model ###########
    model_config = config.model.to_dict()
    model = iMeanFlow(**model_config, eval=True)

    ########### Create Train State ###########
    state = create_train_state(rng, config, model, image_size, lr_fn)
    state = checkpoints.restore_checkpoint(config.load_from, state)
    step = int(state.step)
    state = jax_utils.replicate(state)

    ########### Create Latent Manager ###########

    latent_manager = LatentManager(config.dataset.vae, device_bsz, image_size)

    ########### Create sample pmap ###########

    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=random.PRNGKey(99),
            config=config,
            device_batch_size=device_bsz,
            num_steps=config.sampling.num_steps,
        ),
        axis_name="batch",
    )

    fid_evaluator = get_fid_evaluator(config, writer, latent_manager)

    ############ Evaluate over CFG configs ###########
    best_fid = float("inf")
    best_is = float("-inf")
    best_config = None
    for interval in config.sampling.interval:
        t_min, t_max = interval
        for omega in config.sampling.omegas:
            kwargs = {"omega": omega, "t_min": t_min, "t_max": t_max}
            kwargs = jax_utils.replicate(kwargs)
            fid, is_score = fid_evaluator(state, p_sample_step, step, True, **kwargs)

            if fid < best_fid:
                best_fid, best_is, best_config = fid, is_score, (omega, t_min, t_max)

    log_for_0(
        f"Best FID achieved: {best_fid:.2f}, \n"
        f"IS achieved: {best_is:.2f}, \n"
        f"omega: {best_config[0]:.2f}, t_min: {best_config[1]:.2f}, t_max: {best_config[2]:.2f}"
    )

    return state
