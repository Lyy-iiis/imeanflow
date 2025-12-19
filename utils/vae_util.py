import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import torch
import torch.utils.data
from diffusers.models import FlaxAutoencoderKL
from flax import jax_utils

from utils.logging_util import log_for_0


class LatentDist(
    object
):  # https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/models/vae_flax.py#L689
    """
    Class of Gaussian distribution.

    Method:
        sample: Sample from the distribution.
    """

    def __init__(self, parameters, deterministic=False):
        """
        parameters: concatenated mean and std
        """
        # Last axis to account for channels-last
        self.mean, self.std = jnp.split(parameters, 2, axis=-1)
        self.deterministic = deterministic
        if self.deterministic:
            self.var = self.std = jnp.zeros_like(self.mean)

    def sample(self, key):
        return self.mean + self.std * jax.random.normal(key, self.mean.shape)


class LatentManager:
    def __init__(self, vae_type, decode_batch_size, input_size):
        # init VAE
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            f"pcuenq/sd-vae-ft-{vae_type}-flax"
        )
        self.vae = vae
        self.vae_params = vae_params

        self.batch_size = decode_batch_size
        self.latent_size = input_size

        # create decode function
        self.decode_fn = self.get_decode_fn()

        self.mean = jnp.array([0.86488, -0.27787343, 0.21616915, 0.3738409]).reshape(
            1, -1, 1, 1
        )
        self.std = jnp.array([4.85503674, 5.31922414, 3.93725398, 3.9870003]).reshape(
            1, -1, 1, 1
        )

    def get_decode_fn(self):

        def dist_prepare_batch_data(batch):
            # reshape (host_batch_size, 3, height, width) to
            # (local_devices, device_batch_size, height, width, 3)
            local_device_count = jax.local_device_count()

            return_dict = {}
            for k, v in batch.items():
                v = v.reshape((local_device_count, -1) + v.shape[1:])
                return_dict[k] = v
            return return_dict

        log_for_0("Compiling vae.apply...")
        now = time.time()

        z_dummy = jnp.ones(
            (
                jax.local_device_count(),
                self.batch_size,
                4,
                self.latent_size,
                self.latent_size,
            )
        )

        p_vae_variable = jax_utils.replicate({"params": self.vae_params})
        p_decode_fn = partial(self.vae.apply, method=FlaxAutoencoderKL.decode)
        p_decode_fn = jax.pmap(p_decode_fn, axis_name="batch")

        lowered = p_decode_fn.lower(p_vae_variable, z_dummy)
        compiled_decod_fn = lowered.compile()
        Bflops = (
            compiled_decod_fn.cost_analysis()[0]["flops"]
            / 1e9
            / (self.batch_size * jax.local_device_count())
        )
        log_for_0("Compiling VAE decoder done in %.2f seconds." % (time.time() - now))
        log_for_0(f"FLOPs (1e9): {Bflops}")

        def call_p_compiled_model_fn(x, p_func, var):
            x = dist_prepare_batch_data(dict(x=x))["x"]
            x = p_func(var, x)
            x = x.sample
            x = x.reshape((-1,) + x.shape[2:])
            return dict(sample=x)

        call_compiled_decod_func = partial(
            call_p_compiled_model_fn, p_func=compiled_decod_fn, var=p_vae_variable
        )

        return call_compiled_decod_func

    def cached_encode(self, cached_value, rng):
        latent = LatentDist(cached_value).sample(key=rng).transpose((0, 3, 1, 2))
        latent = (latent - self.mean) / self.std
        return latent.transpose((0, 2, 3, 1))

    def decode(self, latents):
        latents = latents * self.std + self.mean
        return self.decode_fn(latents)["sample"]


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, root, use_flip=False):
        self.root = root
        self.file_list = [
            file for file in os.listdir(self.root) if file.endswith(".pt")
        ]
        self.use_flip = use_flip

    def __len__(self):
        return len(self.file_list)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body.append(f"Use flip: {self.use_flip}")
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.file_list[idx])
        data = torch.load(file_path)
        return data["image"], data["label"]
