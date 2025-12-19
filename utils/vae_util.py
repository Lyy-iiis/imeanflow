import torch
from diffusers.models import AutoencoderKL
import utils.torch_util as tu


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
        self.mean, self.std = torch.split(parameters, parameters.shape[-1] // 2, dim=-1)
        self.deterministic = deterministic
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.std)


class VAEWrapper:

    def __init__(
        self, decode_batch_size, vae_type='mse', dtype=torch.float32, mix_context=None
    ):
        # init VAE
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{vae_type}",
            torch_dtype=dtype,
            local_files_only=False,
        )

        self.dtype = dtype

        # remove encoder (save memory, speed)
        del vae.encoder

        # freeze params
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()

        # move to device
        self.vae = tu.device_put(vae)
        self.mix_context = mix_context

        self.batch_size = decode_batch_size
        self.latent_size = 32

        # ---------- channels_last on model ----------
        self.vae.to(memory_format=torch.channels_last)

        # ---------- compile pure decode (best speed) ----------
        # compile only the inner decode fn
        self.compiled_decode = torch.compile(
            self.vae.decode, mode="reduce-overhead", fullgraph=True
        )
        
        self.mean = torch.tensor([0.86488, -0.27787343, 0.21616915, 0.3738409])
        self.std = torch.tensor([4.85503674, 5.31922414, 3.93725398, 3.9870003])

        # --------- final decode wrapper ----------
    def decode(self, latents):
        # scale back
        assert latents.shape[1:] == (4, self.latent_size, self.latent_size)
        latents = latents * self.std.view(-1, 1, 1).to(latents.device) + self.mean.view(-1, 1, 1).to(latents.device)

        # channels_last on inputs (important)
        latents = latents.contiguous(memory_format=torch.channels_last)

        # autocast
        if self.mix_context is not None:
            with self.mix_context:
                out = self.compiled_decode(latents)["sample"]
        else:
            out = self.compiled_decode(latents)["sample"]

        # preserve channels_last on output (optional)
        out = out.contiguous(memory_format=torch.channels_last)
        return out

    # --------- cached encode path ---------
    def cached_encode(self, cached_value):
        assert cached_value.shape[1:] == (self.latent_size, self.latent_size, 8)
        return (LatentDist(cached_value).sample() - self.mean.view(1, 1, -1)) / self.std.view(1, 1, -1)

    # --------- sliced decode ---------
    def naive_decode(self, latents):
        return self.decode(latents)
