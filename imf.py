import flax.linen as nn
import jax
import jax.numpy as jnp

from models import imfDiT


def generate(variable, model, rng, n_sample, config, 
             num_steps, omega, t_min, t_max, sample_idx=None):
    """
    Generate samples from the model
    
    Args:
        variable: Model parameters.
        model: iMeanFlow model.
        rng: JAX random key.
        n_sample: Number of samples to generate.
        config: Configuration object.
        num_steps: Number of sampling steps.
        omega: CFG scale.
        t_min, t_max: Guidance interval.
        sample_idx: Optional index for class-conditional sampling.

    Returns:
        images: Generated images.
    """
    num_classes = config.dataset.num_classes
    img_size, img_channels = config.dataset.image_size, config.dataset.image_channels

    x_shape = (n_sample, img_size, img_size, img_channels)
    rng, rng_xt, rng_sample = jax.random.split(rng, 3)

    z_t = jax.random.normal(rng_xt, x_shape, dtype=model.dtype)

    if sample_idx is not None:
        all_y = jnp.arange(n_sample, dtype=jnp.int32)
        y = all_y + sample_idx * n_sample
        y = y % num_classes
    else:
        y = jax.random.randint(rng_sample, (n_sample,), 0, num_classes)

    t_steps = jnp.linspace(1.0, 0.0, num_steps + 1)

    def step_fn(i, x_i):
        return model.apply(variable, x_i, y, i, t_steps,
            omega, t_min, t_max, method=model.sample_one_step)

    images = jax.lax.fori_loop(0, num_steps, step_fn, z_t)

    return images


class iMeanFlow(nn.Module):
    """improved MeanFlow"""

    # Model and dataset
    model_str: str
    dtype = jnp.float32
    num_classes: int = 1000

    # Noise distribution
    P_mean: float = -0.4
    P_std: float = 1.0

    # Loss
    data_proportion: float = 0.5
    cfg_beta: float = 1.0
    class_dropout_prob: float = 0.1

    # Training dynamics
    norm_p: float = 1.0
    norm_eps: float = 0.01

    # Evaluation mode
    eval: bool = False

    def setup(self):
        """
        Setup improved MeanFlow model.
        """
        net_fn = getattr(imfDiT, self.model_str)
        self.net: imfDiT.imfDiT = net_fn(
            name="net", num_classes=self.num_classes, eval=self.eval
        )

    #######################################################
    #                       Solver                        #
    #######################################################

    def sample_one_step(self, z_t, labels, i, t_steps, omega, t_min, t_max):
        """
        Perform one sampling step given current state z_t at time step i.

        Args:
            z_t: Current noisy image at time step t.
            labels: Class labels for the batch.
            i: Current time step index.
            t_steps: Array of time steps.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
        """
        t = jnp.take(t_steps, i)
        r = jnp.take(t_steps, i + 1)
        bsz = z_t.shape[0]

        t = jnp.broadcast_to(t, (bsz,))
        r = jnp.broadcast_to(r, (bsz,))
        omega = jnp.broadcast_to(omega, (bsz,))
        t_min = jnp.broadcast_to(t_min, (bsz,))
        t_max = jnp.broadcast_to(t_max, (bsz,))

        u = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)[0]

        return z_t - jnp.einsum("n,n...->n...", t - r, u)

    #######################################################
    #                       Schedule                      #
    #######################################################

    def logit_normal_dist(self, bz):
        rnd_normal = jax.random.normal(
            self.make_rng("gen"), [bz, 1, 1, 1], dtype=self.dtype
        )
        return nn.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz):
        """
        Sample t and r from logit-normal distribution.
        """
        t = self.logit_normal_dist(bz)
        r = self.logit_normal_dist(bz)
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)

        data_size = int(bz * self.data_proportion)
        fm_mask = jnp.arange(bz) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1)
        r = jnp.where(fm_mask, t, r)

        return t, r, fm_mask

    def sample_cfg_scale(self, bz, s_max=7.0):
        """
        Sample CFG scale omega from power distribution.
        """
        ukey = self.make_rng("gen")
        u = jax.random.uniform(
            ukey, (bz, 1, 1, 1), minval=0.0, maxval=1.0, dtype=jnp.float32
        )

        if self.cfg_beta == 1.0:
            s = jnp.exp(u * jnp.log1p(jnp.asarray(s_max, jnp.float32)))
        else:
            smax = jnp.asarray(s_max, jnp.float32)
            b = jnp.asarray(self.cfg_beta, jnp.float32)

            log_base = (1.0 - b) * jnp.log1p(smax)
            log_inner = jnp.log1p(u * jnp.expm1(log_base))

            s = jnp.exp(log_inner / (1.0 - b))

        return jnp.asarray(s, jnp.float32)

    def sample_cfg_interval(self, bz, fm_mask=None):
        """
        Sample CFG interval [t_min, t_max] from uniform distribution.
        """
        rng_start, rng_end = jax.random.split(self.make_rng("gen"))

        t_min = jax.random.uniform(
            rng_start, (bz, 1, 1, 1), minval=0.0, maxval=0.5, dtype=self.dtype
        )
        t_max = jax.random.uniform(
            rng_end, (bz, 1, 1, 1), minval=0.5, maxval=1.0, dtype=self.dtype
        )

        t_min = jnp.where(fm_mask, 0.0, t_min)
        t_max = jnp.where(fm_mask, 1.0, t_max)

        return t_min, t_max

    #######################################################
    #               Training Utils & Guidance             #
    #######################################################

    def u_fn(self, x, t, h, omega, t_min, t_max, y):
        """
        Compute the predicted u component from the model.
        By default, we use auxiliary v-head to predict v component as well.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            h: Time difference t - r.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
            y: Class labels.
        Returns: (u, v)
            u: Predicted u (average velocity field).
            v: Predicted v (instantaneous velocity field).
        """
        bz = x.shape[0]
        return self.net(
            x,
            t.reshape(bz),
            h.reshape(bz),
            omega.reshape(bz),
            t_min.reshape(bz),
            t_max.reshape(bz),
            y,
        )

    def v_cond_fn(self, x, t, omega, y):
        """
        Compute the predicted v component conditioned on class labels.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            omega: CFG scale.
            y: Class labels.
        
        Returns:
            v: Predicted v component.
        """

        # Set h, t_min, t_max to dummy values for v prediction
        h = jnp.zeros_like(t)
        t_min = jnp.zeros_like(t)
        t_max = jnp.ones_like(t)

        v = self.u_fn(x, t, h, omega, t_min, t_max, y=y)[1]

        return v

    def v_fn(self, x, t, omega, y):
        """
        Compute both conditioned and unconditioned predicted v components.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            omega: CFG scale.
            y: Class labels.

        Returns:
            v_c: Predicted v component conditioned on class labels.
            v_u: Predicted v component without class labels.
        """
        bz = x.shape[0]

        # Create duplicated batch for conditioned and unconditioned predictions
        x = jnp.concatenate([x, x], axis=0)
        y_null = jnp.array([self.num_classes] * bz)
        y = jnp.concatenate([y, y_null], axis=0)
        t = jnp.concatenate([t, t], axis=0)
        w = jnp.concatenate([omega, jnp.ones_like(omega)], axis=0)

        out = self.v_cond_fn(x, t, w, y)
        v_c, v_u = jnp.split(out, 2, axis=0)

        return v_c, v_u

    def cond_drop(self, v_t, v_g, labels):
        """
        Drop class labels with a certain probability for CFG.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            v_g: Guided instantaneous velocity at time t.
            labels: Class labels for the batch.

        Returns:
            labels: Possibly dropped class labels.
            v_g: Modified guided instantaneous velocity at time t. For samples
                 with dropped labels, v_g = v_t.
        """
        bz = v_t.shape[0]

        rand_mask = (
            jax.random.uniform(self.make_rng("gen"), shape=(bz,))
            < self.class_dropout_prob
        )
        # I think the sequence before mismatch with the note above (line 265)
        # Note: Expand (B,) -> (B,1,1,1) so it can broadcast over (B,H,W,C) tensors.
        drop_mask = rand_mask[:, None, None, None]

        labels = jnp.where(
            rand_mask,
            self.num_classes,
            labels,
        )
        v_g = jnp.where(drop_mask, v_t, v_g)

        return labels, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, fm_mask, w, t_min, t_max):
        """
        Compute the guided velocity v_g using classifier-free guidance.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            z_t: Noisy image at time t.
            t, r: Two time steps.
            y: Class labels.
            fm_mask: Mask for t=r samples, i.e., flow matching samples.
            t_min, t_max: Guidance interval.
            w: CFG scale.

        Returns:
            v_g: Guided instantaneous velocity at time t, as target for training.
            v_c: Conditioned instantaneous velocity at time t, for jvp computation.
        """

        # Compute CFG target.
        v_c, v_u = self.v_fn(z_t, t, w, y=y)
        v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u)

        # Disable CFG outside [t_min, t_max] by setting w=1.0.
        # Note: in that case (1 - 1/w) becomes 0, so the (v_c - v_u) term is
        # multiplied by 0 and reusing v_u computed above is harmless.
        w = jnp.where((t >= t_min) & (t <= t_max), w, 1.0)

        v_c = self.v_cond_fn(z_t, t, w, y=y)
        v_g = v_t + (1 - 1 / w) * (v_c - v_u)

        # For flow matching samples, there is no CFG interval
        v_g = jnp.where(fm_mask, v_g_fm, v_g) 

        return v_g, v_c

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################

    def forward(self, images, labels):
        """
        Forward process of improved MeanFlow and compute loss.

        Args:
            images: A batch of images, shape (B, H, W, C).
            labels: Corresponding class labels, shape (B,).
        
        Returns:
            loss: Scalar loss value.
            dict_losses: Dictionary of individual loss components.
        """
        x = images.astype(self.dtype)
        bz = images.shape[0]

        # Instantaneous velocity computation
        t, r, fm_mask = self.sample_tr(bz)

        e = jax.random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)
        z_t = (1 - t) * x + t * e
        v_t = e - x

        # Sample CFG scale and interval
        t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
        omega = self.sample_cfg_scale(bz)

        # Compute guided velocity v_g and conditioned velocity v_c
        v_g, v_c = self.guidance_fn(
            v_t, z_t, t, r, labels, fm_mask, omega, t_min, t_max
        )

        # Cond dropout (dropout class labels)
        labels, v_g = self.cond_drop(v_t, v_g, labels)

        # Warped u-function for jvp computation
        def u_fn(z_t, t, r):
            return self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)

        dtdt = jnp.ones_like(t)
        dtdr = jnp.zeros_like(t)

        # Different from original MeanFlow, we use predicted v in the jvp
        u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)

        # Our compound function V = u + (t - r) * du/dt
        V = u + (t - r) * jax.lax.stop_gradient(du_dt)

        v_g = jax.lax.stop_gradient(v_g)

        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / jax.lax.stop_gradient(adp_wt)

        # improved MeanFlow objective is conceptually v-loss
        loss_u = jnp.sum((V - v_g) ** 2, axis=(1, 2, 3))
        loss_u = adp_wt_fn(loss_u)

        # auxiliary v-head loss
        loss_v = jnp.sum((v - v_g) ** 2, axis=(1, 2, 3))
        loss_v = adp_wt_fn(loss_v)

        loss = loss_u + loss_v
        loss = loss.mean()  # mean over batch

        dict_losses = {
            "loss": loss,
            "loss_u": jnp.mean((V - v_g) ** 2),
            "loss_v": jnp.mean((v - v_g) ** 2),
        }

        return loss, dict_losses

    def __call__(self, x, t, y):
        return self.net(x, t, t, t, t, t, y)  # initialization only
