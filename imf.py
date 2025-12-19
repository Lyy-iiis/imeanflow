import torch
import torch.nn as nn
import torch.nn.functional as F

from models import mit

class iMeanFlow(nn.Module):
    """improved MeanFlow"""
    
    def __init__(
        self,
        model_str: str,
        dtype: torch.dtype = torch.float32,
        img_size: int = 32,
        img_channels: int = 4,
        num_classes: int = 1000,
        eval: bool = True,
    ):
        """
        Setup improved MeanFlow model.
        """
        super().__init__()
        self.model_str = model_str
        self.dtype = dtype
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes
        
        assert eval, f'The current codebase only supports inference mode'
        
        
        net_fn = getattr(mit, self.model_str)
        self.net: mit.MiT = net_fn(
            input_size=self.img_size, 
            in_channels=self.img_channels,
            num_classes=self.num_classes, 
            eval_mode=eval
        )

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
        t = t_steps[i]
        r = t_steps[i + 1]
        bsz = z_t.shape[0]

        t = t.expand(bsz)
        r = r.expand(bsz)
        omega = omega.expand(bsz)
        t_min = t_min.expand(bsz)
        t_max = t_max.expand(bsz)

        u = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)[0]

        return z_t - (t - r)[:, None, None, None] * u
    
    @torch.no_grad()
    def generate(self, n_sample, rng, num_steps, omega, t_min, t_max,
                 labels=None):
        """
        Generate samples from the model

        Args:
            n_sample: Number of samples to generate.
            num_steps: Number of sampling steps.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
            labels: Optional labels (if not provided, random labels will be used).
        """
        x_shape = (n_sample, self.img_channels, self.img_size, self.img_size)
        z_t = rng.randn(x_shape).to(self.dtype)

        if labels is not None:
            y = labels.to(z_t.device)
        else:
            y = rng.randint(0, self.num_classes, size=(n_sample,), dtype=torch.int32).to(z_t.device)
            
        t_steps = torch.linspace(1.0, 0.0, num_steps + 1).to(self.dtype).to(z_t.device)
        
        omega = torch.tensor(omega, dtype=self.dtype, device=z_t.device) if not torch.is_tensor(omega) else omega
        t_min = torch.tensor(t_min, dtype=self.dtype, device=z_t.device) if not torch.is_tensor(t_min) else t_min
        t_max = torch.tensor(t_max, dtype=self.dtype, device=z_t.device) if not torch.is_tensor(t_max) else t_max
        
        for i in range(num_steps):
            # the main inference algorithm here
            t = t_steps[i]
            r = t_steps[i + 1]
            bsz = z_t.shape[0]
            t_b = t.expand(bsz)
            r_b = r.expand(bsz)
            omega_b = omega.expand(bsz)
            t_min_b = t_min.expand(bsz)
            t_max_b = t_max.expand(bsz)
            
            u = self.u_fn(z_t, t_b, t_b - r_b, omega_b, t_min_b, t_max_b, y=y)[0]
            z_t = z_t - (t_b - r_b)[:, None, None, None] * u
            
        return z_t