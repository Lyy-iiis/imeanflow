# this is an interface conversion from JAX to PyTorch
import torch
import numpy as np
import utils.torch_dist_util as dist


def seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_get(arr):
    # get the arr and convert to numpy
    assert isinstance(arr, torch.Tensor), f"Expected torch.Tensor but got {type(arr)}"
    return arr.cpu().to(torch.float32).numpy()


def device_put(arr, non_blocking=False):
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    assert hasattr(arr, "to"), f"Expected tensor/module but got {type(arr)}"
    return arr.to(dist.local_device(), non_blocking=non_blocking)


# copied from EDM
class BatchGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.device = device
        self.generators = [
            torch.Generator("cpu").manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randn(size[1:], generator=gen, **kwargs).to(self.device)
                for gen in self.generators
            ]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs).to(
                    self.device
                )
                for gen in self.generators
            ]
        )


def tree_map(f, tree):
    if isinstance(tree, dict):
        return {k: tree_map(f, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(f, v) for v in tree)
    elif isinstance(tree, (int, float)):
        return tree
    else:
        return f(tree)