# copied and modified from https://github.com/NVlabs/edm/blob/main/torch_utils/training_stats.py

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import torch
import torch.distributed

# ----------------------------------------------------------------------------


def all_reduce(*args, **kwargs):
    return torch.distributed.all_reduce(*args, **kwargs)


def initialize():
    # torch.multiprocessing.set_start_method('spawn')

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"

    backend = "gloo" if os.name == "nt" else "nccl"
    print(
        f"Initializing torch distributed with backend={backend} and init_method=env://"
    )
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    sync_device = torch.device("cuda") if process_count() > 1 else None
    init_multiprocessing(rank=process_index(), sync_device=sync_device)
    print0(f"Torch distributed initialized.")


# ----------------------------------------------------------------------------


def process_index():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


# ----------------------------------------------------------------------------


def process_count():
    return (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )


# ----------------------------------------------------------------------------


def local_device():
    # return the torch device for the current process
    if torch.distributed.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        local_rank = 0
    return torch.device("cuda", local_rank)


# ----------------------------------------------------------------------------


def should_stop():
    return False


# ----------------------------------------------------------------------------


def update_progress(cur, total):
    _ = cur, total


# ----------------------------------------------------------------------------


def print0(*args, **kwargs):
    if process_index() == 0:
        print(*args, **kwargs)


# ----------------------------------------------------------------------------


def _all_gather_tensor(t):
    """Gathers a single tensor from all processes."""
    if process_count() == 1:
        return t.unsqueeze(0)

    t = t.to(local_device())

    t_list = [torch.zeros_like(t) for _ in range(process_count())]
    torch.distributed.all_gather(
        t_list, t, async_op=False
    )  # caveat: `local_device` should not be used here
    return torch.stack(t_list, dim=0)


def all_gather(d):
    """Gathers a dict of tensors from all processes."""
    # if d is just a tensor, use _all_gather_tensor
    if torch.is_tensor(d):
        return _all_gather_tensor(d)

    d_gathered = {}
    for key, value in d.items():
        d_gathered[key] = _all_gather_tensor(value)
    return d_gathered


# ----------------------------------------------------------------------------


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# ----------------------------------------------------------------------------

_num_moments = 3  # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype = torch.float32  # Data type to use for initial per-tensor reduction.
_counter_dtype = torch.float64  # Data type to use for the internal counters.
_rank = 0  # Rank of the current process.
_sync_device = (
    None  # Device to use for multiprocess communication. None = single-process.
)
_sync_called = False  # Has _sync() been called yet?
_counters = (
    dict()
)  # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative = (
    dict()
)  # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor

# ----------------------------------------------------------------------------


def init_multiprocessing(rank, sync_device):
    r"""Initializes `torch_utils.training_stats` for collecting statistics
    across multiple processes.

    This function must be called after
    `torch.distributed.init_process_group()` and before `Collector.update()`.
    The call is not necessary if multi-process collection is not needed.

    Args:
        rank:           Rank of the current process.
        sync_device:    PyTorch device to use for inter-process
                        communication, or None to disable multi-process
                        collection. Typically `torch.device('cuda', rank)`.
    """
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device