import os
import torch
import logging
import torch.distributed as dist


from dataclasses import dataclass


@dataclass(frozen=True)
class DDPContext:
    """Immutable snapshot of the current process's distributed identity."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def setup_ddp(
    backend: str = "nccl",
) -> DDPContext:
    """
    initialize the distributed process group
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend=backend,
        device_id=device,
    )
    return DDPContext(
        rank=rank, local_rank=local_rank, world_size=world_size, device=device
    )


def cleanup_ddp() -> None:
    """
    destroy the distributed process group
    """
    dist.destroy_process_group()
    return


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    reduce a tensor across all processes by averaging

    ---------
    Arguments
    ---------
    tensor: torch.Tensor
        the tensor to reduce
    world_size: int
        the total number of processes

    -------
    Returns
    -------
    tensor: torch.Tensor
        the reduced tensor (averaged across all processes)
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt.item()


def is_main_process() -> bool:
    return dist.get_rank() == 0
