import os
import time
import torch

from typing import Tuple, Callable


def gpu_timer(
    fn: Callable[[torch.Tensor], Tuple[torch.Tensor, float]],
) -> Tuple[torch.Tensor, float]:
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.time()

    result = fn()
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
    else:
        end = time.time()

    # Measure time taken to execute the function in miliseconds
    if torch.cuda.is_available():
        duration = start.elapsed_time(end)
    else:
        duration = (end - start) * 1000

    return result, duration
