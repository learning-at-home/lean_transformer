import functools
import os

import torch


@functools.lru_cache()
def maybe_script(fn: callable) -> callable:
    """Apply torch.jit.script to function unless one is using TPU. TPU does not support torch.jit.script."""
    if os.environ.get("TPU_NAME"):
        # this is a reserved variable that must be set to TPU address (e.g. grpc://11.22.33.44:1337) for TPU to function
        return fn
    else:
        return torch.jit.script(fn)
