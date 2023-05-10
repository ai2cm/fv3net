import torch.backends
import torch
import os
import logging

logger = logging.getLogger(__name__)

if os.environ.get("TORCH_CPU_ONLY", False):
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )

logger.info("using device %s for pytorch", DEVICE.type)
