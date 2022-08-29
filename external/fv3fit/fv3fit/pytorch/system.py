import torch.backends
import torch

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
