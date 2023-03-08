import dataclasses
import torch


@dataclasses.dataclass
class RecurrentShape:
    """
    Attributes:
        nx: number of x grid points per tile
        ny: number of y grid points per tile
        n_state: number of state channels
        n_time: number of time steps per backpropagation window, including first point
        n_batch: number of samples per batch
    """

    nx: int
    ny: int
    n_state: int
    n_time: int
    n_batch: int

    def __post_init__(self):
        if self.nx != self.ny:
            raise TypeError("nx and ny must be equal")


def to_shape(tensor: torch.Tensor):
    return RecurrentShape(
        nx=tensor.shape[-3],
        ny=tensor.shape[-2],
        n_state=tensor.shape[-1],
        n_time=tensor.shape[-5],
        n_batch=tensor.shape[0],
    )
