from dataclasses import dataclass
from typing import Optional


@dataclass
class ReservoirTrainingConfig:
    n_burn: int
    noise: float
    seed: int = 0
    n_samples: Optional[int] = None
    n_jobs: int = -1
    subdomain_output_size: Optional[int] = None
    subdomain_overlap: Optional[int] = None
    subdomain_axis: int = 1


@dataclass
class ReadoutConfig:
    l2: float
    square_half_hidden_state: bool = False
