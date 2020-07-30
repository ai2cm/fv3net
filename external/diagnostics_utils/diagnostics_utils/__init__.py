__version__ = "0.1.0"

from .config import VARNAMES
from .utils import (
    reduce_to_diagnostic,
    insert_total_apparent_sources,
    conditional_average,
    weighted_average,
    snap_mask_to_type,
    snap_net_precipitation_to_type,
    insert_column_integrated_vars,
)
from ._diurnal_cycle import bin_diurnal_cycle, create_diurnal_cycle_dataset

__all__ = [
    "VARNAMES",
    "reduce_to_diagnostic",
    "conditional_average",
    "weighted_average",
    "snap_mask_to_type",
    "snap_net_precipitation_to_type",
    "insert_column_integrated_vars",
    "insert_total_apparent_sources",
    "bin_diurnal_cycle",
    "create_diurnal_cycle_dataset",
]
