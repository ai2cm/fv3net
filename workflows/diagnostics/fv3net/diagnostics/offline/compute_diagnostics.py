from fv3net.diagnostics._shared.registry import Registry, prepare_diag_dict
from fv3net.diagnostics._shared.constants import DiagArg
import logging
from typing import Sequence, Tuple, Dict
import xarray as xr

logger = logging.getLogger(__name__)

DOMAINS = (
    "land",
    "sea",
    "global",
    "positive_net_precipitation",
    "negative_net_precipitation",
)
SURFACE_TYPE_ENUMERATION = {0.0: "sea", 1.0: "land", 2.0: "sea"}
DERIVATION_DIM = "derivation"


def merge_diagnostics(metrics: Sequence[Tuple[str, xr.Dataset]]):
    out: Dict[str, xr.DataArray] = {}
    for (name, ds) in metrics:
        out.update(prepare_diag_dict(name, ds))
    # ignoring type error that complains if Dataset created from dict
    return xr.Dataset(out)  # type: ignore


diagnostics_registry = Registry(merge_diagnostics)


def compute_diagnostics(
    prediction: xr.Dataset,
    target: xr.Dataset,
    grid: xr.Dataset,
    delp: xr.DataArray,
    n_jobs: int = -1,
):
    diag_arg = DiagArg(prediction, target, grid, delp=delp)
    return diagnostics_registry.compute(diag_arg, n_jobs=n_jobs)
