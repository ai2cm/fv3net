import jsonschema
import json
import os
import xarray as xr
from typing import Mapping, Optional, Sequence

_metrics_file = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "metrics-schema.json"
)
with open(_metrics_file) as f:
    SCHEMA = json.load(f)


def validate(obj):
    return jsonschema.validate(obj, SCHEMA)


def global_average(comm, array: xr.DataArray, area: xr.DataArray) -> float:
    ans = comm.reduce((area * array).sum().item(), root=0)
    area_all = comm.reduce(area.sum().item(), root=0)
    if comm.rank == 0:
        return float(ans / area_all)
    else:
        return -1


def global_horizontal_sum(comm, array: xr.DataArray) -> xr.DataArray:
    ans = comm.reduce(array, root=0)
    if comm.rank == 0:
        return ans.sum(["x", "y"])
    else:
        return xr.DataArray([-1], dims="z")


def globally_average_2d_diagnostics(
    comm,
    diagnostics: Mapping[str, xr.DataArray],
    exclude: Optional[Sequence[str]] = None,
) -> Mapping[str, float]:
    averages = {}
    exclude = exclude or []
    for v in diagnostics:
        if (set(diagnostics[v].dims) == {"x", "y"}) and (v not in exclude):
            averages[v] = global_average(comm, diagnostics[v], diagnostics["area"])
    return averages


def globally_sum_3d_diagnostics(
    comm, diagnostics: Mapping[str, xr.DataArray], include: Sequence[str],
) -> Mapping[str, xr.DataArray]:
    sums = {}
    for v in diagnostics:
        if set(diagnostics[v].dims) == {"x", "y", "z"} and v in include:
            sums[f"{v}_global_sum"] = global_horizontal_sum(comm, diagnostics[v])
    return sums
