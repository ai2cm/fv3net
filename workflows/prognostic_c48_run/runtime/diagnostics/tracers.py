from typing import Any, Mapping

import vcm
import xarray
from runtime.names import DELP


def compute_column_integrated_tracers(
    tracer_metadata: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, xarray.DataArray],
) -> dict:
    out = {}
    for tracer in tracer_metadata:
        path = vcm.mass_integrate(state[tracer], state[DELP], dim="z").assign_attrs(
            description=f"column integrated {tracer}"
        )
        out[tracer + "_path"] = path
    return out
