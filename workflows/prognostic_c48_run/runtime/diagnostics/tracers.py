from typing import Any, Mapping

import vcm
import xarray
from runtime.names import DELP


def compute_column_integrated_tracers(
    wrapper: Any, state: Mapping[str, xarray.DataArray]
) -> dict:
    out = {}
    tracers = wrapper.get_tracer_metadata()
    for tracer in tracers:
        path = vcm.mass_integrate(state[tracer], state[DELP], dim="z").assign_attrs(
            description=f"column integrated {tracer}"
        )
        out[tracer + "_path"] = path
    return out
