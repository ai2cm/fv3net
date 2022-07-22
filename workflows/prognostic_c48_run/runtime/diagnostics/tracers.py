from typing import Mapping

import fv3gfs.wrapper
import vcm
import xarray
from runtime.names import DELP


def compute_column_integrated_tracers(state: Mapping[str, xarray.DataArray]) -> dict:
    out = {}
    tracers = fv3gfs.wrapper.get_tracer_metadata()
    for tracer in tracers:
        path = vcm.mass_integrate(state[tracer], state[DELP], dim="z").assign_attrs(
            description=f"column integrated {tracer}"
        )
        out[tracer + "_path"] = path
    return out
