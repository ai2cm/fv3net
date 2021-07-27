from typing import Dict, Mapping

import tensorflow as tf
import xarray as xr
from runtime.emulator.thermo import SpecificHumidityBasis

State = Mapping[str, xr.DataArray]


U = "eastward_wind"
V = "northward_wind"
T = "air_temperature"
Q = "specific_humidity"
QC = "cloud_water_mixing_ratio"
DELP = "pressure_thickness_of_atmospheric_layer"
DELZ = "vertical_thickness_of_atmospheric_layer"
DU = f"tendency_of_{U}_due_to_fv3_physics"
DV = f"tendency_of_{V}_due_to_fv3_physics"
DQ = f"tendency_of_{Q}_due_to_fv3_physics"
DT = f"tendency_of_{T}_due_to_fv3_physics"


def get_prognostic_variables():
    return [U, V, T, Q, DELP, DELZ]


def all_required_variables():
    return [DU, DV, DQ, DT, Q, U, V, T, DELZ, DELP]


def nz(x: Dict[str, tf.Tensor]):
    return batch_to_specific_humidity_basis(x).q.shape[-1]


def batch_to_specific_humidity_basis(x: Dict[str, tf.Tensor]):
    scalars = x.copy()
    kw = dict(
        u=scalars.pop(U),
        v=scalars.pop(V),
        T=scalars.pop(T),
        q=scalars.pop(Q),
        dz=scalars.pop(DELZ),
        dp=scalars.pop(DELP),
    )
    scalars_sorted = [scalars[key] for key in sorted(scalars)]
    return SpecificHumidityBasis(**kw, scalars=scalars_sorted)


def to_dict(x: SpecificHumidityBasis) -> Dict[str, tf.Tensor]:
    return {U: x.u, V: x.v, T: x.T, Q: x.q, DELZ: x.dz, DELP: x.dp}


def to_dict_no_static_vars(x: SpecificHumidityBasis) -> Dict[str, tf.Tensor]:
    return {U: x.u, V: x.v, T: x.T, Q: x.q}


def to_tensor(arr: xr.DataArray) -> tf.Variable:
    return tf.cast(tf.Variable(arr), tf.float32)


def to_tensors(ds: xr.Dataset) -> Mapping[str, xr.DataArray]:
    return {k: to_tensor(ds[k]) for k in ds}


def compute_in_out(data: Mapping[str, tf.Tensor], timestep):
    next_state = {}
    next_state[U] = data[U] + timestep * data[DU]
    next_state[V] = data[V] + timestep * data[DV]
    next_state[Q] = data[Q] + timestep * data[DQ]
    next_state[T] = data[T] + timestep * data[DT]
    next_state[DELZ] = data[DELZ]
    next_state[DELP] = data[DELP]
    return data, next_state
