from typing import Hashable, MutableMapping, Protocol, Sequence
import xarray as xr
import vcm

State = MutableMapping[Hashable, xr.DataArray]


class DataTransform(Protocol):
    def forward(self, input: State) -> State:
        pass

    def backward(self, input: State) -> State:
        pass


class IdentityDataTransform(DataTransform):
    def forward(self, data: State) -> State:
        return data

    def backward(self, data: State) -> State:
        return data


class QmDataTransform(DataTransform):
    def __init__(self, q1_name="Q1", q2_name="Q2", qm_name="Qm"):
        self.q1_name = q1_name
        self.q2_name = q2_name
        self.qm_name = qm_name

    def forward(self, data: State) -> State:
        if self.qm_name not in data:
            data[self.qm_name] = vcm.moist_static_energy_tendency(
                data[self.q1_name], data[self.q2_name]
            )
        return data

    def backward(self, data: State) -> State:
        if self.q1_name not in data:
            data[self.q1_name] = vcm.temperature_tendency(
                data[self.qm_name], data[self.q2_name]
            )
        return data


def detect_transform(variables: Sequence[str]) -> DataTransform:
    if "Q2" in variables and ("Q1" in variables or "Qm" in variables):
        return QmDataTransform()
    else:
        return IdentityDataTransform()
