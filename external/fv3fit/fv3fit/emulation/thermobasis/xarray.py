import dataclasses
import logging
from typing import MutableMapping, Hashable, Union
import xarray as xr
import tensorflow as tf
from fv3fit.emulation.thermobasis.batch import to_tensors, to_dict_no_static_vars
from fv3fit.emulation.thermobasis.emulator import Trainer, Config

State = MutableMapping[Hashable, xr.DataArray]


@dataclasses.dataclass
class XarrayEmulator:
    """Wrap an OnlineEmulator to allow for xarrray inputs and outputs
    """

    emulator: Trainer

    @property
    def input_variables(self):
        return self.emulator.input_variables

    @property
    def output_variables(self):
        return self.emulator.output_variables

    def dump(self, path: str):
        return self.emulator.dump(path)

    @classmethod
    def load(cls, path: str):
        return cls(Trainer.load(path))

    def predict(self, state: State) -> State:
        in_ = stack(state, self.input_variables)
        in_tensors = to_tensors(in_)
        x = self.emulator.batch_to_specific_humidity_basis(in_tensors)
        out = self.emulator.model(x)

        tensors = to_dict_no_static_vars(out)

        dims = ["sample", "z"]
        attrs = {"units": "no one cares"}

        return dict(
            xr.Dataset(
                {key: (dims, val, attrs) for key, val in tensors.items()},
                coords=in_.coords,
            ).unstack("sample")
        )

    def partial_fit(self, statein: State, stateout: State):

        in_tensors = _xarray_to_tensor(statein, self.input_variables)
        out_tensors = _xarray_to_tensor(stateout, self.output_variables)
        d = tf.data.Dataset.from_tensor_slices((in_tensors, out_tensors)).shuffle(
            1_000_000
        )
        self.emulator.batch_fit(d)


def _xarray_to_tensor(state, keys):
    in_ = stack(state, keys)
    return to_tensors(in_)


def stack(state: State, keys) -> xr.Dataset:
    ds = xr.Dataset({key: state[key] for key in keys})
    sample_dims = ["y", "x"]
    return ds.stack(sample=sample_dims).transpose("sample", ...)


def get_xarray_emulator(config: Union[Config, str]) -> XarrayEmulator:
    if isinstance(config, str):
        logging.info(f"Loading emulator from checkpoint {config}")
        return XarrayEmulator(Trainer.load(config))
    else:
        return XarrayEmulator(Trainer(config))
