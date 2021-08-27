import dataclasses
from typing import (
    Callable,
    Hashable,
    Mapping,
    MutableMapping,
    Optional,
)
from runtime.monitor import Monitor
import xarray as xr
import tensorflow as tf
from runtime.emulator.batch import (
    to_tensors,
    to_dict_no_static_vars,
)
from runtime.emulator.emulator import OnlineEmulator, OnlineEmulatorConfig, get_emulator
from runtime.types import State, Diagnostics, Step
from runtime.names import SPHUM, DELP


@dataclasses.dataclass
class XarrayEmulator:
    """Wrap an OnlineEmulator to allow for xarrray inputs and outputs
    """

    emulator: OnlineEmulator

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
        return cls(OnlineEmulator.load(path))

    @property
    def config(self) -> OnlineEmulatorConfig:
        return self.emulator.config

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


@dataclasses.dataclass
class PrognosticAdapter:
    """Wrap a Step function with an emulator

    The wrapped function produces diagnostic outputs prefixed with
    ``self.emulator_prefix_`` and trains/applies the emulator to ``state``
    depending on the user configuration.
    """

    emulator: XarrayEmulator
    state: State
    monitor: Monitor
    emulator_prefix: str = "emulator_"

    @property
    def config(self) -> OnlineEmulatorConfig:
        return self.emulator.config

    def emulate(self, name: str, func: Step) -> Diagnostics:
        inputs = {key: self.state[key] for key in self.emulator.input_variables}

        inputs_to_save = {self.emulator_prefix + key: self.state[key] for key in inputs}
        before = self.monitor.checkpoint()
        diags = func()
        change_in_func = self.monitor.compute_change(name, before, self.state)

        if self.config.train:
            self.emulator.partial_fit(inputs, self.state)
        emulator_prediction = self.emulator.predict(inputs)

        emulator_after: MutableMapping[Hashable, xr.DataArray] = {
            DELP: before[DELP],
            **emulator_prediction,
        }

        # insert state variables not predicted by the emulator
        for v in before:
            if v not in emulator_after:
                emulator_after[v] = self.state[v]

        changes = self.monitor.compute_change("emulator", before, emulator_after)

        if self.config.online:
            update_state_with_emulator(
                self.state,
                emulator_prediction,
                ignore_humidity_below=self.config.ignore_humidity_below,
            )
        return {**diags, **inputs_to_save, **changes, **change_in_func}

    def __call__(self, name: str, func: Step) -> Step:
        def step() -> Diagnostics:
            return self.emulate(name, func)

        # functools.wraps modifies the type and breaks mypy type checking
        step.__name__ = func.__name__

        return step


def _xarray_to_tensor(state, keys):
    in_ = stack(state, keys)
    return to_tensors(in_)


def stack(state: State, keys) -> xr.Dataset:
    ds = xr.Dataset({key: state[key] for key in keys})
    sample_dims = ["y", "x"]
    return ds.stack(sample=sample_dims).transpose("sample", ...)


def get_xarray_emulator(config: OnlineEmulatorConfig):
    return XarrayEmulator(get_emulator(config))


def _update_state_with_emulator(
    state: MutableMapping[Hashable, xr.DataArray],
    src: Mapping[Hashable, xr.DataArray],
    from_orig: Callable[[Hashable, xr.DataArray], xr.DataArray],
) -> None:
    """
    Args:
        state: the mutable state object
        src: updates to put into state
        from_orig: a function returning a mask. Where this mask is True, the
            original state array will be used.

    """
    for key in src:
        arr = state[key]
        mask = from_orig(key, arr)
        state[key] = arr.where(mask, src[key].variable)


@dataclasses.dataclass
class from_orig:
    ignore_humidity_below: Optional[int] = None

    def __call__(self, name: Hashable, arr: xr.DataArray) -> xr.DataArray:
        if name == SPHUM:
            if self.ignore_humidity_below is not None:
                return arr.z > self.ignore_humidity_below
            else:
                return xr.DataArray(False)
        else:
            return xr.DataArray(True)


def update_state_with_emulator(
    state: MutableMapping[Hashable, xr.DataArray],
    src: Mapping[Hashable, xr.DataArray],
    ignore_humidity_below: Optional[int] = None,
) -> None:
    return _update_state_with_emulator(state, src, from_orig(ignore_humidity_below))
