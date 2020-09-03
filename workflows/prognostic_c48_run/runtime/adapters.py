from typing import Mapping, Set, Sequence, Hashable

import xarray as xr

from fv3fit._shared import Predictor

NameDict = Mapping[Hashable, Hashable]


def _invert_dict(d: Mapping) -> Mapping:
    return dict(zip(d.values(), d.keys()))


class RenamingAdapter:
    """Adapter object for renaming model variables

    Attributes:
        model: a model to rename
        rename_in: mapping from standard names to input names of model
        rename_out: mapping from standard names to the output names of model
    
    """

    def __init__(
        self, model: Predictor, rename_in: NameDict, rename_out: NameDict = None
    ):
        self.model = model
        self.rename_in = rename_in
        self.rename_out = {} if rename_out is None else rename_out

    def _rename(self, ds: xr.Dataset, rename: NameDict) -> xr.Dataset:

        all_names = set(ds.dims) & set(rename)
        rename_restricted = {key: rename[key] for key in all_names}
        redimed = ds.rename_dims(rename_restricted)

        all_names = set(ds.data_vars) & set(rename)
        rename_restricted = {key: rename[key] for key in all_names}
        return redimed.rename(rename_restricted)

    def _rename_inputs(self, ds: xr.Dataset) -> xr.Dataset:
        return self._rename(ds, self.rename_in)

    def _rename_outputs(self, ds: xr.Dataset) -> xr.Dataset:
        return self._rename(ds, _invert_dict(self.rename_out))

    @property
    def input_variables(self) -> Set[str]:
        invert_rename_in = _invert_dict(self.rename_in)
        return {invert_rename_in.get(var, var) for var in self.model.input_variables}

    def predict(self, arg: xr.Dataset) -> xr.Dataset:
        input_ = self._rename_inputs(arg)
        prediction = self.model.predict(input_)
        return self._rename_outputs(prediction)


class StackingAdapter:
    """Wrap a model to work with unstacked inputs
    
    Attributes
        model: a model to stack
        sample_dims: sample dimension name of the resulting stacked array
    
    """

    def __init__(self, model: Predictor, sample_dims: Sequence[str]):
        self.model = model
        self.sample_dims = sample_dims

    @property
    def input_variables(self) -> Set[str]:
        return set(self.model.input_variables)

    def predict(self, ds: xr.Dataset) -> xr.Dataset:
        self.model.predict_columnwise(ds, sample_dims=self.sample_dims)
