from typing import Mapping, Set
from fv3net.regression.sklearn import SklearnWrapper

import xarray as xr


class RenamingAdapter:
    """Adapter object for renaming"""

    def __init__(self, model: SklearnWrapper, rename_in, rename_out=None):
        self.model = model
        self.rename_in = rename_in
        if rename_out is None:
            self.rename_out = dict(zip(rename_in.values(), rename_in.keys()))
        else:
            self.rename_out = rename_out

    @staticmethod
    def _rename(ds: xr.Dataset, rename: Mapping[str, str]) -> xr.Dataset:
        all_names = (set(ds.data_vars) | set(ds.dims)) & set(rename)
        rename_restricted = {key: rename[key] for key in all_names}
        return ds.rename(rename_restricted)

    def _rename_inputs(self, ds):
        # TODO typehints
        return self._rename(ds, self.rename_in)

    def _rename_outputs(self, ds):
        return self._rename(ds, self.rename_out)

    @property
    def variables(self) -> Set[str]:
        return {self.rename_out.get(var, var) for var in self.model.input_vars_}

    def predict(self, arg: xr.Dataset, sample_dim: str) -> xr.Dataset:
        input_ = self._rename_inputs(arg)
        prediction = self.model.predict(input_, sample_dim)
        return self._rename_outputs(prediction)
