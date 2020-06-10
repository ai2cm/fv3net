from typing import Mapping, Set, Any, Sequence

import xarray as xr

NameDict = Mapping[str, str]


class BaseAdapter:

    variables: Sequence[str]

    def predict(self, ds: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError


class RenamingAdapter(BaseAdapter):
    """Adapter object for renaming"""

    def __init__(self, model: Any, rename_in: NameDict, rename_out: NameDict = None):
        # unforunately have to use Any with model to avoid dependency on fv3net
        # regression. We could also upgraded to python 3.8 to get access to the
        # Protocol [1] object which allows duck-typed types
        #
        # [1]: https://www.python.org/dev/peps/pep-0544/
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
        return self._rename(ds, self.rename_out)

    @property
    def variables(self) -> Set[str]:
        invert_rename_in = dict(zip(self.rename_in.values(), self.rename_in.keys()))
        return {invert_rename_in.get(var, var) for var in self.model.input_vars_}

    def predict(self, arg: xr.Dataset, sample_dim: str) -> xr.Dataset:
        input_ = self._rename_inputs(arg)
        prediction = self.model.predict(input_, sample_dim)
        return self._rename_outputs(prediction)
