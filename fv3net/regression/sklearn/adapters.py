from typing import Mapping, Set, Any, Sequence

import xarray as xr

NameDict = Mapping[str, str]


def _invert_dict(d: Mapping) -> Mapping:
    return dict(zip(d.values(), d.keys()))


class RenamingAdapter:
    """Adapter object for renaming

    Attributes:
        model: a sklearn model to wrap
        rename_in: mapping from standard names to input names of model
        rename_out: mapping from standard names to the output names of model
    
    """

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
        return self._rename(ds, _invert_dict(self.rename_out))

    @property
    def variables(self) -> Set[str]:
        invert_rename_in = _invert_dict(self.rename_in)
        return {invert_rename_in.get(var, var) for var in self.model.input_vars_}

    def predict(self, arg: xr.Dataset) -> xr.Dataset:
        input_ = self._rename_inputs(arg)
        prediction = self.model.predict(input_)
        return self._rename_outputs(prediction)
