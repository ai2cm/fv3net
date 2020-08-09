from typing import Mapping, Set, Any, Sequence, Hashable, Union
import abc

from sklearn.utils import parallel_backend
import xarray as xr

from vcm import safe
from fv3fit.sklearn import SklearnWrapper
from fv3fit.keras import Model as KerasModel

NameDict = Mapping[Hashable, Hashable]
Model = Union[SklearnWrapper, KerasModel]


def _invert_dict(d: Mapping) -> Mapping:
    return dict(zip(d.values(), d.keys()))


class RenamingAdapter:
    """Adapter object for renaming

    Attributes:
        model: a model to wrap
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
    def input_variables(self) -> Set[str]:
        invert_rename_in = _invert_dict(self.rename_in)
        return {invert_rename_in.get(var, var) for var in self.model.input_variables}

    def predict(self, arg: xr.Dataset) -> xr.Dataset:
        input_ = self._rename_inputs(arg)
        prediction = self.model.predict(input_)
        return self._rename_outputs(prediction)


class StackingAdapter(abc.ABC):
    """Base class to wrap a model to work with unstacked inputs
    """

    def __init__(self, model: Model, sample_dims: Sequence[str]):
        self.model = model
        self.sample_dims = sample_dims

    @abc.abstractmethod
    def input_variables(self) -> Set[str]:
        pass

    @abc.abstractmethod
    def predict(self, ds: xr.Dataset) -> xr.Dataset:
        pass


class SklearnStackingAdapter(StackingAdapter):
    def __init__(self, model: SklearnWrapper, sample_dims: Sequence[str]):
        super().__init__(model, sample_dims)

    @property
    def input_variables(self) -> Set[str]:
        return set(self.model.input_vars_)

    def predict(self, ds: xr.Dataset) -> xr.Dataset:
        with parallel_backend("threading", n_jobs=1):
            stacked = ds.stack(sample=self.sample_dims)
            return self.model.predict(stacked, "sample").unstack("sample")


class KerasStackingAdapter(StackingAdapter):
    def __init__(self, model: KerasModel, sample_dims: Sequence[str]):
        super().__init__(model, sample_dims)

    @property
    def input_variables(self) -> Set[str]:
        return set(self.model.input_variables)

    def predict(self, ds: xr.Dataset) -> xr.Dataset:
        with parallel_backend("threading", n_jobs=1):
            ds_stacked = safe.stack_once(
                ds, "sample", self.sample_dims, allowed_broadcast_dims=["z"],
            )
            ds_stacked = ds_stacked.transpose("sample", "z")
            sample_multiindex = ds_stacked["sample"]
            ds_pred = (
                self.model.predict(ds_stacked)
                .assign_coords({"sample": sample_multiindex})
                .unstack()
            )
        return ds_pred
