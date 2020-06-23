from typing import Sequence
import xarray as xr
from vcm import safe

from ._base import GeoMapper
from ..constants import DERIVATION_DIM


def get_sample_dataset(mapper):
    sample_key = list(mapper.keys())[0]
    return mapper[sample_key]


class MergeOverlappingData(GeoMapper):
    """
    Mapper for merging data sources that have overlapping data vars.
    The overlapping variable will be given an additional data_source
    coordinate to be used in reference a given dataset's variable.
    """

    def __init__(
        self,
        mappers: Sequence,
        source_names: Sequence[str],
        overlap_dim: str = DERIVATION_DIM,
        variables: Sequence[str] = None,
    ):
        if len(mappers) < 2:
            raise TypeError(
                "MergeData should be instantiated with two or more mappers."
            )
        if len(source_names) != len(mappers):
            raise ValueError("Need to provide same number of source names as mappers.")
        self._mappers = mappers
        self._source_names = source_names
        self._var_overlap = self._get_var_overlap(self._mappers)
        self._overlap_dim = overlap_dim

    def keys(self):
        mappers_keys = [mapper.keys() for mapper in self._mappers]
        return set(mappers_keys[0]).intersection(*mappers_keys)

    def __getitem__(self, key: str):
        datasets_to_merge = [mapper[key] for mapper in self._mappers]
        return self._merge_with_overlap(datasets_to_merge)

    def _merge_with_overlap(self, datasets: Sequence[xr.Dataset]) -> xr.Dataset:
        ds_nonoverlap = xr.merge([ds.drop(self._var_overlap) for ds in datasets])
        overlapping = []
        for ds, source_coord in zip(datasets, self._source_names):
            if self._overlap_dim in ds.dims:
                overlapping.append(safe.get_variables(ds, self._var_overlap))
            else:
                overlapping.append(
                    safe.get_variables(ds, self._var_overlap)
                    .expand_dims(self._overlap_dim)
                    .assign_coords({self._overlap_dim: [source_coord]})
                )
        return xr.merge(overlapping + [ds_nonoverlap])

    @staticmethod
    def _get_var_overlap(mappers_to_combine):
        ds_var_sets = []
        for mapper in mappers_to_combine:
            ds_var_sets.append(set(get_sample_dataset(mapper).data_vars))
        overlap = set()
        checked = set()
        for data_var in ds_var_sets:
            overlap |= data_var & checked
            checked |= data_var
        return overlap

    @staticmethod
    def _check_overlap_vars_dims(mappers, overlap_vars, overlap_dim):
        datasets = [get_sample_dataset(mapper) for mapper in mappers]
        overlap_var_dims = [
            safe.get_variables(ds, overlap_vars).dims for ds in datasets
        ]
        if not all(x == overlap_var_dims[0] for x in overlap_var_dims):
            vars_missing_dim = []
            # if a dataset already has overlap dim, get names of dataarrays
            # that are missing this dimension
            for ds in datasets:
                ds_overlap_vars = safe.get_variables(ds, overlap_vars)
                if overlap_dim in ds_overlap_vars.dims:
                    for var in overlap_vars:
                        if overlap_dim not in ds[var]:
                            vars_missing_dim.append(var)
            raise ValueError(
                "If overlap dimension {overlap_dim} already exists in "
                "one of the mappers it must be present for all overlapping variables. "
                f"Variables {vars_missing_dim} are missing dimension {overlap_dim}."
            )
