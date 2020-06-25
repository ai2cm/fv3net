from typing import Sequence
import xarray as xr
from vcm import safe

from ._base import GeoMapper
from ..constants import DERIVATION_DIM
from .._utils import get_sample_dataset


class MergeOverlappingData(GeoMapper):
    def __init__(
        self,
        mapper_left: GeoMapper,
        mapper_right: GeoMapper,
        source_name_left: str = None,
        source_name_right: str = None,
        overlap_dim: str = DERIVATION_DIM,
    ):
        """ Initialize mapper for merging data sources that have overlapping
        data vars. The overlapping variable will be given an additional overlap_dim
        coordinate to be used in reference a given dataset's variable.
        The coordinates of this dimension are the elements of the source_names
        arg. They must be given in the order corresponding to the mappers arg.
        Args:
            mappers_left: mapper to merge
            mappers_right: mapper to merge
            source_name_left (str): source names for the left mappers to be merged,
                used as coordinate for overlapping variables in the merged
                dataset if there is no existing overlap_dim coordinate for those
                variables. Must be provided if overlap_dim is not an existing
                dimension in left mapper. If overlap_dim exists, give as None.
                Defaults to None.
            source_name_right (str): Same as source_name_left but for right mapper.
            overlap_dim (str): name of dimension to concat overlapping variables along.
        """
        self._mappers = [mapper_left, mapper_right]
        self._source_names = [source_name_left, source_name_right]
        self._var_overlap = self._get_var_overlap(self._mappers)
        self._overlap_dim = overlap_dim
        self._check_overlap_vars_dims(
            self._mappers, self._source_names, self._var_overlap, self._overlap_dim
        )

    def keys(self):
        mappers_keys = [mapper.keys() for mapper in self._mappers]
        return set(mappers_keys[0]).intersection(*mappers_keys)

    def __getitem__(self, key: str):
        datasets_to_merge = [mapper[key] for mapper in self._mappers]
        return self._merge_with_overlap(datasets_to_merge)

    def _merge_with_overlap(self, datasets: Sequence[xr.Dataset]) -> xr.Dataset:
        ds_nonoverlap = xr.merge([ds.drop(list(self._var_overlap)) for ds in datasets])
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
    def _check_overlap_vars_dims(mappers, source_names, overlap_vars, overlap_dim):
        datasets = [get_sample_dataset(mapper) for mapper in mappers]
        for ds, source_coord in zip(datasets, source_names):
            overlap_dim_occurence = 0
            vars_missing_overlap_dim = []
            for var in overlap_vars:
                if overlap_dim in ds[var].dims:
                    overlap_dim_occurence += 1
                else:
                    vars_missing_overlap_dim.append(var)
            if 0 < overlap_dim_occurence < len(overlap_vars):
                raise ValueError(
                    f"If overlap dimension {overlap_dim} already exists in "
                    "one of the mappers it must be present for all overlapping "
                    f"variables. Variables {vars_missing_overlap_dim} are missing "
                    f"dimension {overlap_dim}."
                )
            if overlap_dim in ds.dims and source_coord is not None:
                raise ValueError(
                    "Overlap dimension already exists in dataset with source name "
                    f"given as {source_coord}. If overlap_dim exists, source name "
                    "must be given as None."
                )
