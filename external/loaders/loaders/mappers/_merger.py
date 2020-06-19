from typing import Sequence
import xarray as xr
from vcm import safe

from ._base import GeoMapper
from ..constants import DERIVATION_DIM


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
                "MergeData should be instantiated with two or more data sources."
            )
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
            overlapping.append(
                safe.get_variables(ds, self._var_overlap) \
                    .expand_dims(self._overlap_dim) \
                    .assign_coords({self._overlap_dim: [source_coord]})
            )
        return xr.merge(overlapping + [ds_nonoverlap])

    @staticmethod
    def _get_var_overlap(mappers_to_combine):
        ds_var_sets = []
        for mapper in mappers_to_combine:
            sample_key = list(mapper.keys())[0]
            ds_var_sets.append(set(mapper[sample_key].data_vars))
        overlap = set()
        checked = set()
        for data_var in ds_var_sets:
            overlap |= data_var & checked
            checked |= data_var
        return overlap
