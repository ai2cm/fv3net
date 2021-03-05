from typing import Sequence, Union, Mapping
import xarray as xr
import vcm

from ._base import GeoMapper, XarrayMapper
from ..constants import DERIVATION_DIM, TIME_NAME
from .._utils import get_sample_dataset


def _round_ds_times(ds: xr.Dataset, time: str = TIME_NAME) -> xr.Dataset:
    rounded_times = vcm.convenience.round_time(ds[time].values)
    return ds.assign_coords({time: rounded_times})


class MergedMapper(XarrayMapper):
    """
    Mapper which is produced from merging two XarrayMappers, or two xarray datasets,
    via an inner join. Assumes no overlapping data variables between the two inputs.
    """

    def __init__(
        self,
        *sources: Sequence[Union[XarrayMapper, xr.Dataset]],
        rename_vars: Mapping[str, str] = None,
        time: str = TIME_NAME,
    ):
        rename_vars = rename_vars or {}
        if len(sources) < 2:
            raise TypeError(
                "MergedMapper should be instantiated with two or more data sources."
            )
        sources = self._mapper_to_datasets(sources)
        sources = self._rename_vars(sources, rename_vars)
        self._check_dvar_overlap(*sources)
        self.ds = xr.merge(sources, join="inner")
        times = self.ds[time].values.tolist()
        time_strings = [vcm.encode_time(single_time) for single_time in times]
        self.time_lookup = dict(zip(time_strings, times))
        self.time_string_lookup = dict(zip(times, time_strings))

    @staticmethod
    def _rename_vars(
        datasets: Sequence[xr.Dataset], rename_vars: Mapping[str, str]
    ) -> Sequence[xr.Dataset]:
        renamed_datasets = []
        for ds in datasets:
            ds_rename_vars = {k: v for k, v in rename_vars.items() if k in ds}
            renamed_datasets.append(ds.rename(ds_rename_vars))
        return renamed_datasets

    @staticmethod
    def _mapper_to_datasets(
        data_sources: Sequence[Union[XarrayMapper, xr.Dataset]]
    ) -> Sequence[xr.Dataset]:

        datasets = []
        for source in data_sources:
            if isinstance(source, XarrayMapper):
                source = source.ds
            datasets.append(_round_ds_times(source))

        return datasets

    @staticmethod
    def _check_dvar_overlap(*ds_to_combine):
        ds_var_sets = [set(ds.data_vars.keys()) for ds in ds_to_combine]

        overlap = set()
        checked = set()
        for data_var in ds_var_sets:
            overlap |= data_var & checked
            checked |= data_var

        if overlap:
            raise ValueError(
                "Could not combine requested data sources due to "
                f"overlapping variables {overlap}"
            )


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
        ds_nonoverlap = xr.merge(
            [ds.drop_vars(list(self._var_overlap)) for ds in datasets]
        )
        overlapping = []
        for ds, source_coord in zip(datasets, self._source_names):
            if self._overlap_dim in ds.dims:
                overlapping.append(vcm.safe.get_variables(ds, self._var_overlap))
            else:
                overlapping.append(
                    vcm.safe.get_variables(ds, self._var_overlap)
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
