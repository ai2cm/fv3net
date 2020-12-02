import os
import fsspec
import zarr.storage as zstore
import xarray as xr
from typing import Sequence, Mapping, Union

from .._base import LongRunMapper, GeoMapper
from ..._utils import standardize_zarr_time_coord

Time = str


class SubsetTimes(GeoMapper):
    """
    Sort and subset a timestep-based mapping to skip spin-up and limit
    the number of available times.
    """

    def __init__(
        self,
        i_start: int,
        n_times: Union[int, None],
        nudged_data: Mapping[str, xr.Dataset],
    ):
        timestep_keys = list(nudged_data.keys())
        timestep_keys.sort()

        i_end = None if n_times is None else i_start + n_times
        self._keys = timestep_keys[slice(i_start, i_end)]
        self._nudged_data = nudged_data

    def keys(self):
        return set(self._keys)

    def __getitem__(self, time: Time):
        if time not in self._keys:
            raise KeyError("Time {time} not found in SubsetTimes mapper.")
        return self._nudged_data[time]


class MergeNudged(LongRunMapper):
    """
    Mapper for merging data sources available from a nudged run.
    
    Currently used to merge the nudging tendencies with the after
    physics checkpointed state information. Could be useful for
    merging prognostic run output by time in the future.
    """

    def __init__(
        self,
        *nudged_sources: Sequence[Union[LongRunMapper, xr.Dataset]],
        rename_vars: Mapping[str, str] = None,
    ):
        rename_vars = rename_vars or {}
        if len(nudged_sources) < 2:
            raise TypeError(
                "MergeNudged should be instantiated with two or more data sources."
            )
        nudged_sources = self._mapper_to_datasets(nudged_sources)
        nudged_sources = self._rename_vars(nudged_sources, rename_vars)
        self._check_dvar_overlap(*nudged_sources)
        self.ds = xr.merge(nudged_sources, join="inner")

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
        data_sources: Sequence[Union[LongRunMapper, xr.Dataset]]
    ) -> Sequence[xr.Dataset]:

        datasets = []
        for source in data_sources:
            if isinstance(source, LongRunMapper):
                source = source.ds
            datasets.append(standardize_zarr_time_coord(source))

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
                "Could not combine requested nudged data sources due to "
                f"overlapping variables {overlap}"
            )


class SubtractNudgingTendency(GeoMapper):
    """Subtract nudging tendency from physics tendency. Necessary for nudge-to-obs."""

    def __init__(
        self,
        nudged_mapper: Mapping[Time, xr.Dataset],
        nudging_to_physics_tendency: Mapping[str, str],
    ):
        self._nudged_mapper = nudged_mapper
        self._nudging_to_physics_tendency = nudging_to_physics_tendency

    def keys(self):
        return self._nudged_mapper.keys()

    def __getitem__(self, time: Time) -> xr.Dataset:
        return self._derived_ds(time)

    def _derived_ds(self, time: Time):
        differenced_physics_tendency = self._subtract_nudging_tendency(time)
        return self._nudged_mapper[time].assign(differenced_physics_tendency)

    def _subtract_nudging_tendency(self, time: Time) -> Mapping[str, xr.DataArray]:
        differenced_physics_tendency = {}
        for nudging_name, physics_name in self._nudging_to_physics_tendency.items():
            differenced_physics_tendency[physics_name] = (
                self._nudged_mapper[time][physics_name]
                - self._nudged_mapper[time][nudging_name]
            )
        return differenced_physics_tendency


def _get_source_datasets(
    url: str, sources: Sequence[str], consolidated: bool = False
) -> Sequence[xr.Dataset]:
    datasets = []
    for source in sources:
        mapper = fsspec.get_mapper(os.path.join(url, f"{source}"))
        ds = xr.open_zarr(
            zstore.LRUStoreCache(mapper, 1024),
            consolidated=consolidated,
            mask_and_scale=False,
        )
        datasets.append(ds)
    return datasets
