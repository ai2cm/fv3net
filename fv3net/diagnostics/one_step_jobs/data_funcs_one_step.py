from typing import Sequence, Tuple#, List
import xarray as xr
import numpy as np
from vcm.cubedsphere.constants import TIME_FMT
# from vcm import parse_timestep_str_from_path, parse_datetime_from_str
from datetime import datetime, timedelta
# import logging

from fv3net.diagnostics.one_step_jobs import thermo
from fv3net.pipelines.common import subsample_timesteps_at_interval

# logger = logging.getLogger(__name__)

INIT_TIME_DIM = 'initial_time'
FORECAST_TIME_DIM = 'forecast_time'
DELTA_DIM = 'model_run'
VARS_FOR_PLOTS = [
    'psurf',
    'total_heat',
    'precipitable_water',
    'total_water',
    'specific_humidity',
    'vertical_wind',
    'air_temperature',
    'pressure_thickness_of_atmospheric_layer',
    'land_sea_mask'
]
GRID_VARS = ['lat', 'lon', 'latb', 'lonb', 'area']
VAR_TYPE_DIM = 'var_type'


def time_inds_to_open(time_coord: xr.DataArray, n_sample: int) -> Sequence:
    
    timestamp_list = list(time_coord[INIT_TIME_DIM].values)
    timestamp_list_no_last = timestamp_list[:-1]
    duration_minutes, restart_interval_minutes = _duration_and_interval(timestamp_list_no_last)
    sampling_interval_minutes = _sampling_interval_minutes(duration_minutes, restart_interval_minutes, n_sample)
    timestamp_subset = subsample_timesteps_at_interval(timestamp_list_no_last, sampling_interval_minutes)
    timestamp_subset_indices = [
        _timestamp_pairs_indices(timestamp_list, timestamp, restart_interval_minutes)
        for timestamp in timestamp_subset
    ]
    timestep_subset_indices = filter(lambda x: True if x[0] is not None else False, timestamp_subset_indices)
    
    return timestep_subset_indices


def _duration_and_interval(timestamp_list: list) -> (int, int):
    
    first_and_last = [datetime.strptime(timestamp_list[i], TIME_FMT) for i in (0, -1)]
    first_and_second = [datetime.strptime(timestamp_list[i], TIME_FMT) for i in (0, 1)]
    duration_minutes = (first_and_last[1] - first_and_last[0]).seconds/60
    n_steps = len(timestamp_list)
    restart_interval_minutes = duration_minutes//(n_steps - 1)
    restart_interval_minutes_first = (first_and_second[1] - first_and_second[0]).seconds/60
    if restart_interval_minutes != restart_interval_minutes_first:
        raise ValueError('Incomplete initialization set detected in .zarr')
        
    return duration_minutes, restart_interval_minutes
    

def _sampling_interval_minutes(duration_minutes: int, restart_interval_minutes: int, n_sample: int) -> int:
    try: 
        interval_minutes = restart_interval_minutes*((duration_minutes/(n_sample - 1))//restart_interval_minutes)
    except ZeroDivisionError:
        interval_minutes = None
    return interval_minutes


def _timestamp_pairs_indices(timestamp_list: list, timestamp: str, time_fmt: str = TIME_FMT, time_interval_minutes: int = 15) -> Tuple:
    
    timestamp1 = datetime.strftime(datetime.strptime(timestamp, TIME_FMT) + timedelta(minutes = time_interval_minutes), TIME_FMT)
    try:
        index0 = timestamp_list.index(timestamp)
        index1 = timestamp_list.index(timestamp1)
        pairs = index0, index1
    except ValueError:
        pairs = None, None
    return pairs


def time_coord_from_str_coord(str_coord: xr.DataArray, dim: str = INIT_TIME_DIM, fmt = TIME_FMT) -> xr.DataArray:
    return xr.DataArray(
        data = [datetime.strptime(time.item(), TIME_FMT) for time in str_coord],
        dims = (dim,)
    )


def insert_derived_vars_from_ds_zarr(ds: xr.Dataset) -> xr.Dataset:
    
    ds = ds.assign({'total_water' : thermo._total_water(
        ds['specific_humidity'],
        ds['cloud_ice_mixing_ratio'],
        ds['cloud_water_mixing_ratio'],
        ds['rain_mixing_ratio'],
        ds['snow_mixing_ratio'],
        ds['graupel_mixing_ratio'],
        ds['pressure_thickness_of_atmospheric_layer'],
    )})
    ds = ds.assign({'psurf' : thermo._psurf_from_delp(
        ds['pressure_thickness_of_atmospheric_layer']
    )})
    ds = ds.assign({'precipitable_water': thermo._precipitable_water(
        ds['specific_humidity'],
        ds['pressure_thickness_of_atmospheric_layer']
    )})
    ds = ds.assign({'total_heat' : thermo._total_heat(
        ds['air_temperature'],
        ds['pressure_thickness_of_atmospheric_layer']
    )})
    
    return ds


def _align_time_and_concat(ds_hires: xr.Dataset, ds_coarse: xr.Dataset) -> xr.Dataset:

    ds_coarse = (
        ds_coarse
        .isel({INIT_TIME_DIM : [0]})
        .assign_coords({DELTA_DIM : 'coarse'})
        .drop(labels = 'step')
        .squeeze()
    )

    ds_hires = (
        ds_hires
        .isel({FORECAST_TIME_DIM : [0]})
        .sel({'step': 'begin'})
        .drop(labels = 'step')
        .assign_coords({DELTA_DIM : 'hi-res'})
        .squeeze()
    )
    
    dim_excl = [dim for dim in ds_hires.dims if dim != FORECAST_TIME_DIM]
    ds_hires = ds_hires.broadcast_like(ds_coarse, exclude = dim_excl)

    return xr.concat([ds_hires, ds_coarse], dim = DELTA_DIM)


def _compute_both_tendencies_and_concat(ds: xr.Dataset) -> xr.Dataset:
    
    dt_init = ds[INIT_TIME_DIM].diff(INIT_TIME_DIM).isel({INIT_TIME_DIM: 0})/np.timedelta64(1,'s')
    tendencies_hires = ds[VARS_FOR_PLOTS].diff(INIT_TIME_DIM, label = 'lower')/dt_init

    dt_forecast = ds[FORECAST_TIME_DIM].diff(FORECAST_TIME_DIM).isel({FORECAST_TIME_DIM: 0})
    tendencies_coarse = ds[VARS_FOR_PLOTS].diff('step', label = 'lower')/dt_forecast
    
    tendencies_both = _align_time_and_concat(tendencies_hires, tendencies_coarse)
    
    return tendencies_both


def _select_both_states_and_concat(ds: xr.Dataset) -> xr.Dataset:
    
    states_hires = ds[VARS_FOR_PLOTS].isel({INIT_TIME_DIM: slice(None, -1)})
    states_coarse = ds[VARS_FOR_PLOTS].sel({'step': "begin"}) 
    states_both = _align_time_and_concat(states_hires, states_coarse)
    
    return states_both


def get_states_and_tendencies(ds_sample: Sequence[xr.Dataset]) -> xr.Dataset:
    
    tendencies_both_sample = [_compute_both_tendencies_and_concat(ds) for ds in ds_sample]
    tendencies = xr.concat([tendencies_both for tendencies_both in tendencies_both_sample],
                             dim = INIT_TIME_DIM)
    
    states_both_sample = [_select_both_states_and_concat(ds) for ds in ds_sample]
    states = xr.concat([states_both for states_both in states_both_sample], dim = INIT_TIME_DIM)

    states_and_tendencies = xr.concat([states, tendencies], dim=VAR_TYPE_DIM)
    states_and_tendencies = states_and_tendencies.assign_coords({VAR_TYPE_DIM: ['states', 'tendencies']})
    
    return states_and_tendencies