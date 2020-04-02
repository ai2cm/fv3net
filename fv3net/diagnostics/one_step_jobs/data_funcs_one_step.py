from typing import Sequence, Tuple
import xarray as xr
import numpy as np
from vcm.cubedsphere.constants import TIME_FMT
from vcm.select import mask_to_surface_type
from vcm.convenience import parse_datetime_from_str
from datetime import datetime, timedelta
import logging

from fv3net.diagnostics.one_step_jobs import thermo
from fv3net.pipelines.common import subsample_timesteps_at_interval

logger = logging.getLogger(__name__)

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
    "DSWRFtoa",
    "DSWRFsfc",
    "USWRFtoa",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFtoa",
    "ULWRFsfc"
]
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


def time_coord_to_datetime(ds: xr.Dataset, time_coord: str = INIT_TIME_DIM) -> xr.Dataset:
    
    init_datetime_coords = [parse_datetime_from_str(timestamp) for timestamp in ds[time_coord].values]
    ds = ds.assign_coords({time_coord: init_datetime_coords})
    
    return ds


def insert_derived_vars_from_ds_zarr(ds: xr.Dataset) -> xr.Dataset:
    
    ds = ds.assign({
        'total_water' : thermo.total_water(
            ds['specific_humidity'],
            ds['cloud_ice_mixing_ratio'],
            ds['cloud_water_mixing_ratio'],
            ds['rain_mixing_ratio'],
            ds['snow_mixing_ratio'],
            ds['graupel_mixing_ratio'],
            ds['pressure_thickness_of_atmospheric_layer']
        ),
        'psurf' : thermo.psurf_from_delp(
            ds['pressure_thickness_of_atmospheric_layer']
        ),
        'precipitable_water': thermo.precipitable_water(
            ds['specific_humidity'],
            ds['pressure_thickness_of_atmospheric_layer']
        ),
        'total_heat' : thermo.total_heat(
            ds['air_temperature'],
            ds['pressure_thickness_of_atmospheric_layer']
        )
    })
    
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


def insert_column_integrated_tendencies(ds: xr.Dataset) -> xr.Dataset:
    
    ds = ds.assign({
        'column_integrated_heating' : thermo.column_integrated_heating(
            ds['air_temperature'].sel({VAR_TYPE_DIM: 'tendencies'}),
            ds['pressure_thickness_of_atmospheric_layer'].sel({VAR_TYPE_DIM: 'states'}),
        ).expand_dims({VAR_TYPE_DIM: ['tendencies']}),
        'column_integrated_moistening' : thermo.column_integrated_moistening(
            ds['specific_humidity'].sel({VAR_TYPE_DIM: 'tendencies'}),
            ds['pressure_thickness_of_atmospheric_layer'].sel({VAR_TYPE_DIM: 'states'}).expand_dims({VAR_TYPE_DIM: ['tendencies']}),
        )
    })
    
    return ds
    
    
def insert_model_run_differences(ds: xr.Dataset) -> xr.Dataset:

    ds_residual = (
        ds.sel({DELTA_DIM:  'hi-res'}) - ds.sel({DELTA_DIM: 'coarse'})
    ).assign_coords({DELTA_DIM: 'hi-res - coarse'})

    # combine into one dataset
    return xr.concat([ds, ds_residual], dim = DELTA_DIM)


def insert_abs_vars(ds: xr.Dataset, varnames: Sequence) -> xr.Dataset:
    
    for var in varnames:
        if var in ds:
            ds[var + '_abs'] = np.abs(ds[var])
            ds[var + '_abs'].attrs.update({'long_name' : f"absolute {ds[var].attrs['long_name']}"})
        else:
            raise ValueError('Invalid variable name for absolute tendencies.')
    
    return ds


def insert_variable_at_model_level(ds: xr.Dataset, varnames: list, level: int):
    
    for var in varnames:
        if var in ds:
            new_name = f"{var}_level_{level}"
            ds = ds.assign({new_name: ds[var].sel({"z": level})})
            ds[new_name].attrs.update({
                'long_name': f"{var} at model level {level}",
            })
        else:
            raise ValueError('Invalid variable for model level selection.')
    
    return ds


def make_init_time_dim_intelligible(ds: xr.Dataset):
    
    ds = ds.assign_coords({
        INIT_TIME_DIM: [
            datetime
            .utcfromtimestamp((time - np.datetime64(0, 's'))/np.timedelta64(1, 's'))
            .strftime(TIME_FMT) for time in ds[INIT_TIME_DIM]
        ]
    })
    
    return ds


def insert_weighted_mean_vars(
    ds: xr.Dataset,
    weights: xr.DataArray,
    varnames: list,
    mask_name = 'land_sea_mask',
    dims = ['tile', 'x', 'y']
) -> xr.Dataset:
    
    wm = (ds*weights).sum(dim = dims)/weights.sum(dim = dims)
    for var in varnames:
        if var in ds:
            new_name = f"{var}_global_mean"
            ds = ds.assign({new_name: wm[var]})
            ds[new_name] = ds[new_name].assign_attrs(ds[var].attrs)
        else:
            raise ValueError('Variable for global mean calculations not in dataset.')
        
    if mask_name == 'land_sea_mask':
        ds_land = mask_to_surface_type(ds.merge(weights), 'land', surface_type_varname = mask_name)
        weights_land = ds_land['area']
        wm_land = (ds_land*weights_land).sum(dim = dims)/weights_land.sum(dim = dims)
        ds_sea = mask_to_surface_type(ds.merge(weights), 'sea', surface_type_varname = mask_name)
        weights_sea = ds_sea['area']
        wm_sea = (ds_sea*weights_sea).sum(dim = dims)/weights_sea.sum(dim = dims)
        for var in varnames:
            if var in ds:
                land_new_name = f"{var}_land_mean"
                sea_new_name = f"{var}_sea_mean"
                ds = ds.assign({
                    land_new_name: wm_land[var],
                    sea_new_name: wm_sea[var]
                })
                ds[land_new_name] = ds[land_new_name].assign_attrs(ds[var].attrs)
                ds[sea_new_name] = ds[sea_new_name].assign_attrs(ds[var].attrs)
            else:
                raise ValueError('Variable for global mean calculations not in dataset.')
    else:
        raise ValueError('Only "land_sea_mask" is suppored as a mask.')
            
    return ds


def shrink_ds(ds: xr.Dataset, init_dim: str = INIT_TIME_DIM):
    
    for var in ds:
        if init_dim in ds[var].dims:
            ds = ds.assign({f"{var}_std": ds[var].std(dim = init_dim, keep_attrs=True)})
    ds_mean = ds.mean(dim=INIT_TIME_DIM, keep_attrs=True)
    
    dropvars = []
    for var in ds:
        if 'z' in ds[var].dims and 'tile' in ds[var].dims:
            dropvars.append(var)
            
    return ds.drop_vars(dropvars)
                                           
