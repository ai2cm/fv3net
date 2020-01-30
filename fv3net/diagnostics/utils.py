from dataclasses import dataclass
import fsspec
import os
from typing import List
import xarray as xr
import yaml
from vcm.calc import diag_ufuncs
from vcm.fv3_restarts import (
    open_restarts_with_time_coordinates,
    open_grid,
    _split_url,
    _parse_time,
)


@dataclass
class PlotConfig:
    """
    Stores instructions for creating a single figure.
    """

    diagnostic_variable: str
    plot_name: str
    plotting_function: str
    dim_slices: dict
    functions: List
    function_kwargs: List[dict]
    plot_params: dict
    time_dim: str = None


def _open_rundir_data(data_path, grid_path):
    protocol, path = _split_url(data_path)
    fs = fsspec.filesystem(protocol)
    data = open_restarts_with_time_coordinates(data_path).drop("file_prefix")
    if grid_path:
        grid = xr.open_zarr(fs.get_mapper(grid_path))
    else:
        grid = open_grid(data_path)
    return xr.merge([data, grid])


def _open_zarr_data(data_path, grid_path):
    protocol, path = _split_url(data_path)
    fs = fsspec.filesystem(protocol)
    data = xr.open_zarr(fs.get_mapper(data_path))
    grid = xr.open_zarr(fs.get_mapper(grid_path))
    return xr.merge([data, grid])


def open_dataset(data_path, grid_path):
    try:
        if ".zarr" in data_path:
            ds = _open_zarr_data(data_path, grid_path)
        else:
            ds = _open_rundir_data(data_path, grid_path)
    except ValueError as e:
        raise (
            "Cannot open zarr or run directory at data path provided."
            "Check the input argument and make sure it is one of"
            f"these allowed data formats. {e}"
        )
    return ds


def load_ufuncs(raw_config):
    """Uses list of function names in config to get the user defined functions and kwargs
    used to calculate diagnostic quantities

    Args:
        raw_config: Config entry as read in directly from file (before formatting into
        the PlotConfig object)

    Returns:
        list: functions to be piped to data
        list[dict]: function kwargs for each function
    """
    if "function_specifications" in raw_config:
        functions, kwargs = [], []
        for function_spec in raw_config["function_specifications"]:
            # sacrificing some ugliness here so that specification in the yaml is
            # safer / more readable: single entry dict seemed like a more foolproof
            # format than list of func name and kwargs
            function_name, function_kwargs = (
                list(function_spec.keys())[0],
                list(function_spec.values())[0],
            )
            # handle case where function name is given with no kwargs attached
            if not function_kwargs:
                function_kwargs = {}
            if not hasattr(diag_ufuncs, function_name):
                raise ValueError(
                    f"Function name {function_name} is not in vcm.calc.diag_ufuncs."
                )
            functions.append(getattr(diag_ufuncs, function_name))
            kwargs.append(function_kwargs)
        return functions, kwargs
    else:
        return [], []


def load_dim_slices(raw_config):
    """ Parse config for the appropriate input to use in xr.dataset.isel()

    Args:
        raw_config: config entry as read directly from file

    Returns:
        dict: {dim: selection} to be used as input to ds.isel()
    """
    dim_selection = {}
    if "dim_slices" in raw_config:
        for dim, indices in raw_config["dim_slices"].items():
            if len(indices) == 1:
                dim_selection[dim] = indices[0]
            else:
                indices += [None]
                dim_selection[dim] = slice(indices[0], indices[1], indices[2])
        return dim_selection
    else:
        return {}


def load_plot_params(raw_config):
    """ Load arguments for plotting functions

    Args:
        raw_config: config entry as read directly from file

    Returns:
        dict: arguments for plot functions in fv3net.diagnostics.visualize
    """
    if "plot_params" in raw_config and raw_config["plot_params"] is not None:
        return raw_config["plot_params"]
    else:
        return {}


def load_diagnostic_vars(raw_config):
    """ Get diagnostic variable(s) from config. If single string provided put it into
    single element list. This is so that later plotting functions can put two variables
    on same figure.

    Args:
        raw_config: config entry as read directly from file

    Returns:
        list: diagnostic variable names
    """
    if isinstance(raw_config["diagnostic_variable"], list):
        diag_var = raw_config["diagnostic_variable"]
    elif isinstance(raw_config["diagnostic_variable"], str):
        diag_var = [raw_config["diagnostic_variable"]]
    else:
        raise TypeError(
            "diagnostic_variable in the config file must be a single string"
            "or list of strings"
        )
    return diag_var


def load_configs(config_path):
    """ Loads the plot configs from yaml file into PlotConfig objects for
    ease of use later on

    Args:
        config_path: configuration yaml file path

    Returns:
        list: PlotConfig objects
    """
    with open(config_path, "r") as stream:
        try:
            raw_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad yaml config: {exc}")
    plot_configs = []
    if isinstance(raw_configs, dict):
        raw_configs = [raw_configs]
    for raw_config in raw_configs:
        functions, function_kwargs = load_ufuncs(raw_config)
        plot_config = PlotConfig(
            plot_name=raw_config["plot_name"],
            plotting_function=raw_config["plotting_function"],
            diagnostic_variable=load_diagnostic_vars(raw_config),
            dim_slices=load_dim_slices(raw_config),
            functions=functions,
            function_kwargs=function_kwargs,
            plot_params=load_plot_params(raw_config),
        )
        if "time_dim" in raw_config:
            plot_config.time_dim = raw_config["time_dim"]
        plot_configs.append(plot_config)
    return plot_configs
