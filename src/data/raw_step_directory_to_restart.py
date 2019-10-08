"""
Converts a raw tar extraction with high-res surface data to an
input data directory at a specified resolution.

This directory has a format like this:


data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile1.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile2.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile3.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile4.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile5.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile6.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile1.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile2.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile3.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile4.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile5.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile6.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile1.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile2.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile3.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile4.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile5.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile6.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile1.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile2.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile3.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile4.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile5.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile6.nc.0001

It needs to be changed to:

coupler.res           fv_core.res.tile6.nc     fv_tracer.res.tile1.nc  oro_data.tile1.nc
fv_core.res.nc        fv_srf_wnd.res.tile1.nc  fv_tracer.res.tile2.nc  oro_data.tile2.nc
fv_core.res.tile1.nc  fv_srf_wnd.res.tile2.nc  fv_tracer.res.tile3.nc  oro_data.tile3.nc
fv_core.res.tile2.nc  fv_srf_wnd.res.tile3.nc  fv_tracer.res.tile4.nc  oro_data.tile4.nc
fv_core.res.tile3.nc  fv_srf_wnd.res.tile4.nc  fv_tracer.res.tile5.nc  oro_data.tile5.nc
fv_core.res.tile4.nc  fv_srf_wnd.res.tile5.nc  fv_tracer.res.tile6.nc  oro_data.tile6.nc
fv_core.res.tile5.nc  fv_srf_wnd.res.tile6.nc  sfc_data.tile1....

"""
from src.fv3 import *
import xarray as xr
from itertools import product
import logging

logging.basicConfig(level=logging.INFO)


def grid_and_sfc_data_paths(tile, subtile, time):
    grid_path = f"data/raw/grid_specs/c3072/grid_spec.tile{tile:d}.nc.{subtile:04d}"
    sfc_path = f"data/extracted/{time}/{time}.sfc_data.tile{tile:d}.nc.{subtile:04d}"
    return tile, subtile, sfc_path, grid_path

num_tiles = 6
num_subtiles = 16
time = '20160805.170000'
tiles = list(range(1, num_tiles + 1))
subtiles = list(range(num_subtiles))

files = [grid_and_sfc_data_paths(tile, proc, time)
         for tile, proc in product(tiles, subtiles)]

sfc_data = coarsen_sfc_data_in_directory(files, method='median')
sfc_data.to_netcdf("output.nc")
