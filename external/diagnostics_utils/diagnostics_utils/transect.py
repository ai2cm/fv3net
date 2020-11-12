import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple
import xarray as xr
import vcm


def meridional_ring(lon=0, n=180):
    attrs = {"description": f"Lon = {lon}"}
    lat = np.linspace(-90, 90, n)
    lon = np.ones_like(lat) * lon
 
    return {
        "lat": xr.DataArray(lat, dims="sample", attrs=attrs),
        "lon": xr.DataArray(lon, dims="sample", attrs=attrs)
    }


def zonal_ring(lat=45, n=360):
    attrs = {"description": f"Lat = {lat}"}
    lon = np.linspace(0, 360, n)
    lat = np.ones_like(lon) * lat

    return {
        "lat": xr.DataArray(lat, dims="sample", attrs=attrs),
        "lon": xr.DataArray(lon, dims="sample", attrs=attrs)
    }


def plot_transect(
        truth: xr.DataArray,
        prediction: xr.DataArray,
        transect_coords: xr.DataArray,
        x: str = "lon",
        figsize: Tuple[int] = (10, 4)):
    
    concatenated = xr.concat([truth, prediction], dim=pd.Index(["truth", "prediction"], name="source"))
    meridian_pred = vcm.regrid.interpolate_unstructured(
        concatenated, transect_coords
    )
    for output in meridian_pred.data_vars:
        meridian_pred[output].plot(y="z", x=x, yincrease=False, col="source", figsize=figsize)
        plt.suptitle(transect_coords["lat"].attrs["description"])
    