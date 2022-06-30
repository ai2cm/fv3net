import sys
sys.path.append("/home/yakelyn/explore/yakelyn/utils/")
import mycolors
import os
import fv3viz
import xarray as xr
import intake
import fsspec
from dask.diagnostics import ProgressBar
from vcm.catalog import catalog as CATALOG
from vcm.fv3.metadata import standardize_fv3_diagnostics
from vcm.safe import get_variables
from vcm import convert_timestamps
import cftime
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import vcm

def plot_global_map_biases(
    ds_a,
    ds_b, 
    GRID,
    var_name,
    cmap,                    
    units,
    output_dir,
    plot_filename,
    title,
    title_a,
    title_b):

    "Plot a 3 row figure. Global maps for ds_a, ds_b, bias = ds_a - ds_b "
    vmin = np.nanmin(ds_a).round()
    vmax = np.nanmax(ds_a).round()

    fig = plt.figure(figsize = (4, 6)) 
    gs = GridSpec(3,1)
    gs.update(wspace=.35, hspace=0.2)

    ax = fig.add_subplot(gs[0,0],projection=ccrs.Robinson())
    fv3viz.plot_cube(xr.merge([ds_a, GRID]), var_name, cbar_label= units, ax = ax, vmin = vmin, vmax = vmax ,cmap = cmap)
    ax.set_title( title +'\n' + title_a, fontsize = 7)


    ax = fig.add_subplot(gs[1,0],projection=ccrs.Robinson())
    fv3viz.plot_cube(xr.merge([ds_b, GRID]), var_name, cbar_label= units, ax = ax, vmin = vmin, vmax = vmax,cmap = cmap)
    ax.set_title(title_b, fontsize = 7)


    ax = fig.add_subplot(gs[2,0],projection=ccrs.Robinson())
    fv3viz.plot_cube(xr.merge([ds_a - ds_b , GRID]), var_name, cbar_label= units, ax = ax, cmap = mycolors.blues2reds())
    ax.set_title( title_a + ' - ' + title_b +' bias',fontsize = 7)

    plt.savefig(os.path.join(output_dir, plot_filename), dpi=200, bbox_inches='tight')
    plt.close('all')
    return fig

def find_common_times(ds1, ds2):
    return list(set(ds1.time.values).intersection(set(ds2.time.values)))
    
def global_average(x):
    return x.mean(['x', 'y','tile'])

def sse(truth,prediction):
    ## returns the sum of squares of the residual errors
    return (truth-prediction)**2

def ss(truth, prediction):
    ## returns the total sum of the errors relative its global mean
    return (truth - global_average(prediction))**2

def r2_score(truth, prediction):
    mse = global_average(sse(truth,prediction))
    ss = global_average(ss(truth, truth))
    return 1 - mse / ss

def r2_integrated_skill(truth, prediction, delp , dim):
    mse = global_average(vcm.mass_integrate(sse(truth,prediction),delp ,dim))
    ss  = global_average(vcm.mass_integrate(ss(truth,truth), delp, dim))
    return 1 - mse / ss

def mse(truth, prediction):
    return global_average(sse(truth,prediction))

def subset(data3d, GRID,range_lats, range_lons):    
    lons   = GRID.lon.stack(sample=["tile", "x", "y"])
    lats   = GRID.lat.stack(sample=["tile", "x", "y"])
    data2d = data3d.stack(sample=["tile", "x", "y"]).transpose("sample", ...)
    find_this = np.where(np.logical_and(np.logical_and(lats >= range_lats[0] , lats <= range_lats[-1]),
                         np.logical_and(lons >= range_lons[0] , lons <= range_lons[-1])))[0]
    
    return data2d[find_this,:], lons[find_this], lats[find_this]

