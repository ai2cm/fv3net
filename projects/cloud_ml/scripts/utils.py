import os
import fv3viz
import xarray as xr
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from matplotlib.colors import ListedColormap

def WhiteBlueGreenYellowRed():
    rgbset2 = np.array([
    [256,256,256,256],
    [204,229,255,256],
    [150,210,250,256],
    [100,190,240,256],
    [80,165,225,256],
    [70,145,215,256],
    [60,150,235,256],
    [50,140,225,256],
    [40,100,220,256],
    [40,80,200,256],
    [204,255,229,256],
    [153,255,204,256],
    [102,245,165,256],
    [90,235,155,256],
    [10,204,102,256],
    [0,210,102,256],
    [0,183,90,256],
    [0,153,80,256],
    [0,140,80,256],
    [0,130,70,256],
    [255,255,204,256],
    [255,232,120,256],
    [252,220,10,256],
    [252,187,10,256],
    [252,163,10,256],
    [252,123,10,256],
    [252,82,10,256],
    [255,51,51,256],
    [185,0,0,256],
    [145,0,0,256]])
    newcmp = ListedColormap(rgbset2/256)
    return newcmp

def blues2reds():
    rgbset2 = np.array([[0,0,180,256],
    [10,50,200,256],
    [10,80,230,256],
    [30,110,245,256],
    [40,120,255,256],
    [60,140,255,256],
    [80,160,255,256],
    [120,185,255,256],
    [150,210,255,256],
    [180,230,255,256],
    [190,240,255,256],
    [255,255,220,256],
    [255,232,120,256],
    [255,192,60,256],
    [255,160,0,256],
    [255,96,0,256],
    [255,60,0,256],
    [255,40,0,256],
    [225,20,0,256],
    [190,0,0,256],
    [170,0,0,256],
    [140,0,0,256]])
    newcmp = ListedColormap(rgbset2/256)

    return newcmp

def subset(data3d, GRID,range_lats, range_lons):    
    lons   = GRID.lon.stack(sample=["tile", "x", "y"])
    lats   = GRID.lat.stack(sample=["tile", "x", "y"])
    data2d = data3d.stack(sample=["tile", "x", "y"]).transpose("sample", ...)
    find_this = np.where(np.logical_and(np.logical_and(lats >= range_lats[0] , lats <= range_lats[-1]),
                         np.logical_and(lons >= range_lons[0] , lons <= range_lons[-1])))[0]
    
    return data2d[find_this,:], lons[find_this], lats[find_this]

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
    fv3viz.plot_cube(xr.merge([ds_a - ds_b , GRID]), var_name, cbar_label= units, ax = ax, cmap = blues2reds())
    ax.set_title( title_a + ' - ' + title_b +' bias',fontsize = 7)

    plt.savefig(os.path.join(output_dir, plot_filename), dpi=200, bbox_inches='tight')
    plt.close('all')
    return fig
