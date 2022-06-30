#Yakelyn modified from several scripts in github 
import sys
sys.path.append("/home/orca/yakelynr/working-directory/mypy/")
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

import numpy as np                       # Import the Numpy package
import colorsys

def WhBlGrYlOrRd():
    cmapList = ['#f2eded','#ede8e8','#d6d4d4','#bab6b6','#adabab',
    '#b9e3f8','#459ce3','#2a7cdf' ,'#1d66bf', '#2144de',
    '#9be396','#60d558','#36b42d','#369d2f', '#2f8929',
    '#ecee96', '#e5e773', '#e1e359', '#d9dc2e', '#cacd23',
    '#ffa229','#ff950a','#eb8500','#eb7100','#eb6200',
    '#ff6b90','#ff5c85','#ff4271','#ff384c','#ff1f35',
    '#e60017','#cc0014','#ad0011','#80000d','#61000a']
        
    cmap = ListedColormap(cmapList, cmapList)
    cmap.set_over('#61000a')
    cmap.set_under('#ffffff')
    return cmap
        
def precip_colors():
    cmapList = ["#ffffff", "#d7e0e0","#bcdfdf","#96e3e3", "#6cebe9","#04e9e7", "#019ff4", "#0300f4", "#02fd02",  
    "#01c501","#008e00","#fdf802","#e5bc00","#fd9500", "#fd0000", "#d40000", "#bc0000","#f800fd"]
    cmap = ListedColormap(cmapList,cmapList)
    return cmap

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

def blues2reds2():
    rgbset2 = np.array([[25,25,112,256],
    [0,0,205,256],
    [0,57,197,256],
    [0,79,226,256],
    [0,102,255,256],
    [77,148,255,256],
    [0,255,255,256],
    [122,215,240,256],
    [0,148,102,256],
    [0,204,153,256], # 13
    [66,233,180,256],
    [102,255,207,256],
    [228,255,255,256],
    
    [255,250,170,256],
    [255,255,51,256],
    [255,232,120,256],
    [255,192,60,256],
    [255,160,0,256],
    [255,96,0,256],
    [255,57,57,256],
    [227,0,34,256],
    [255,0,0,256],
    [187,0,0,256],
    [194,0,11,256],
    [150,0,24,256],
    [101,0,11,256]])
    newcmp = ListedColormap(rgbset2/256)
    return newcmp


def purple2reds():
    cmm1 = plt.cm.jet(np.linspace(0.65, 1, 12))
    cmm2 = plt.cm.gist_rainbow(np.linspace(0.55,0.95, 12))
    cmap = ListedColormap(np.concatenate((cmm2[::-1],cmm1),axis = 0))  
    return cmap

def purples2darkred2():
    cm1 = plt.cm.hot(np.linspace(0, 0.8, 25))
    cm1[24 ,:] = 0.
    cm1[23 ,:] = [1,1,0.8,1]
    cm2 = plt.cm.hsv(np.linspace(0.5, 0.8, 20))
    cm2[1 ,:] = [0.898039,0.992156,1,1]
    cm2[0 ,:] = [0.8,0.9843,1,1]
    cmap = ListedColormap(np.concatenate((cm2[::-1],cm1[::-1]),axis =0)) 
    return cmap

def salinity():
    cm1 = plt.cm.hot(np.linspace(0,0.94, 10)) 
    cm2 = plt.cm.winter(np.linspace(0, 1, 10)) 
    cmap2 = ListedColormap(np.concatenate((cm2,cm1[::-1]),axis =0)) 
    return cmap2

def temperature():
    cm1 = plt.cm.hot(np.linspace(0.1, 0.8, 20))
    cm2 = plt.cm.hsv(np.linspace(0.48, 0.8, 20))
    cmap = ListedColormap(np.concatenate((cm2[::-1],cm1[::-1]),axis =0)) 
    return cmap
