import numpy as np
import xarray as xr

import sys

sys.path.insert(0, "..")
from config import *

dlist = list()

with open(os.path.join(FORCING_DIR, "co2historicaldata_2016.txt"), "r") as f:
    x = f.readline()
    co2g1 = float(x.split(",")[1].split()[-1])
    co2g2 = float(x.split(",")[2].split()[-1])
    cline = x[:99]
    iyr = int(x[:4])
    while x != "":
        x = f.readline()
        dlist.append(x)
        print(x)

dlist = dlist[:-1]

darray = np.zeros((24, 144))

for n, row in enumerate(dlist):
    darray[:, n] = np.array([float(n) for n in row.split()])

darr = np.reshape(darray, (24, 12, 12), order="F")
print(darr[:, -2, -1])

print(cline + str(co2g1))
print(iyr)

ds = xr.Dataset(
    {
        "co2dat": (("I", "J", "month"), darr),
        "co2g1": co2g1,
        "co2g2": co2g2,
        "cline": cline,
        "iyr": iyr,
    }
)

print(ds)

dout = os.path.join(FORCING_DIR, "co2historicaldata_2016.nc")

ds.to_netcdf(dout)
