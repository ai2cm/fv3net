import sys
import numpy as np
import os
import xarray as xr

sys.path.insert(0, "..")
from config import *

import serialbox as ser

serializer = ser.Serializer(ser.OpenModeKind.Read, LW_SERIALIZED_DIR, "Init_rank0")

cline = "SURFACE EMISSIVITY INDEX, IDM,JDM: 360 180    NOTE: DATA FROM N TO S"
idxems = serializer.read("idxems", serializer.savepoint["sfc_init_data"])

NCLINE = 80
IMXEMS = 360
JMXEMS = 180

ds = xr.Dataset({"cline": cline, "idxems": (("IMXEMS", "JMXEMS"), idxems)})

dout = os.path.join(FORCING_DIR, "semisdata.nc")

print(ds)

ds.to_netcdf(dout)
