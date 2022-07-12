import sys
import numpy as np
import os
import xarray as xr

sys.path.insert(0, "..")
from config import *

import serialbox as ser

scheme = "LW"
smallscheme = scheme.lower()

# Check files are there
if not os.path.isdir(os.path.join(FORTRANDATA_DIR, scheme)):
    raise FileNotFoundError(
        "Serialized fortran output not found, please download from Google Cloud first"
    )

if not os.path.isdir(LW_SERIALIZED_DIR):
    raise FileNotFoundError(
        "Serialized LW standalone output not found, please download from Google Cloud first"
    )
else:
    if len(os.listdir(LW_SERIALIZED_DIR)) == 0:
        raise FileNotFoundError(
            "Serialized LW standalone output not found, please download from Google Cloud first"
        )

if not os.path.isdir(SW_SERIALIZED_DIR):
    raise FileNotFoundError(
        "Serialized SW standalone output not found, please download from Google Cloud first"
    )
else:
    if len(os.listdir(SW_SERIALIZED_DIR)) == 0:
        raise FileNotFoundError(
            "Serialized SW standalone output not found, please download from Google Cloud first"
        )

for tile in range(6):
    print(f"Rank = {tile}")
    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        os.path.join(FORTRANDATA_DIR, scheme),
        "Generator_rank" + str(tile),
    )

    if scheme == "SW":
        nday = serializer.read(
            "nday", serializer.savepoint[smallscheme + "rad-in-000000"]
        )[0]

        if nday > 0:
            serializer_sw = ser.Serializer(
                ser.OpenModeKind.Read, SW_SERIALIZED_DIR, "Serialized_rank" + str(tile)
            )
            savepoints = serializer_sw.savepoint_list()

            rnlist = list()
            for pt in savepoints:
                if "random_number-output-000000" in pt.name:
                    rnlist.append(pt)

            nlay = 63
            ngptsw = 112
            rand2d = np.zeros((24, nlay * ngptsw))

            for n, sp in enumerate(rnlist):
                tmp = serializer_sw.read("rand2d", sp)
                lat = int(sp.name[-2:]) - 1
                rand2d[lat, :] = tmp

            ds = xr.Dataset({"rand2d": (("iplon", "n"), rand2d)})

            dout = "../lookupdata/rand2d_tile" + str(tile) + "_" + smallscheme + ".nc"
            print(dout)

            ds.to_netcdf(dout)

    elif scheme == "LW":
        serializer_lw = ser.Serializer(
            ser.OpenModeKind.Read, LW_SERIALIZED_DIR, "Serialized_rank" + str(tile)
        )
        savepoints = serializer_lw.savepoint_list()

        rnlist = list()
        for pt in savepoints:
            if "random_number-output-000000" in pt.name:
                rnlist.append(pt)

        nlay = 63
        ngptlw = 140
        rand2d = np.zeros((24, nlay * ngptlw))

        for n, sp in enumerate(rnlist):
            tmp = serializer_lw.read("rand2d", sp)
            lat = int(sp.name[-2:]) - 1
            rand2d[lat, :] = tmp

        ds = xr.Dataset({"rand2d": (("iplon", "n"), rand2d)})

        dout = "../lookupdata/rand2d_tile" + str(tile) + "_" + smallscheme + ".nc"
        print(dout)

        ds.to_netcdf(dout)
