import numpy as np
import sys
import os
import time

sys.path.insert(0, "..")
from config import *
from radlw_main import RadLWClass
from util import compare_data

import serialbox as ser

serializer = ser.Serializer(
    ser.OpenModeKind.Read, os.path.join(FORTRANDATA_DIR, "LW"), "Generator_rank0"
)
savepoints = serializer.savepoint_list()

invars = [
    "plyr",
    "plvl",
    "tlyr",
    "tlvl",
    "qlyr",
    "olyr",
    "gasvmr",
    "clouds",
    "icsdlw",
    "faerlw",
    "semis",
    "tsfg",
    "dz",
    "delp",
    "de_lgth",
    "im",
    "lmk",
    "lmp",
    "lprnt",
]

indict = dict()
for var in invars:
    indict[var] = serializer.read(var, serializer.savepoint["lwrad-in-000000"])

outvars = [
    "htlwc",
    "upfxc_t",
    "upfx0_t",
    "upfxc_s",
    "upfx0_s",
    "dnfxc_s",
    "dnfx0_s",
    "cldtaulw",
    "htlw0",
]

outdict = dict()
for var in outvars:
    if var == "htlwc" or var == "cldtau" or var == "htlw0":
        outdict[var] = serializer.read(var, serializer.savepoint["lwrad-out-000000"])
    else:
        outdict[var] = serializer.read(var, serializer.savepoint["lwrad-out-000000"])

indict["lhlwb"] = False
indict["lhlw0"] = True
indict["lflxprf"] = False

me = 0
iovrlw = 1
isubclw = 2

lw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile0_lw.nc")

rlw = RadLWClass(me, iovrlw, isubclw)


start = time.time()
(hlwc, upfxc_t, upfx0_t, upfxc_s, upfx0_s, dnfxc_s, dnfx0_s, cldtau, hlw0) = rlw.lwrad(
    indict["plyr"],
    indict["plvl"],
    indict["tlyr"],
    indict["tlvl"],
    indict["qlyr"],
    indict["olyr"],
    indict["gasvmr"],
    indict["clouds"],
    indict["icsdlw"],
    indict["faerlw"],
    indict["semis"],
    indict["tsfg"],
    indict["dz"],
    indict["delp"],
    indict["de_lgth"],
    indict["im"][0],
    indict["lmk"][0],
    indict["lmp"][0],
    indict["lprnt"],
    indict["lhlwb"],
    indict["lhlw0"],
    indict["lflxprf"],
    lw_rand_file,
)
end = time.time()
print(f"Total elapsed time = {end-start}")

outdict_val = {
    "htlwc": hlwc,
    "upfxc_t": upfxc_t,
    "upfx0_t": upfx0_t,
    "upfxc_s": upfxc_s,
    "upfx0_s": upfx0_s,
    "dnfxc_s": dnfxc_s,
    "dnfx0_s": dnfx0_s,
    "cldtaulw": cldtau,
    "htlw0": hlw0,
}

print("Comparing")
compare_data(outdict, outdict_val)
print("Done")
