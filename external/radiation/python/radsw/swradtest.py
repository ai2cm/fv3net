import numpy as np
import sys
import os
import time

sys.path.insert(0, "..")
from config import *
from radsw_main import RadSWClass
from util import compare_data

import serialbox as ser

serializer = ser.Serializer(
    ser.OpenModeKind.Read, os.path.join(FORTRANDATA_DIR, "SW"), "Generator_rank1"
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
    "icsdsw",
    "faersw",
    "sfcalb",
    "dz",
    "delp",
    "de_lgth",
    "coszen",
    "solcon",
    "nday",
    "idxday",
    "im",
    "lmk",
    "lmp",
    "lprnt",
]

indict = dict()
for var in invars:
    indict[var] = serializer.read(var, serializer.savepoint["swrad-in-000000"])

indict["lhswb"] = False
indict["lhsw0"] = True
indict["lflxprf"] = False
indict["lfdncmp"] = True

me = 0
iovrsw = 1
isubcsw = 2
icldflg = 1
sw_rand_file = os.path.join(LOOKUP_DIR, "rand2d_tile1_sw.nc")

radsw = RadSWClass(me, iovrsw, isubcsw, icldflg)

(
    hswc,
    upfxc_t,
    dnfxc_t,
    upfx0_t,
    upfxc_s,
    dnfxc_s,
    upfx0_s,
    dnfx0_s,
    cldtau,
    hsw0,
    uvbf0,
    uvbfc,
    nirbm,
    nirdf,
    visbm,
    visdf,
) = radsw.swrad(
    indict["plyr"],
    indict["plvl"],
    indict["tlyr"],
    indict["tlvl"],
    indict["qlyr"],
    indict["olyr"],
    indict["gasvmr"],
    indict["clouds"],
    indict["icsdsw"],
    indict["faersw"],
    indict["sfcalb"],
    indict["dz"],
    indict["delp"],
    indict["de_lgth"],
    indict["coszen"],
    indict["solcon"],
    indict["nday"][0],
    indict["idxday"],
    indict["im"][0],
    indict["lmk"][0],
    indict["lmp"][0],
    indict["lprnt"],
    indict["lhswb"],
    indict["lhsw0"],
    indict["lflxprf"],
    indict["lfdncmp"],
    sw_rand_file,
)

outvars = [
    "htswc",
    "upfxc_t",
    "dnfxc_t",
    "upfx0_t",
    "upfxc_s",
    "dnfxc_s",
    "upfx0_s",
    "dnfx0_s",
    "cldtausw",
    "htsw0",
    "uvbf0",
    "uvbfc",
    "nirbm",
    "nirdf",
    "visbm",
    "visdf",
]

outdict = dict()

outdict["htswc"] = hswc
outdict["upfxc_t"] = upfxc_t
outdict["dnfxc_t"] = dnfxc_t
outdict["upfx0_t"] = upfx0_t
outdict["upfxc_s"] = upfxc_s
outdict["dnfxc_s"] = dnfxc_s
outdict["upfx0_s"] = upfx0_s
outdict["dnfx0_s"] = dnfx0_s
outdict["cldtausw"] = cldtau
outdict["htsw0"] = hsw0
outdict["uvbf0"] = uvbf0
outdict["uvbfc"] = uvbfc
outdict["nirbm"] = nirbm
outdict["nirdf"] = nirdf
outdict["visbm"] = visbm
outdict["visdf"] = visdf

valdict = dict()
for var in outvars:
    valdict[var] = serializer.read(var, serializer.savepoint["swrad-out-000000"])

compare_data(outdict, valdict)
