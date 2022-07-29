import sys

sys.path.insert(0, "..")
import os

from radiation_driver import RadiationDriver
from radphysparam import icldflg
from util import compare_data
from config import *

import serialbox as ser

serializer = ser.Serializer(
    ser.OpenModeKind.Read, os.path.join(FORTRANDATA_DIR, "LW"), "Generator_rank0"
)
serializer2 = ser.Serializer(
    ser.OpenModeKind.Read, LW_SERIALIZED_DIR, "Serialized_rank0"
)
serializer3 = ser.Serializer(ser.OpenModeKind.Read, LW_SERIALIZED_DIR, "Init_rank0")
savepoints = serializer.savepoint_list()
savepoints2 = serializer2.savepoint_list()
savepoints3 = serializer3.savepoint_list()

invars = [
    "si",
    "levr",
    "ictm",
    "isol",
    "ico2",
    "iaer",
    "ialb",
    "iems",
    "ntcw",
    "num_p2d",
    "num_p3d",
    "npdf3d",
    "ntoz",
    "iovr_sw",
    "iovr_lw",
    "isubc_sw",
    "isubc_lw",
    "icliq_sw",
    "crick_proof",
    "ccnorm",
    "imp_physics",
    "norad_precip",
    "idate",
    "iflip",
    "me",
]

initdict = dict()

for var in invars:
    if var != "levr" and var != "me":
        initdict[var] = serializer.read(var, serializer.savepoint["rad-initialize"])
    elif var == "levr":
        initdict[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

initdict["me"] = 0
initdict["icld"] = icldflg

invars = ["idat", "jdat", "fhswr", "dtf", "lsswr"]

indict = {}
for var in invars:
    indict[var] = serializer.read(var, serializer.savepoint["rad-update"])

driver = RadiationDriver()
driver.radinit(
    initdict["si"],
    initdict["levr"][0],
    initdict["imp_physics"][0],
    initdict["me"],
    initdict["iems"][0],
    initdict["ntoz"][0],
    initdict["ictm"][0],
    initdict["isol"][0],
    initdict["ico2"][0],
    initdict["iaer"][0],
    initdict["ialb"][0],
    initdict["icld"],
    initdict["iflip"][0],
    initdict["iovr_sw"][0],
    initdict["iovr_lw"][0],
    initdict["isubc_sw"][0],
    initdict["isubc_lw"][0],
    initdict["crick_proof"][0],
    initdict["ccnorm"][0],
    initdict["norad_precip"][0],
    initdict["icliq_sw"][0],
    do_test=False,
)

print(" ")
print("Running update")
soldict, aerdict, gasdict = driver.radupdate(
    indict["idat"],
    indict["jdat"],
    indict["fhswr"],
    indict["dtf"],
    indict["lsswr"],
    do_test=True,
)

solvars = ["slag", "sdec", "cdec", "solcon"]
aervars = ["kprfg", "idxcg", "cmixg", "denng", "ivolae"]
gasvars = ["co2vmr_sav", "gco2cyc"]

soldict_val = dict()
for var in solvars:
    soldict_val[var] = serializer3.read(
        var, serializer3.savepoint["lw_sol_update_out000000"]
    )

aerdict_val = dict()
for var in aervars:
    aerdict_val[var] = serializer3.read(
        var, serializer3.savepoint["lw_aer_update_out000000"]
    )

gasdict_val = dict()
for var in gasvars:
    gasdict_val[var] = serializer3.read(
        var, serializer3.savepoint["lw_gas_update_out000000"]
    )

compare_data(soldict, soldict_val)
compare_data(aerdict, aerdict_val)
compare_data(gasdict, gasdict_val)
