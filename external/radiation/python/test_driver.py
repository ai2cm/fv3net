import numpy as np

from config import *
from util import compare_data
import serialbox as ser
from radiation_driver import RadiationDriver
import time
startTime = time.time()

# defining useful functions
def getscalars(indict):
    for var in indict.keys():
        if not type(indict[var]) == dict:
            if indict[var].size == 1:
                indict[var] = indict[var][0]

    return indict

def read_from_serializer(variables, serializer, in_out = 'in'):
    out = {}
    for var in variables:
        out[var] = serializer.read(var, serializer.savepoint["driver-"+ in_out +"-000000"])
    return out

## Defining variables 
invars = ["idat", "jdat", "fhswr", "dtf", "lsswr"]
statein_vars = [
        "prsi",
        "prsl",
        "tgrs",
        "prslk",
        "qgrs",]
sfcprop_vars = ["tsfc",
        "slmsk",
        "snowd",
        "sncovr",
        "snoalb",
        "zorl",
        "hprime",
        "alvsf",
        "alnsf",
        "alvwf",
        "alnwf",
        "facsf",
        "facwf",
        "fice",
        "tisfc",]
model_vars = [
        "me",
        "levr",
        "levs",
        "nfxr",
        "ntrac",
        "ntcw",
        "ntiw",
        "ncld",
        "ntrw",
        "ntsw",
        "ntgl",
        "ncnd",
        "fhswr",
        "fhlwr",
        "ntoz",
        "lsswr",
        "solhr",
        "lslwr",
        "imp_physics",
        "lgfdlmprad",
        "uni_cld",
        "effr_in",
        "indcld",
        "ntclamt",
        "num_p3d",
        "npdf3d",
        "ncnvcld3d",
        "lmfdeep2",
        "sup",
        "kdt",
        "lmfshal",
        "do_sfcperts",
        "pertalb",
        "do_only_clearsky_rad",
        "swhtr",
        "solcon",
        "lprnt",
        "lwhtr",
        "lssav",]

coupling_vars = ["nirbmdi",
        "nirdfdi",
        "visbmdi",
        "visdfdi",
        "nirbmui",
        "nirdfui",
        "visbmui",
        "visdfui",
        "sfcnsw",
        "sfcdsw",
        "sfcdlw",]

radtend_vars_out = ["upfxc_s_lw",
        "upfx0_s_lw",
        "dnfxc_s_lw",
        "dnfx0_s_lw",
        "upfxc_s_sw",
        "upfx0_s_sw",
        "dnfxc_s_sw",
        "dnfx0_s_sw",
        "sfalb",
        "htrsw",
        "swhc",
        "semis",
        "tsflw",
        "htrlw",
        "lwhc",]

radtend_vars = ["coszen",
        "coszdg",
        "sfalb",
        "htrsw",
        "swhc",
        "lwhc",
        "semis",
        "tsflw",]

grid_vars = [
        "xlon",
        "xlat",
        "sinlat",
        "coslat",]

diag_vars_out = [
        "fluxr",
        "upfxc_t_sw",
        "dnfxc_t_sw",
        "upfx0_t_sw",
        "upfxc_t_lw",
        "upfx0_t_lw",]

tbd_vars = ["phy_f3d","icsdsw","icsdlw"]
diag_vars = ["fluxr"]


isolar = 2  # solar constant control flag
ictmflg = 1  # data ic time/date control flag
ico2flg = 2  # co2 data source control flag
ioznflg = 7  # ozone data source control flag

iaer = 111

if ictmflg == 0 or ictmflg == -2:
    iaerflg = iaer % 100  # no volcanic aerosols for clim hindcast
else:
    iaerflg = iaer % 1000

iaermdl = iaer / 1000  # control flag for aerosol scheme selection
if iaermdl < 0 or iaermdl > 2 and iaermdl != 5:
    print("Error -- IAER flag is incorrect, Abort")

iswcliq = 1  # optical property for liquid clouds for sw
iovrsw = 1  # cloud overlapping control flag for sw
iovrlw = 1  # cloud overlapping control flag for lw
lcrick = False  # control flag for eliminating CRICK
lcnorm = False  # control flag for in-cld condensate
lnoprec = False  # precip effect on radiation flag (ferrier microphysics)
isubcsw = 2  # sub-column cloud approx flag in sw radiation
isubclw = 2  # sub-column cloud approx flag in lw radiation
ialbflg = 1  # surface albedo control flag
iemsflg = 1  # surface emissivity control flag
icldflg = 1
ivflip = 1  # vertical index direction control flag
me = 0

Model_all = dict()
Statein_all = dict()
Sfcprop_all = dict()
Coupling_all = dict()
Grid_all = dict()
Tbd_all = dict()
Radtend_all = dict()
Diag_all = dict()
Valdict_all = dict()

for rank in range(6):
    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        "../fortran/data/radiation_driver",
        "Generator_rank" + str(rank),
    )

    Model_all[rank] = read_from_serializer(model_vars, serializer, in_out='in')
    Statein_all[rank] = read_from_serializer(statein_vars, serializer, in_out='in')
    Sfcprop_all[rank] = read_from_serializer(sfcprop_vars, serializer, in_out='in')
    Coupling_all[rank] = read_from_serializer(coupling_vars,serializer, in_out='in')
    Grid_all[rank] = read_from_serializer(grid_vars,serializer, in_out='in')
    Tbd_all[rank] = read_from_serializer(tbd_vars, serializer, in_out='in')
    Radtend_all[rank] = read_from_serializer(radtend_vars, serializer, in_out='in')
    Diag_all[rank] = read_from_serializer(diag_vars, serializer, in_out='in')
    Valdict_all[rank] = read_from_serializer(radtend_vars_out + diag_vars_out, serializer, in_out='out')

## Run GFS radiation driver


Outdict_all = dict()
for rank in range(6):
########################  Reading data from serialbox ####################################
    if rank == 0:
        serial = ser.Serializer(
            ser.OpenModeKind.Read,
            os.path.join(FORTRANDATA_DIR, "SW"),"Generator_rank" + str(rank))
        si = serial.read("si", serial.savepoint["rad-initialize"])
        imp_physics = serial.read("imp_physics", serial.savepoint["rad-initialize"])

        driver = RadiationDriver()
        driver.radinit(
            si,
            nlay,
            imp_physics,
            me,
            iemsflg,
            ioznflg,
            ictmflg,
            isolar,
            ico2flg,
            iaerflg,
            ialbflg,
            icldflg,
            ivflip,
            iovrsw,
            iovrlw,
            isubcsw,
            isubclw,
            lcrick,
            lcnorm,
            lnoprec,
            iswcliq,
        )

        updatedict = dict()
        for var in invars:
            updatedict[var] = serial.read(var, serial.savepoint["rad-update"])

        slag, sdec, cdec, solcon = driver.radupdate(
            updatedict["idat"],
            updatedict["jdat"],
            updatedict["fhswr"],
            updatedict["dtf"],
            updatedict["lsswr"],)

            
    Radtend = Radtend_all[rank]
    Radtend["sfcfsw"] = dict()
    Radtend["sfcflw"] = dict()

    Diag = Diag_all[rank]
    Diag["topflw"] = dict()
    Diag["topfsw"] = dict()

    Model = getscalars(Model_all[rank])
    Statein = getscalars(Statein_all[rank])
    Sfcprop = getscalars(Sfcprop_all[rank])
    Coupling = getscalars(Coupling_all[rank])
    Grid = getscalars(Grid_all[rank])
    Tbd = getscalars(Tbd_all[rank])
    Radtend = getscalars(Radtend)
    Diag = getscalars(Diag)

    Radtendout, Diagout = driver.GFS_radiation_driver(
        Model, Statein, Sfcprop, Coupling, Grid, Tbd, Radtend, Diag
    )

    # Process output to be compatible with serialized Fortran output for validation
    outdict = dict()
    for var in radtend_vars_out:
        if var[:2] in ["up", "dn"]:
            if var.split("_")[1] == "s":
                if var.split("_")[-1] == "lw":
                    outdict[var] = Radtendout["sfcflw"][var.split("_")[0]]
                else:
                    outdict[var] = Radtendout["sfcfsw"][var.split("_")[0]]
        else:
            outdict[var] = Radtendout[var]

    for var in diag_vars_out:
        if var[:2] in ["up", "dn"]:
            if var[:2] in ["up", "dn"]:
                if var.split("_")[-1] == "lw":
                    outdict[var] = Diagout["topflw"][var.split("_")[0]]
                else:
                    outdict[var] = Diagout["topfsw"][var.split("_")[0]]
        else:
            outdict[var] = Diagout[var]
    Outdict_all[rank] = outdict

## Validation
columns_validated = 0
for rank in range(6):
    compare_data(Valdict_all[rank], Outdict_all[rank])
    columns_validated += Valdict_all[rank][radtend_vars_out[0]].shape[0]

executionTime = (time.time() - startTime)

print(f'Execution time: {executionTime:.2f} seconds for {columns_validated} columns.')
