import numpy as np

from config import *
from util import compare_data
import serialbox as ser
from radiation_driver import RadiationDriver

rank = 0
driver = RadiationDriver()

serial = ser.Serializer(
    ser.OpenModeKind.Read,
    os.path.join(FORTRANDATA_DIR, "SW"),
    "Generator_rank0",
)

si = serial.read("si", serial.savepoint["rad-initialize"])
imp_physics = serial.read("imp_physics", serial.savepoint["rad-initialize"])

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

invars = ["idat", "jdat", "fhswr", "dtf", "lsswr"]
updatedict = dict()

for var in invars:
    updatedict[var] = serial.read(var, serial.savepoint["rad-update"])

slag, sdec, cdec, solcon = driver.radupdate(
    updatedict["idat"],
    updatedict["jdat"],
    updatedict["fhswr"],
    updatedict["dtf"],
    updatedict["lsswr"],
)

for rank in range(6):
    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        "../fortran/data/radiation_driver",
        "Generator_rank" + str(rank),
    )

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
        "lssav",
    ]

    Model = dict()
    for var in model_vars:
        Model[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    statein_vars = [
        "prsi",
        "prsl",
        "tgrs",
        "prslk",
        "qgrs",
    ]

    Statein = dict()
    for var in statein_vars:
        Statein[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    sfcprop_vars = [
        "tsfc",
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
        "tisfc",
    ]

    Sfcprop = dict()
    for var in sfcprop_vars:
        Sfcprop[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    coupling_vars = [
        "nirbmdi",
        "nirdfdi",
        "visbmdi",
        "visdfdi",
        "nirbmui",
        "nirdfui",
        "visbmui",
        "visdfui",
        "sfcnsw",
        "sfcdsw",
        "sfcdlw",
    ]

    Coupling = dict()
    for var in coupling_vars:
        Coupling[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    grid_vars = [
        "xlon",
        "xlat",
        "sinlat",
        "coslat",
    ]

    Grid = dict()
    for var in grid_vars:
        Grid[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    tbd_vars = [
        "phy_f3d",
        "icsdsw",
        "icsdlw",
    ]

    Tbd = dict()
    for var in tbd_vars:
        Tbd[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    radtend_vars = [
        "coszen",
        "coszdg",
        "sfalb",
        "htrsw",
        "swhc",
        "lwhc",
        "semis",
        "tsflw",
    ]

    Radtend = dict()
    for var in radtend_vars:
        Radtend[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    Radtend["sfcfsw"] = dict()
    Radtend["sfcflw"] = dict()

    diag_vars = ["fluxr"]
    Diag = dict()
    for var in diag_vars:
        Diag[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    Diag["topflw"] = dict()
    Diag["topfsw"] = dict()

    def getscalars(indict):
        for var in indict.keys():
            if not type(indict[var]) == dict:
                if indict[var].size == 1:
                    indict[var] = indict[var][0]

        return indict

    Model = getscalars(Model)
    Statein = getscalars(Statein)
    Sfcprop = getscalars(Sfcprop)
    Coupling = getscalars(Coupling)
    Grid = getscalars(Grid)
    Tbd = getscalars(Tbd)
    Radtend = getscalars(Radtend)
    Diag = getscalars(Diag)

    Radtendout, Diagout = driver.GFS_radiation_driver(
        Model, Statein, Sfcprop, Coupling, Grid, Tbd, Radtend, Diag
    )

    radtend_vars_out = [
        "upfxc_s_lw",
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
        "lwhc",
    ]

    diag_vars_out = [
        "fluxr",
        "upfxc_t_sw",
        "dnfxc_t_sw",
        "upfx0_t_sw",
        "upfxc_t_lw",
        "upfx0_t_lw",
    ]

    # Process output to be compatible with serialized Fortran output for validation
    valdict = dict()
    outdict = dict()

    for var in radtend_vars_out:
        valdict[var] = serializer.read(var, serializer.savepoint["driver-out-000000"])
        if var[:2] in ["up", "dn"]:
            if var.split("_")[1] == "s":
                if var.split("_")[-1] == "lw":
                    outdict[var] = Radtendout["sfcflw"][var.split("_")[0]]
                else:
                    outdict[var] = Radtendout["sfcfsw"][var.split("_")[0]]
        else:
            outdict[var] = Radtendout[var]

    for var in diag_vars_out:
        valdict[var] = serializer.read(var, serializer.savepoint["driver-out-000000"])

        if var[:2] in ["up", "dn"]:
            if var[:2] in ["up", "dn"]:
                if var.split("_")[-1] == "lw":
                    outdict[var] = Diagout["topflw"][var.split("_")[0]]
                else:
                    outdict[var] = Diagout["topfsw"][var.split("_")[0]]
        else:
            outdict[var] = Diagout[var]

    compare_data(valdict, outdict)
