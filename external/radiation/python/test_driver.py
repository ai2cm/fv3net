from config import FORCING_DIR, LOOKUP_DIR, nlay
from util import compare_data
import serialbox as ser
from radiation_driver import RadiationDriver
import time
import getdata
import variables_to_read

variables = variables_to_read.vars_dict

# defining useful functions


def getscalars(indict):
    for var in indict.keys():
        if not type(indict[var]) == dict:
            if indict[var].size == 1:
                indict[var] = indict[var][0]

    return indict


startTime = time.time()

rank = 0
driver = RadiationDriver()

serial = ser.Serializer(
    ser.OpenModeKind.Read, "../fortran/data/radiation_driver", "Generator_rank0",
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


# reading datasets needed for radinit() and radupdate()
aer_dict = getdata.aerosol(FORCING_DIR)
solar_filename, solar_data = getdata.astronomy(FORCING_DIR, isolar, me)
sfc_file, sfc_data = getdata.sfc(FORCING_DIR)
gas_data = getdata.gases(FORCING_DIR, ictmflg)

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
    aer_dict,
    solar_filename,
    sfc_file,
    sfc_data,
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
    aer_dict["kprfg"],
    aer_dict["idxcg"],
    aer_dict["cmixg"],
    aer_dict["denng"],
    aer_dict["cline"],
    solar_data,
    gas_data,
)


columns_validated = 0

for rank in range(6):

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        "../fortran/data/radiation_driver",
        "Generator_rank" + str(rank),
    )

    Model = dict()
    for var in variables["model"]:
        Model[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    Statein = dict()
    for var in variables["statein"]:
        Statein[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    Sfcprop = dict()
    for var in variables["sfcprop"]:
        Sfcprop[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    Coupling = dict()
    for var in variables["coupling"]:
        Coupling[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    Grid = dict()
    for var in variables["grid"]:
        Grid[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    Tbd = dict()
    for var in variables["tbd"]:
        Tbd[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    Radtend = dict()
    for var in variables["radtend"]:
        Radtend[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    Radtend["sfcfsw"] = dict()
    Radtend["sfcflw"] = dict()

    Diag = dict()
    for var in variables["diag"]:
        Diag[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

    Diag["topflw"] = dict()
    Diag["topfsw"] = dict()

    Model = getscalars(Model)
    Statein = getscalars(Statein)
    Sfcprop = getscalars(Sfcprop)
    Coupling = getscalars(Coupling)
    Grid = getscalars(Grid)
    Tbd = getscalars(Tbd)
    Radtend = getscalars(Radtend)
    Diag = getscalars(Diag)
    randomdict = getdata.random_numbers(LOOKUP_DIR, rank)
    lwdict = getdata.lw(LOOKUP_DIR)
    swdict = getdata.sw(LOOKUP_DIR)

    Radtendout, Diagout, Couplingout = driver.GFS_radiation_driver(
        Model,
        Statein,
        Sfcprop,
        Coupling,
        Grid,
        Tbd,
        Radtend,
        Diag,
        randomdict,
        lwdict,
        swdict,
    )

    # Process output to be compatible with serialized Fortran output for validation
    valdict = dict()
    outdict = dict()

    for var in variables["radtend_out"]:
        valdict[var] = serializer.read(var, serializer.savepoint["driver-out-000000"])
        if var[:2] in ["up", "dn"]:
            if var.split("_")[1] == "s":
                if var.split("_")[-1] == "lw":
                    outdict[var] = Radtendout["sfcflw"][var.split("_")[0]]
                else:
                    outdict[var] = Radtendout["sfcfsw"][var.split("_")[0]]
        else:
            outdict[var] = Radtendout[var]

    for var in variables["diag_out"]:
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

    columns_validated += valdict[variables["radtend_out"][0]].shape[0]

executionTime = time.time() - startTime

print(f"Execution time: {executionTime:.2f} seconds for {columns_validated} columns.")
