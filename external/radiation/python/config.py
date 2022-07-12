import numpy as np
import sys
import os
import gt4py

IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else False
IS_TEST = (os.getenv("IS_TEST") == "True") if ("IS_TEST" in os.environ) else False
backend = (os.getenv("BACKEND")) if ("BACKEND" in os.environ) else "gtc:gt:cpu_ifirst"

if IS_DOCKER:
    if IS_TEST:
        sys.path.insert(0, "/deployed/radiation/python")
    else:
        sys.path.insert(0, "/work/radiation/python")
else:
    sys.path.insert(
        0, "/Users/andrewp/Documents/work/physics_standalone/radiation/python"
    )
from radlw.radlw_param import nbands, maxgas, maxxsec, ngptlw, nrates
from radsw.radsw_param import ngptsw, nbandssw, nbdsw, ntbmx
from gt4py import gtscript
from gt4py.gtscript import Field

gt4py.config.build_settings["extra_compile_args"]["cxx"].extend(
    ["-fno-strict-aliasing"]
)
if IS_DOCKER:
    SERIALBOX_DIR = "/usr/local/serialbox"
    if IS_TEST:
        LOOKUP_DIR = "/deployed/radiation/python/lookupdata"
        FORTRANDATA_DIR = "/deployed/radiation/fortran/data"
        LW_SERIALIZED_DIR = "/deployed/radiation/fortran/radlw/dump"
        SW_SERIALIZED_DIR = "/deployed/radiation/fortran/radsw/dump"
    else:
        LOOKUP_DIR = "/work/radiation/python/lookupdata"
        FORTRANDATA_DIR = "/work/radiation/fortran/data"
        LW_SERIALIZED_DIR = "/work/radiation/fortran/radlw/dump"
        SW_SERIALIZED_DIR = "/work/radiation/fortran/radsw/dump"
        FORCING_DIR = "/work/radiation/python/forcing"
else:
    SERIALBOX_DIR = "/usr/local/serialbox"
    LOOKUP_DIR = "./lookupdata"
    FORTRANDATA_DIR = "../fortran/data"
    LW_SERIALIZED_DIR = "../fortran/radlw/dump"
    SW_SERIALIZED_DIR = "../fortran/radsw/dump"
    FORCING_DIR = "./forcing"

backend = "gtc:gt:cpu_ifirst"

sys.path.append(SERIALBOX_DIR + "/python")

npts = 24

nlay = 63
nlp1 = 64

ilwrgas = 1
ilwcliq = 1
isubclw = 2

DTYPE_INT = np.int32
DTYPE_FLT = np.float64
DTYPE_BOOL = bool
FIELD_INT = Field[DTYPE_INT]
FIELD_FLT = Field[DTYPE_FLT]
FIELD_BOOL = Field[DTYPE_BOOL]
FIELD_2D = Field[gtscript.IJ, DTYPE_FLT]
FIELD_2DINT = Field[gtscript.IJ, DTYPE_INT]
FIELD_2DBOOL = Field[gtscript.IJ, DTYPE_BOOL]

shape = (npts, 1, 1)
shape_2D = (npts, 1)
shape_nlay = (npts, 1, nlay)
shape_nlp1 = (npts, 1, nlp1)
shape_nlp2 = (npts, 1, nlp1 + 1)
default_origin = (0, 0, 0)

type_nbands = (DTYPE_FLT, (nbands,))
type_nbandssw_int = (DTYPE_INT, (nbandssw,))
type_nbandssw_flt = (DTYPE_FLT, (nbandssw,))
type_nbandssw3 = (DTYPE_FLT, (nbandssw, 3))
type_ngptlw = (DTYPE_FLT, (ngptlw,))
type_ngptsw = (DTYPE_FLT, (ngptsw,))
type_ngptsw_bool = (DTYPE_BOOL, (ngptsw,))
type_nbands3 = (DTYPE_FLT, (nbands, 3))
type_maxgas = (DTYPE_FLT, (maxgas,))
type_maxxsec = (DTYPE_FLT, (maxxsec,))
type_nrates = (DTYPE_FLT, (nrates, 2))
type_nbdsw = (DTYPE_FLT, (nbdsw,))
type_ntbmx = (DTYPE_FLT, ((ntbmx + 1),))
type_9 = (DTYPE_FLT, (9,))
type_10 = (DTYPE_FLT, (10,))
