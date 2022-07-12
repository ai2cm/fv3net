import numpy as np
import xarray as xr
import sys
import time
import warnings

sys.path.insert(0, "..")
from radphysparam import (
    ilwrgas as ilwrgas,
    icldflg as icldflg,
    ilwcliq as ilwcliq,
    ilwrate as ilwrate,
    ilwcice as ilwcice,
)
from radlw.radlw_param import *
from phys_const import con_g, con_cp, con_amd, con_amw, con_amo3
from util import (
    create_storage_from_array,
    loadlookupdata,
    compare_data,
    read_data,
    read_intermediate_data,
    numpy_dict_to_gt4py_dict,
    create_gt4py_dict_zeros,
    convert_gt4py_output_for_validation,
)
from config import *
from stencils_gt4py import *

import serialbox as ser


class RadLWClass:
    VTAGLW = "NCEP LW v5.1  Nov 2012 -RRTMG-LW v4.82"
    expeps = 1.0e-20

    bpade = 1.0 / 0.278
    eps = 1.0e-6
    oneminus = 1.0 - eps
    cldmin = 1.0e-80
    stpfac = 296.0 / 1013.0
    wtdiff = 0.5
    tblint = ntbl

    ipsdlw0 = ngptlw

    amdw = con_amd / con_amw
    amdo3 = con_amd / con_amo3

    nspa = [1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9]
    nspb = [1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]

    delwave = np.array(
        [
            340.0,
            150.0,
            130.0,
            70.0,
            120.0,
            160.0,
            100.0,
            100.0,
            210.0,
            90.0,
            320.0,
            280.0,
            170.0,
            130.0,
            220.0,
            650.0,
        ]
    )

    a0 = [
        1.66,
        1.55,
        1.58,
        1.66,
        1.54,
        1.454,
        1.89,
        1.33,
        1.668,
        1.66,
        1.66,
        1.66,
        1.66,
        1.66,
        1.66,
        1.66,
    ]
    a1 = [
        0.00,
        0.25,
        0.22,
        0.00,
        0.13,
        0.446,
        -0.10,
        0.40,
        -0.006,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]
    a2 = [
        0.00,
        -12.0,
        -11.7,
        0.00,
        -0.72,
        -0.243,
        0.19,
        -0.062,
        0.414,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]

    A0 = create_storage_from_array(a0, backend, shape_nlp1, type_nbands)
    A1 = create_storage_from_array(a1, backend, shape_nlp1, type_nbands)
    A2 = create_storage_from_array(a2, backend, shape_nlp1, type_nbands)

    NGB = create_storage_from_array(
        np.tile(np.array(ngb)[None, None, :], (npts, 1, 1)),
        backend,
        shape_2D,
        (DTYPE_INT, (ngptlw,)),
        default_origin=(0, 0),
    )

    def __init__(self, me, iovrlw, isubclw):
        """Initialize the LW scheme

        Args:
            me (int): Current rank, used as print flag
            iovrlw (int): control flag for cloud overlapping method.
                =0: random
                =1: maximum/random
                =2: maximum
                =3: decorr
            isubclw (int): sub-column cloud approximation control flag
                =0: no sub-col cld treatment, use grid-mean cld quantities
                =1: mcica sub-col, prescribed seeds to get random numbers
                =2: mcica sub-col, providing array icseed for random numbers
        """
        self.lhlwb = False
        self.lhlw0 = False
        self.lflxprf = False

        self.semiss0 = np.ones(nbands)

        self.iovrlw = iovrlw
        self.isubclw = isubclw

        self.exp_tbl = np.zeros(ntbl + 1)
        self.tau_tbl = np.zeros(ntbl + 1)
        self.tfn_tbl = np.zeros(ntbl + 1)

        expeps = 1e-20

        if self.iovrlw < 0 or self.iovrlw > 3:
            raise ValueError(
                f"  *** Error in specification of cloud overlap flag",
                f" IOVRLW={self.iovrlw}, in RLWINIT !!",
            )
        elif self.iovrlw >= 2 and self.isubclw == 0:
            if me == 0:
                warnings.warn(
                    f"  *** IOVRLW={self.iovrlw} is not available for",
                    " ISUBCLW=0 setting!!",
                )
                warnings.warn("      The program uses maximum/random overlap instead.")
            self.iovrlw = 1

        if me == 0:
            print(f"- Using AER Longwave Radiation, Version: {self.VTAGLW}")

            if ilwrgas > 0:
                print(
                    "   --- Include rare gases N2O, CH4, O2, CFCs ", "absorptions in LW"
                )
            else:
                print("   --- Rare gases effect is NOT included in LW")

            if self.isubclw == 0:
                print(
                    "   --- Using standard grid average clouds, no ",
                    "   sub-column clouds approximation applied",
                )
            elif self.isubclw == 1:
                print(
                    "   --- Using MCICA sub-colum clouds approximation ",
                    "   with a prescribed sequence of permutaion seeds",
                )
            elif self.isubclw == 2:
                print(
                    "   --- Using MCICA sub-colum clouds approximation ",
                    "   with provided input array of permutation seeds",
                )
            else:
                raise ValueError(
                    f"  *** Error in specification of sub-column cloud ",
                    f" control flag isubclw = {self.isubclw}!!",
                )

        #  --- ...  check cloud flags for consistency

        if (icldflg == 0 and ilwcliq != 0) or (icldflg == 1 and ilwcliq == 0):
            raise ValueError(
                "*** Model cloud scheme inconsistent with LW",
                "radiation cloud radiative property setup !!",
            )

        #  --- ...  setup constant factors for flux and heating rate
        #           the 1.0e-2 is to convert pressure from mb to N/m**2

        pival = 2.0 * np.arcsin(1.0)
        self.fluxfac = pival * 2.0e4

        if ilwrate == 1:
            self.heatfac = con_g * 864.0 / con_cp  #   (in k/day)
        else:
            self.heatfac = con_g * 1.0e-2 / con_cp  #   (in k/second)

        #  --- ...  compute lookup tables for transmittance, tau transition
        #           function, and clear sky tau (for the cloudy sky radiative
        #           transfer).  tau is computed as a function of the tau
        #           transition function, transmittance is calculated as a
        #           function of tau, and the tau transition function is
        #           calculated using the linear in tau formulation at values of
        #           tau above 0.01.  tf is approximated as tau/6 for tau < 0.01.
        #           all tables are computed at intervals of 0.001.  the inverse
        #           of the constant used in the pade approximation to the tau
        #           transition function is set to b.

        self.tau_tbl[0] = 0.0
        self.exp_tbl[0] = 1.0
        self.tfn_tbl[0] = 0.0

        self.tau_tbl[ntbl] = 1.0e10
        self.exp_tbl[ntbl] = expeps
        self.tfn_tbl[ntbl] = 1.0

        explimit = int(np.floor(-np.log(np.finfo(float).tiny)))

        for i in range(1, ntbl):
            tfn = (i) / (ntbl - i)
            self.tau_tbl[i] = self.bpade * tfn
            if self.tau_tbl[i] >= explimit:
                self.exp_tbl[i] = expeps
            else:
                self.exp_tbl[i] = np.exp(-self.tau_tbl[i])

            if self.tau_tbl[i] < 0.06:
                self.tfn_tbl[i] = self.tau_tbl[i] / 6.0
            else:
                self.tfn_tbl[i] = 1.0 - 2.0 * (
                    (1.0 / self.tau_tbl[i])
                    - (self.exp_tbl[i] / (1.0 - self.exp_tbl[i]))
                )

        self.exp_tbl = np.tile(self.exp_tbl[None, None, None, :], (npts, 1, nlp1, 1))
        self.tau_tbl = np.tile(self.tau_tbl[None, None, None, :], (npts, 1, nlp1, 1))
        self.tfn_tbl = np.tile(self.tfn_tbl[None, None, None, :], (npts, 1, nlp1, 1))

        self.exp_tbl = create_storage_from_array(
            self.exp_tbl, backend, shape_nlp1, type_ntbmx
        )
        self.tau_tbl = create_storage_from_array(
            self.tau_tbl, backend, shape_nlp1, type_ntbmx
        )
        self.tfn_tbl = create_storage_from_array(
            self.tfn_tbl, backend, shape_nlp1, type_ntbmx
        )

        self._load_lookup_table_data()

    def return_initdata(self):
        """
        Return output of init routine for validation against Fortran
        """

        outdict = {
            "semiss0": self.semiss0,
            "fluxfac": self.fluxfac,
            "heatfac": self.heatfac,
            "exp_tbl": self.exp_tbl,
            "tau_tbl": self.tau_tbl,
            "tfn_tbl": self.tfn_tbl,
        }
        return outdict

    def create_input_data(self, rank):
        """
        Load input data from serialized Fortran model output and transform into
        gt4py storages. Also creates the necessary local variables as gt4py storages
        """

        self.serializer2 = ser.Serializer(
            ser.OpenModeKind.Read, LW_SERIALIZED_DIR, "Serialized_rank" + str(rank)
        )

        invars = {
            "plyr": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "plvl": {"shape": (npts, nlp1), "type": DTYPE_FLT},
            "tlyr": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "tlvl": {"shape": (npts, nlp1), "type": DTYPE_FLT},
            "qlyr": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "olyr": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "gasvmr": {"shape": (npts, nlay, 10), "type": type_10},
            "clouds": {"shape": (npts, nlay, 9), "type": type_9},
            "icsdlw": {"shape": (npts,), "type": DTYPE_INT},
            "faerlw": {"shape": (npts, nlay, nbands, 3), "type": type_nbands3},
            "semis": {"shape": (npts,), "type": DTYPE_FLT},
            "tsfg": {"shape": (npts,), "type": DTYPE_FLT},
            "dz": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "delp": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "de_lgth": {"shape": (npts,), "type": DTYPE_FLT},
            "im": {"shape": (), "type": DTYPE_INT},
            "lmk": {"shape": (), "type": DTYPE_INT},
            "lmp": {"shape": (), "type": DTYPE_INT},
            "lprnt": {"shape": (), "type": DTYPE_BOOL},
        }

        indict = read_data(
            os.path.join(FORTRANDATA_DIR, "LW"), "lwrad", rank, 0, True, invars
        )
        indict_gt4py = numpy_dict_to_gt4py_dict(indict, invars)

        outvars = {
            "htlwc": {
                "shape": shape_nlp1,
                "type": DTYPE_FLT,
                "fortran_shape": (npts, nlay),
            },
            "htlw0": {
                "shape": shape_nlp1,
                "type": DTYPE_FLT,
                "fortran_shape": (npts, nlay),
            },
            "cldtaulw": {
                "shape": shape_nlp1,
                "type": DTYPE_FLT,
                "fortran_shape": (npts, nlay),
            },
            "upfxc_t": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "upfx0_t": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "upfxc_s": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "upfx0_s": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "dnfxc_s": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "dnfx0_s": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
        }

        outdict_gt4py = create_gt4py_dict_zeros(outvars)

        locvars = {
            "cldfrc": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "totuflux": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "totdflux": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "totuclfl": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "totdclfl": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tz": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "htr": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "htrb": {"shape": shape_nlp1, "type": type_nbands},
            "htrcl": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "pavel": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tavel": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "delp": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "clwp": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ciwp": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "relw": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "reiw": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cda1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cda2": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cda3": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cda4": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "coldry": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "colbrd": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "h2ovmr": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "o3vmr": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac00": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac01": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac10": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac11": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "selffac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "selffrac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "forfac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "forfrac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "minorfrac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "scaleminor": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "scaleminorn2": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "temcol": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "dz": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "pklev": {"shape": shape_nlp1, "type": type_nbands},
            "pklay": {"shape": shape_nlp1, "type": type_nbands},
            "taucld": {"shape": shape_nlp1, "type": type_nbands},
            "tauaer": {"shape": shape_nlp1, "type": type_nbands},
            "fracs": {"shape": shape_nlp1, "type": type_ngptlw},
            "tautot": {"shape": shape_nlp1, "type": type_ngptlw},
            "cldfmc": {"shape": shape_nlp1, "type": type_ngptlw},
            "semiss": {"shape": shape_2D, "type": type_nbands},
            "semiss0": {"shape": shape_2D, "type": type_nbands},
            "secdiff": {"shape": shape_2D, "type": type_nbands},
            "colamt": {"shape": shape_nlp1, "type": type_maxgas},
            "wx": {"shape": shape_nlp1, "type": type_maxxsec},
            "rfrate": {"shape": shape_nlp1, "type": type_nrates},
            "tem0": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tem1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tem2": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "pwvcm": {"shape": shape_2D, "type": DTYPE_FLT},
            "summol": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "stemp": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "delgth": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ipseed": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jp": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jt": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jt1": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indself": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indfor": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indminor": {"shape": shape_nlp1, "type": DTYPE_INT},
            "tem00": {"shape": shape_2D, "type": DTYPE_FLT},
            "tem11": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tem22": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tauliq": {"shape": shape_nlp1, "type": type_nbands},
            "tauice": {"shape": shape_nlp1, "type": type_nbands},
            "cldf": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "dgeice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "factor": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fint": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tauran": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tausnw": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cldliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "refliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cldice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "refice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "index": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ia": {"shape": shape_nlp1, "type": DTYPE_INT},
            "lcloudy": {"shape": shape_nlp1, "type": (DTYPE_INT, (ngptlw,))},
            "lcf1": {"shape": shape_2D, "type": DTYPE_BOOL},
            "cldsum": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tlvlfr": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tlyrfr": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "plog": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "indlay": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indlev": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jp1": {"shape": shape_nlp1, "type": DTYPE_INT},
            "tzint": {"shape": shape_nlp1, "type": DTYPE_INT},
            "stempint": {"shape": shape_nlp1, "type": DTYPE_INT},
            "tavelint": {"shape": shape_nlp1, "type": DTYPE_INT},
            "laytrop": {"shape": shape_nlp1, "type": DTYPE_BOOL},
            "ib": {"shape": shape_2D, "type": DTYPE_INT},
            "ind0": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind0p": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind1": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind1p": {"shape": shape_nlp1, "type": DTYPE_INT},
            "inds": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indsp": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indf": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indfp": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indm": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indmp": {"shape": shape_nlp1, "type": DTYPE_INT},
            "js": {"shape": shape_nlp1, "type": DTYPE_INT},
            "js1": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jmn2o": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jmn2op": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jpl": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jplp": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id000": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id010": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id100": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id110": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id200": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id210": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id001": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id011": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id101": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id111": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id201": {"shape": shape_nlp1, "type": DTYPE_INT},
            "id211": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jmo3": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jmo3p": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jmco2": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jmco2p": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jmco": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jmcop": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jmn2": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jmn2p": {"shape": shape_nlp1, "type": DTYPE_INT},
            "taug": {"shape": shape_nlp1, "type": type_ngptlw},
            "pp": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "corradj": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "scalen2": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tauself": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "taufor": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "taun2": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fpl": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "speccomb": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "speccomb1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac001": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac101": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac201": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac011": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac111": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac211": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac000": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac100": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac200": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac010": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac110": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac210": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "specparm": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "specparm1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "specparm_planck": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ratn2o": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ratco2": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "clrurad": {"shape": shape_nlp1, "type": type_nbands},
            "clrdrad": {"shape": shape_nlp1, "type": type_nbands},
            "toturad": {"shape": shape_nlp1, "type": type_nbands},
            "totdrad": {"shape": shape_nlp1, "type": type_nbands},
            "gassrcu": {"shape": shape_nlp1, "type": type_ngptlw},
            "totsrcu": {"shape": shape_nlp1, "type": type_ngptlw},
            "trngas": {"shape": shape_nlp1, "type": type_ngptlw},
            "efclrfr": {"shape": shape_nlp1, "type": type_ngptlw},
            "rfdelp": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fnet": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fnetc": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "totsrcd": {"shape": shape_nlp1, "type": type_ngptlw},
            "gassrcd": {"shape": shape_nlp1, "type": type_ngptlw},
            "tblind": {"shape": shape_nlp1, "type": type_ngptlw},
            "odepth": {"shape": shape_nlp1, "type": type_ngptlw},
            "odtot": {"shape": shape_nlp1, "type": type_ngptlw},
            "odcld": {"shape": shape_nlp1, "type": type_ngptlw},
            "atrtot": {"shape": shape_nlp1, "type": type_ngptlw},
            "atrgas": {"shape": shape_nlp1, "type": type_ngptlw},
            "reflct": {"shape": shape_nlp1, "type": type_ngptlw},
            "totfac": {"shape": shape_nlp1, "type": type_ngptlw},
            "gasfac": {"shape": shape_nlp1, "type": type_ngptlw},
            "flxfac": {"shape": shape_nlp1, "type": type_ngptlw},
            "plfrac": {"shape": shape_nlp1, "type": type_ngptlw},
            "blay": {"shape": shape_nlp1, "type": type_ngptlw},
            "bbdgas": {"shape": shape_nlp1, "type": type_ngptlw},
            "bbdtot": {"shape": shape_nlp1, "type": type_ngptlw},
            "bbugas": {"shape": shape_nlp1, "type": type_ngptlw},
            "bbutot": {"shape": shape_nlp1, "type": type_ngptlw},
            "dplnku": {"shape": shape_nlp1, "type": type_ngptlw},
            "dplnkd": {"shape": shape_nlp1, "type": type_ngptlw},
            "radtotu": {"shape": shape_nlp1, "type": type_ngptlw},
            "radclru": {"shape": shape_nlp1, "type": type_ngptlw},
            "radtotd": {"shape": shape_nlp1, "type": type_ngptlw},
            "radclrd": {"shape": shape_nlp1, "type": type_ngptlw},
            "rad0": {"shape": shape_nlp1, "type": type_ngptlw},
            "clfm": {"shape": shape_nlp1, "type": type_ngptlw},
            "trng": {"shape": shape_nlp1, "type": type_ngptlw},
            "gasu": {"shape": shape_nlp1, "type": type_ngptlw},
            "itgas": {"shape": shape_nlp1, "type": (DTYPE_INT, (ngptlw,))},
            "ittot": {"shape": shape_nlp1, "type": (DTYPE_INT, (ngptlw,))},
        }

        locdict_gt4py = create_gt4py_dict_zeros(locvars)

        self.indict_gt4py = indict_gt4py
        self.locdict_gt4py = locdict_gt4py
        self.outdict_gt4py = outdict_gt4py
        self.outvars = outvars

    def _load_lookup_table_data(self):
        """
        Read in lookup table data from netcdf data that has been serialized out from
        radlw_datatb.F
        """

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_cldprlw_data.nc"))

        cldprop_types = {
            "absliq1": {"ctype": (DTYPE_FLT, (58, nbands)), "data": ds["absliq1"].data},
            "absice0": {"ctype": (DTYPE_FLT, (2,)), "data": ds["absice0"].data},
            "absice1": {"ctype": (DTYPE_FLT, (2, 5)), "data": ds["absice1"].data},
            "absice2": {"ctype": (DTYPE_FLT, (43, nbands)), "data": ds["absice2"].data},
            "absice3": {"ctype": (DTYPE_FLT, (46, nbands)), "data": ds["absice3"].data},
            "ipat": {"ctype": (DTYPE_INT, (nbands,)), "data": ipat},
        }

        lookupdict_gt4py = dict()

        for name, info in cldprop_types.items():
            lookupdict_gt4py[name] = create_storage_from_array(
                info["data"], backend, shape_nlp1, info["ctype"]
            )

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "totplnk.nc"))
        totplnk = ds["totplnk"].data

        totplnk = np.tile(totplnk[None, None, None, :, :], (npts, 1, nlp1, 1, 1))
        lookupdict_gt4py["totplnk"] = create_storage_from_array(
            totplnk, backend, shape_nlp1, (DTYPE_FLT, (nplnk, nbands))
        )

        refvars = ["pref", "preflog", "tref", "chi_mls"]
        ds2 = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))

        for var in refvars:
            tmp = ds2[var].data

            if var == "chi_mls":
                tmp = np.tile(tmp[None, None, None, :, :], (npts, 1, nlp1, 1, 1))
                lookupdict_gt4py[var] = create_storage_from_array(
                    tmp, backend, shape_nlp1, (DTYPE_FLT, (7, 59))
                )
            else:
                tmp = np.tile(tmp[None, None, None, :], (npts, 1, nlp1, 1))
                lookupdict_gt4py[var] = create_storage_from_array(
                    tmp, backend, shape_nlp1, (DTYPE_FLT, (59,))
                )

        delwave = np.tile(self.delwave[None, None, None, :], (npts, 1, nlp1, 1))
        delwave = create_storage_from_array(delwave, backend, shape_nlp1, type_nbands)
        lookupdict_gt4py["delwave"] = delwave

        print("Loading lookup table data . . .")
        self.lookupdict_gt4py1 = loadlookupdata("kgb01", "radlw")
        self.lookupdict_gt4py2 = loadlookupdata("kgb02", "radlw")
        self.lookupdict_gt4py3 = loadlookupdata("kgb03", "radlw")
        self.lookupdict_gt4py4 = loadlookupdata("kgb04", "radlw")
        self.lookupdict_gt4py5 = loadlookupdata("kgb05", "radlw")
        self.lookupdict_gt4py6 = loadlookupdata("kgb06", "radlw")
        self.lookupdict_gt4py7 = loadlookupdata("kgb07", "radlw")
        self.lookupdict_gt4py8 = loadlookupdata("kgb08", "radlw")
        self.lookupdict_gt4py9 = loadlookupdata("kgb09", "radlw")
        self.lookupdict_gt4py10 = loadlookupdata("kgb10", "radlw")
        self.lookupdict_gt4py11 = loadlookupdata("kgb11", "radlw")
        self.lookupdict_gt4py12 = loadlookupdata("kgb12", "radlw")
        self.lookupdict_gt4py13 = loadlookupdata("kgb13", "radlw")
        self.lookupdict_gt4py14 = loadlookupdata("kgb14", "radlw")
        self.lookupdict_gt4py15 = loadlookupdata("kgb15", "radlw")
        self.lookupdict_gt4py16 = loadlookupdata("kgb16", "radlw")
        print("Done")
        print(" ")

        self.lookupdict_gt4py = lookupdict_gt4py

    def _load_random_numbers(self, rank):
        """
        Read in 2-D array of random numbers used in mcica_subcol, this will change
        in the future once there is a solution for the RNG in python/gt4py

        This serialized set of random numbers will be used for testing, and the python
        RNG for running the model.

        rand2d is shape (npts, ngptlw*nlay), and I will reshape it to (npts, 1, nlp1, ngptlw)
        - First reshape to (npts, ngptlw, nlay)
        - Second pad k axis with one zero
        - Third switch order of k and data axes
        """
        ds = xr.open_dataset(
            os.path.join(LOOKUP_DIR, "rand2d_tile" + str(rank) + "_lw.nc")
        )
        rand2d = ds["rand2d"][:, :].data
        cdfunc = np.zeros((npts, ngptlw, nlay))
        for n in range(npts):
            cdfunc[n, :, :] = np.reshape(rand2d[n, :], (ngptlw, nlay), order="C")
        cdfunc = np.insert(cdfunc, 0, 0, axis=2)
        cdfunc = np.transpose(cdfunc, (0, 2, 1))

        cdfunc = np.tile(cdfunc[:, None, :, :], (1, 1, 1, 1))
        self.lookupdict_gt4py["cdfunc"] = create_storage_from_array(
            cdfunc, backend, shape_nlp1, type_ngptlw
        )

    def lwrad(self, rank, do_subtest=False):
        """Run the main longwave radiation scheme

        Requires create_input_data to have been run before calling
        Currently uses serialized random number arrays in cldprop

        Args:
            rank (int): current rank
            do_subtest (bool, optional): flag to test individual stencil outputs. Defaults to False.
        """

        start0 = time.time()
        firstloop(
            self.indict_gt4py["plyr"],
            self.indict_gt4py["plvl"],
            self.indict_gt4py["tlyr"],
            self.indict_gt4py["tlvl"],
            self.indict_gt4py["qlyr"],
            self.indict_gt4py["olyr"],
            self.indict_gt4py["gasvmr"],
            self.indict_gt4py["clouds"],
            self.indict_gt4py["icsdlw"],
            self.indict_gt4py["faerlw"],
            self.indict_gt4py["semis"],
            self.indict_gt4py["tsfg"],
            self.indict_gt4py["dz"],
            self.indict_gt4py["delp"],
            self.indict_gt4py["de_lgth"],
            self.locdict_gt4py["cldfrc"],
            self.locdict_gt4py["pavel"],
            self.locdict_gt4py["tavel"],
            self.locdict_gt4py["delp"],
            self.locdict_gt4py["dz"],
            self.locdict_gt4py["h2ovmr"],
            self.locdict_gt4py["o3vmr"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colbrd"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["wx"],
            self.locdict_gt4py["tauaer"],
            self.locdict_gt4py["semiss0"],
            self.locdict_gt4py["semiss"],
            self.locdict_gt4py["tem11"],
            self.locdict_gt4py["tem22"],
            self.locdict_gt4py["tem00"],
            self.locdict_gt4py["summol"],
            self.locdict_gt4py["pwvcm"],
            self.locdict_gt4py["clwp"],
            self.locdict_gt4py["relw"],
            self.locdict_gt4py["ciwp"],
            self.locdict_gt4py["reiw"],
            self.locdict_gt4py["cda1"],
            self.locdict_gt4py["cda2"],
            self.locdict_gt4py["cda3"],
            self.locdict_gt4py["cda4"],
            self.locdict_gt4py["secdiff"],
            self.A0,
            self.A1,
            self.A2,
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:

            outvars_firstloop = {
                "pavel": {"fortran_shape": (npts, nlay)},
                "tavel": {"fortran_shape": (npts, nlay)},
                "delp": {"fortran_shape": (npts, nlay)},
                "colbrd": {"fortran_shape": (npts, nlay)},
                "cldfrc": {"fortran_shape": (npts, nlp1 + 1)},
                "taucld": {"fortran_shape": (npts, nbands, nlay)},
                "dz": {"fortran_shape": (npts, nlay)},
                "semiss": {"fortran_shape": (npts, nbands)},
                "coldry": {"fortran_shape": (npts, nlay)},
                "colamt": {"fortran_shape": (npts, nlay, maxgas)},
                "tauaer": {"fortran_shape": (npts, nbands, nlay)},
                "h2ovmr": {"fortran_shape": (npts, nlay)},
                "o3vmr": {"fortran_shape": (npts, nlay)},
                "wx": {"fortran_shape": (npts, nlay, maxxsec)},
                "clwp": {"fortran_shape": (npts, nlay)},
                "relw": {"fortran_shape": (npts, nlay)},
                "ciwp": {"fortran_shape": (npts, nlay)},
                "reiw": {"fortran_shape": (npts, nlay)},
                "cda1": {"fortran_shape": (npts, nlay)},
                "cda2": {"fortran_shape": (npts, nlay)},
                "cda3": {"fortran_shape": (npts, nlay)},
                "cda4": {"fortran_shape": (npts, nlay)},
                "pwvcm": {"fortran_shape": (npts,)},
                "secdiff": {"fortran_shape": (npts, nbands)},
            }

            outdict_firstloop = convert_gt4py_output_for_validation(
                self.locdict_gt4py, outvars_firstloop
            )
            valdict_firstloop = read_intermediate_data(
                LW_SERIALIZED_DIR, "lwrad", rank, 0, "firstloop", outvars_firstloop
            )

            print("Testing firstloop...")
            print(" ")
            compare_data(outdict_firstloop, valdict_firstloop)
            print(" ")
            print("Firstloop validates!")
            print(" ")

        self._load_random_numbers(rank)

        cldprop(
            self.locdict_gt4py["cldfrc"],
            self.locdict_gt4py["clwp"],
            self.locdict_gt4py["relw"],
            self.locdict_gt4py["ciwp"],
            self.locdict_gt4py["reiw"],
            self.locdict_gt4py["cda1"],
            self.locdict_gt4py["cda2"],
            self.locdict_gt4py["cda3"],
            self.locdict_gt4py["cda4"],
            self.locdict_gt4py["dz"],
            self.locdict_gt4py["cldfmc"],
            self.locdict_gt4py["taucld"],
            self.outdict_gt4py["cldtaulw"],
            self.lookupdict_gt4py["absliq1"],
            self.lookupdict_gt4py["absice1"],
            self.lookupdict_gt4py["absice2"],
            self.lookupdict_gt4py["absice3"],
            self.lookupdict_gt4py["ipat"],
            self.locdict_gt4py["tauliq"],
            self.locdict_gt4py["tauice"],
            self.locdict_gt4py["cldf"],
            self.locdict_gt4py["dgeice"],
            self.locdict_gt4py["factor"],
            self.locdict_gt4py["fint"],
            self.locdict_gt4py["tauran"],
            self.locdict_gt4py["tausnw"],
            self.locdict_gt4py["cldliq"],
            self.locdict_gt4py["refliq"],
            self.locdict_gt4py["cldice"],
            self.locdict_gt4py["refice"],
            self.locdict_gt4py["index"],
            self.locdict_gt4py["ia"],
            self.locdict_gt4py["lcloudy"],
            self.lookupdict_gt4py["cdfunc"],
            self.locdict_gt4py["tem1"],
            self.locdict_gt4py["lcf1"],
            self.locdict_gt4py["cldsum"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:

            outvars_cldprop = {
                "cldfmc": {"fortran_shape": (npts, ngptlw, nlay)},
                "taucld": {"fortran_shape": (npts, nbands, nlay)},
            }

            outdict_cldprop = convert_gt4py_output_for_validation(
                self.locdict_gt4py, outvars_cldprop
            )
            valdict_cldprop = read_intermediate_data(
                LW_SERIALIZED_DIR, "lwrad", rank, 0, "cldprop", outvars_cldprop
            )

            print("Testing cldprop...")
            print(" ")
            compare_data(outdict_cldprop, valdict_cldprop)
            print(" ")
            print("cldprop validates!")
            print(" ")

        setcoef(
            self.locdict_gt4py["pavel"],
            self.locdict_gt4py["tavel"],
            self.indict_gt4py["tlvl"],
            self.indict_gt4py["tsfg"],
            self.locdict_gt4py["h2ovmr"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colbrd"],
            self.lookupdict_gt4py["totplnk"],
            self.lookupdict_gt4py["pref"],
            self.lookupdict_gt4py["preflog"],
            self.lookupdict_gt4py["tref"],
            self.lookupdict_gt4py["chi_mls"],
            self.lookupdict_gt4py["delwave"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["pklay"],
            self.locdict_gt4py["pklev"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["scaleminor"],
            self.locdict_gt4py["scaleminorn2"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["tzint"],
            self.locdict_gt4py["stempint"],
            self.locdict_gt4py["tavelint"],
            self.locdict_gt4py["indlay"],
            self.locdict_gt4py["indlev"],
            self.locdict_gt4py["tlyrfr"],
            self.locdict_gt4py["tlvlfr"],
            self.locdict_gt4py["jp1"],
            self.locdict_gt4py["plog"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:
            outvars_setcoef = {
                "laytrop": {"fortran_shape": (npts,)},
                "pklay": {"fortran_shape": (npts, nbands, nlp1)},
                "pklev": {"fortran_shape": (npts, nbands, nlp1)},
                "jp": {"fortran_shape": (npts, nlay)},
                "jt": {"fortran_shape": (npts, nlay)},
                "jt1": {"fortran_shape": (npts, nlay)},
                "rfrate": {"fortran_shape": (npts, nlay, nrates, 2)},
                "fac00": {"fortran_shape": (npts, nlay)},
                "fac01": {"fortran_shape": (npts, nlay)},
                "fac10": {"fortran_shape": (npts, nlay)},
                "fac11": {"fortran_shape": (npts, nlay)},
                "selffac": {"fortran_shape": (npts, nlay)},
                "selffrac": {"fortran_shape": (npts, nlay)},
                "indself": {"fortran_shape": (npts, nlay)},
                "forfac": {"fortran_shape": (npts, nlay)},
                "forfrac": {"fortran_shape": (npts, nlay)},
                "indfor": {"fortran_shape": (npts, nlay)},
                "minorfrac": {"fortran_shape": (npts, nlay)},
                "scaleminor": {"fortran_shape": (npts, nlay)},
                "scaleminorn2": {"fortran_shape": (npts, nlay)},
                "indminor": {"fortran_shape": (npts, nlay)},
            }

            outdict_setcoef = convert_gt4py_output_for_validation(
                self.locdict_gt4py, outvars_setcoef
            )
            valdict_setcoef = read_intermediate_data(
                LW_SERIALIZED_DIR, "lwrad", rank, 0, "setcoef", outvars_setcoef
            )

            print("Testing setcoef...")
            print(" ")
            compare_data(outdict_setcoef, valdict_setcoef)
            print(" ")
            print("setcoef validates!")
            print(" ")

        taugb01(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["pavel"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colbrd"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["scaleminorn2"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py1["absa"],
            self.lookupdict_gt4py1["absb"],
            self.lookupdict_gt4py1["selfref"],
            self.lookupdict_gt4py1["forref"],
            self.lookupdict_gt4py1["fracrefa"],
            self.lookupdict_gt4py1["fracrefb"],
            self.lookupdict_gt4py1["ka_mn2"],
            self.lookupdict_gt4py1["kb_mn2"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["pp"],
            self.locdict_gt4py["corradj"],
            self.locdict_gt4py["scalen2"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["taun2"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb02(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["pavel"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py2["absa"],
            self.lookupdict_gt4py2["absb"],
            self.lookupdict_gt4py2["selfref"],
            self.lookupdict_gt4py2["forref"],
            self.lookupdict_gt4py2["fracrefa"],
            self.lookupdict_gt4py2["fracrefb"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["corradj"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb03(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py3["absa"],
            self.lookupdict_gt4py3["absb"],
            self.lookupdict_gt4py3["selfref"],
            self.lookupdict_gt4py3["forref"],
            self.lookupdict_gt4py3["fracrefa"],
            self.lookupdict_gt4py3["fracrefb"],
            self.lookupdict_gt4py3["ka_mn2o"],
            self.lookupdict_gt4py3["kb_mn2o"],
            self.lookupdict_gt4py3["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jmn2o"],
            self.locdict_gt4py["jmn2op"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            self.locdict_gt4py["ratn2o"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb04(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py4["absa"],
            self.lookupdict_gt4py4["absb"],
            self.lookupdict_gt4py4["selfref"],
            self.lookupdict_gt4py4["forref"],
            self.lookupdict_gt4py4["fracrefa"],
            self.lookupdict_gt4py4["fracrefb"],
            self.lookupdict_gt4py4["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb05(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["wx"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py5["absa"],
            self.lookupdict_gt4py5["absb"],
            self.lookupdict_gt4py5["selfref"],
            self.lookupdict_gt4py5["forref"],
            self.lookupdict_gt4py5["fracrefa"],
            self.lookupdict_gt4py5["fracrefb"],
            self.lookupdict_gt4py5["ka_mo3"],
            self.lookupdict_gt4py5["ccl4"],
            self.lookupdict_gt4py5["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["jmo3"],
            self.locdict_gt4py["jmo3p"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb06(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["wx"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py6["absa"],
            self.lookupdict_gt4py6["selfref"],
            self.lookupdict_gt4py6["forref"],
            self.lookupdict_gt4py6["fracrefa"],
            self.lookupdict_gt4py6["ka_mco2"],
            self.lookupdict_gt4py6["cfc11adj"],
            self.lookupdict_gt4py6["cfc12"],
            self.lookupdict_gt4py6["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["ratco2"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb07(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py7["absa"],
            self.lookupdict_gt4py7["absb"],
            self.lookupdict_gt4py7["selfref"],
            self.lookupdict_gt4py7["forref"],
            self.lookupdict_gt4py7["fracrefa"],
            self.lookupdict_gt4py7["fracrefb"],
            self.lookupdict_gt4py7["ka_mco2"],
            self.lookupdict_gt4py7["kb_mco2"],
            self.lookupdict_gt4py7["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jmco2"],
            self.locdict_gt4py["jmco2p"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            self.locdict_gt4py["ratco2"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb08(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["wx"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py8["absa"],
            self.lookupdict_gt4py8["absb"],
            self.lookupdict_gt4py8["selfref"],
            self.lookupdict_gt4py8["forref"],
            self.lookupdict_gt4py8["fracrefa"],
            self.lookupdict_gt4py8["fracrefb"],
            self.lookupdict_gt4py8["ka_mo3"],
            self.lookupdict_gt4py8["ka_mco2"],
            self.lookupdict_gt4py8["kb_mco2"],
            self.lookupdict_gt4py8["cfc12"],
            self.lookupdict_gt4py8["ka_mn2o"],
            self.lookupdict_gt4py8["kb_mn2o"],
            self.lookupdict_gt4py8["cfc22adj"],
            self.lookupdict_gt4py8["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["ratco2"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb09(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py9["absa"],
            self.lookupdict_gt4py9["absb"],
            self.lookupdict_gt4py9["selfref"],
            self.lookupdict_gt4py9["forref"],
            self.lookupdict_gt4py9["fracrefa"],
            self.lookupdict_gt4py9["fracrefb"],
            self.lookupdict_gt4py9["ka_mn2o"],
            self.lookupdict_gt4py9["kb_mn2o"],
            self.lookupdict_gt4py9["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jmco2"],
            self.locdict_gt4py["jmco2p"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            self.locdict_gt4py["ratn2o"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb10(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py10["absa"],
            self.lookupdict_gt4py10["absb"],
            self.lookupdict_gt4py10["selfref"],
            self.lookupdict_gt4py10["forref"],
            self.lookupdict_gt4py10["fracrefa"],
            self.lookupdict_gt4py10["fracrefb"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb11(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["scaleminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py11["absa"],
            self.lookupdict_gt4py11["absb"],
            self.lookupdict_gt4py11["selfref"],
            self.lookupdict_gt4py11["forref"],
            self.lookupdict_gt4py11["fracrefa"],
            self.lookupdict_gt4py11["fracrefb"],
            self.lookupdict_gt4py11["ka_mo2"],
            self.lookupdict_gt4py11["kb_mo2"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb12(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py12["absa"],
            self.lookupdict_gt4py12["selfref"],
            self.lookupdict_gt4py12["forref"],
            self.lookupdict_gt4py12["fracrefa"],
            self.lookupdict_gt4py12["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            self.locdict_gt4py["specparm_planck"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb13(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py13["absa"],
            self.lookupdict_gt4py13["selfref"],
            self.lookupdict_gt4py13["forref"],
            self.lookupdict_gt4py13["fracrefa"],
            self.lookupdict_gt4py13["fracrefb"],
            self.lookupdict_gt4py13["ka_mco"],
            self.lookupdict_gt4py13["ka_mco2"],
            self.lookupdict_gt4py13["kb_mo3"],
            self.lookupdict_gt4py13["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["jmco"],
            self.locdict_gt4py["jmcop"],
            self.locdict_gt4py["jmco2"],
            self.locdict_gt4py["jmco2p"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            self.locdict_gt4py["ratco2"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb14(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py14["absa"],
            self.lookupdict_gt4py14["absb"],
            self.lookupdict_gt4py14["selfref"],
            self.lookupdict_gt4py14["forref"],
            self.lookupdict_gt4py14["fracrefa"],
            self.lookupdict_gt4py14["fracrefb"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb15(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colbrd"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["indminor"],
            self.locdict_gt4py["minorfrac"],
            self.locdict_gt4py["scaleminor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py15["absa"],
            self.lookupdict_gt4py15["selfref"],
            self.lookupdict_gt4py15["forref"],
            self.lookupdict_gt4py15["fracrefa"],
            self.lookupdict_gt4py15["ka_mn2"],
            self.lookupdict_gt4py15["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["indm"],
            self.locdict_gt4py["indmp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["taun2"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["jmn2"],
            self.locdict_gt4py["jmn2p"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["fpl"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taugb16(
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["rfrate"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["taug"],
            self.lookupdict_gt4py16["absa"],
            self.lookupdict_gt4py16["absb"],
            self.lookupdict_gt4py16["selfref"],
            self.lookupdict_gt4py16["forref"],
            self.lookupdict_gt4py16["fracrefa"],
            self.lookupdict_gt4py16["fracrefb"],
            self.lookupdict_gt4py16["chi_mls"],
            self.locdict_gt4py["ind0"],
            self.locdict_gt4py["ind0p"],
            self.locdict_gt4py["ind1"],
            self.locdict_gt4py["ind1p"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["tauself"],
            self.locdict_gt4py["taufor"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["js1"],
            self.locdict_gt4py["jpl"],
            self.locdict_gt4py["jplp"],
            self.locdict_gt4py["id000"],
            self.locdict_gt4py["id010"],
            self.locdict_gt4py["id100"],
            self.locdict_gt4py["id110"],
            self.locdict_gt4py["id200"],
            self.locdict_gt4py["id210"],
            self.locdict_gt4py["id001"],
            self.locdict_gt4py["id011"],
            self.locdict_gt4py["id101"],
            self.locdict_gt4py["id111"],
            self.locdict_gt4py["id201"],
            self.locdict_gt4py["id211"],
            self.locdict_gt4py["fpl"],
            self.locdict_gt4py["speccomb"],
            self.locdict_gt4py["speccomb1"],
            self.locdict_gt4py["fac000"],
            self.locdict_gt4py["fac100"],
            self.locdict_gt4py["fac200"],
            self.locdict_gt4py["fac010"],
            self.locdict_gt4py["fac110"],
            self.locdict_gt4py["fac210"],
            self.locdict_gt4py["fac001"],
            self.locdict_gt4py["fac101"],
            self.locdict_gt4py["fac201"],
            self.locdict_gt4py["fac011"],
            self.locdict_gt4py["fac111"],
            self.locdict_gt4py["fac211"],
            self.locdict_gt4py["specparm"],
            self.locdict_gt4py["specparm1"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        combine_optical_depth(
            self.NGB,
            self.locdict_gt4py["ib"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["tauaer"],
            self.locdict_gt4py["tautot"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:

            outvars_t = {
                "fracs": {"fortran_shape": (npts, ngptlw, nlay)},
                "tautot": {"fortran_shape": (npts, ngptlw, nlay)},
            }

            valdict_taumol = read_intermediate_data(
                LW_SERIALIZED_DIR, "lwrad", rank, 0, "taumol", outvars_t
            )

            outdict_taumol = convert_gt4py_output_for_validation(
                self.locdict_gt4py, outvars_t
            )

            print("Testing taumol...")
            print(" ")
            compare_data(outdict_taumol, valdict_taumol)
            print(" ")
            print("taumol validates!")
            print(" ")

        rtrnmc(
            self.locdict_gt4py["semiss"],
            self.locdict_gt4py["secdiff"],
            self.locdict_gt4py["delp"],
            self.locdict_gt4py["taucld"],
            self.locdict_gt4py["fracs"],
            self.locdict_gt4py["tautot"],
            self.locdict_gt4py["cldfmc"],
            self.locdict_gt4py["pklay"],
            self.locdict_gt4py["pklev"],
            self.exp_tbl,
            self.tau_tbl,
            self.tfn_tbl,
            self.NGB,
            self.locdict_gt4py["totuflux"],
            self.locdict_gt4py["totdflux"],
            self.locdict_gt4py["totuclfl"],
            self.locdict_gt4py["totdclfl"],
            self.outdict_gt4py["upfxc_t"],
            self.outdict_gt4py["upfx0_t"],
            self.outdict_gt4py["upfxc_s"],
            self.outdict_gt4py["upfx0_s"],
            self.outdict_gt4py["dnfxc_s"],
            self.outdict_gt4py["dnfx0_s"],
            self.outdict_gt4py["htlwc"],
            self.outdict_gt4py["htlw0"],
            self.locdict_gt4py["clrurad"],
            self.locdict_gt4py["clrdrad"],
            self.locdict_gt4py["toturad"],
            self.locdict_gt4py["totdrad"],
            self.locdict_gt4py["gassrcu"],
            self.locdict_gt4py["totsrcu"],
            self.locdict_gt4py["trngas"],
            self.locdict_gt4py["efclrfr"],
            self.locdict_gt4py["rfdelp"],
            self.locdict_gt4py["fnet"],
            self.locdict_gt4py["fnetc"],
            self.locdict_gt4py["totsrcd"],
            self.locdict_gt4py["gassrcd"],
            self.locdict_gt4py["tblind"],
            self.locdict_gt4py["odepth"],
            self.locdict_gt4py["odtot"],
            self.locdict_gt4py["odcld"],
            self.locdict_gt4py["atrtot"],
            self.locdict_gt4py["atrgas"],
            self.locdict_gt4py["reflct"],
            self.locdict_gt4py["totfac"],
            self.locdict_gt4py["gasfac"],
            self.locdict_gt4py["plfrac"],
            self.locdict_gt4py["blay"],
            self.locdict_gt4py["bbdgas"],
            self.locdict_gt4py["bbdtot"],
            self.locdict_gt4py["bbugas"],
            self.locdict_gt4py["bbutot"],
            self.locdict_gt4py["dplnku"],
            self.locdict_gt4py["dplnkd"],
            self.locdict_gt4py["radtotu"],
            self.locdict_gt4py["radclru"],
            self.locdict_gt4py["radtotd"],
            self.locdict_gt4py["radclrd"],
            self.locdict_gt4py["rad0"],
            self.locdict_gt4py["clfm"],
            self.locdict_gt4py["trng"],
            self.locdict_gt4py["gasu"],
            self.locdict_gt4py["itgas"],
            self.locdict_gt4py["ittot"],
            self.locdict_gt4py["ib"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        end0 = time.time()
        print(f"Total time taken = {end0 - start0}")

        if do_subtest:
            outvars_rtrnmc = {
                "totuflux": {"fortran_shape": (npts, nlp1), "type": DTYPE_FLT},
                "totdflux": {"fortran_shape": (npts, nlp1), "type": DTYPE_FLT},
                "totuclfl": {"fortran_shape": (npts, nlp1), "type": DTYPE_FLT},
                "totdclfl": {"fortran_shape": (npts, nlp1), "type": DTYPE_FLT},
            }

            outdict_rtrnmc = convert_gt4py_output_for_validation(
                self.locdict_gt4py, outvars_rtrnmc
            )

            valdict_rtrnmc = read_intermediate_data(
                LW_SERIALIZED_DIR, "lwrad", rank, 0, "rtrnmc", outvars_rtrnmc
            )

            print("Testing rtrnmc...")
            print(" ")
            compare_data(outdict_rtrnmc, valdict_rtrnmc)
            print(" ")
            print("rtrnmc validates!")
            print(" ")

        valdict = dict()
        outdict_np = dict()

        valdict = read_data(
            os.path.join(FORTRANDATA_DIR, "LW"), "lwrad", rank, 0, False, self.outvars
        )
        outdict_np = convert_gt4py_output_for_validation(
            self.outdict_gt4py, self.outvars
        )

        print("Testing final output...")
        print(" ")
        compare_data(valdict, outdict_np)
        print(" ")
        print("lwrad validates!")
        print(" ")
