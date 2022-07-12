from attr.setters import convert
import numpy as np
import xarray as xr
import os
import sys
import time
import warnings

sys.path.insert(0, "..")
from radsw_param import ntbmx, NGB, nbandssw, ngs
from radphysparam import iswmode, iswrgas, iswrate, iswcice, iswcliq
from phys_const import con_amd, con_amw, con_amo3, con_g, con_cp, con_avgd
from util import *
from config import *
from stencils_sw_gt4py import *


class RadSWClass:
    VTAGSW = "NCEP SW v5.1  Nov 2012 -RRTMG-SW v3.8"

    # constant values
    eps = 1.0e-6
    oneminus = 1.0 - eps
    # pade approx constant
    bpade = 1.0 / 0.278
    stpfac = 296.0 / 1013.0
    ftiny = 1.0e-12
    flimit = 1.0e-20
    # internal solar constant
    s0 = 1368.22
    f_zero = 0.0
    f_one = 1.0

    # atomic weights for conversion from mass to volume mixing ratios
    amdw = con_amd / con_amw
    amdo3 = con_amd / con_amo3

    # band indices
    nspa = [9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1]
    nspb = [1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1]
    # band index for sfc flux
    idxsfc = [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 1]
    # band index for cld prop
    idxebc = [5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 5]

    # uv-b band index
    nuvb = 27

    # initial permutation seed used for sub-column cloud scheme
    ipsdsw0 = 1

    def __init__(self, me, iovrsw, isubcsw, icldflg):

        self.iovrsw = iovrsw
        self.isubcsw = isubcsw
        self.icldflg = icldflg

        expeps = 1.0e-20

        #
        # ===> ... begin here
        #
        if self.iovrsw < 0 or self.iovrsw > 3:
            raise ValueError(
                "*** Error in specification of cloud overlap flag",
                f" IOVRSW={self.iovrsw} in RSWINIT !!",
            )

        if me == 0:
            print(f"- Using AER Shortwave Radiation, Version: {self.VTAGSW}")

            if iswmode == 1:
                print("   --- Delta-eddington 2-stream transfer scheme")
            elif iswmode == 2:
                print("   --- PIFM 2-stream transfer scheme")
            elif iswmode == 3:
                print("   --- Discrete ordinates 2-stream transfer scheme")

            if iswrgas <= 0:
                print("   --- Rare gases absorption is NOT included in SW")
            else:
                print("   --- Include rare gases N2O, CH4, O2, absorptions in SW")

            if self.isubcsw == 0:
                print(
                    "   --- Using standard grid average clouds, no ",
                    "   sub-column clouds approximation applied",
                )
            elif self.isubcsw == 1:
                print(
                    "   --- Using MCICA sub-colum clouds approximation ",
                    "   with a prescribed sequence of permutation seeds",
                )
            elif self.isubcsw == 2:
                print(
                    "   --- Using MCICA sub-colum clouds approximation ",
                    "   with provided input array of permutation seeds",
                )
            else:
                raise ValueError(
                    "  *** Error in specification of sub-column cloud ",
                    f" control flag isubcsw = {self.isubcsw} !!",
                )

        #  --- ...  check cloud flags for consistency

        if (icldflg == 0 and iswcliq != 0) or (icldflg == 1 and iswcliq == 0):
            raise ValueError(
                "*** Model cloud scheme inconsistent with SW",
                " radiation cloud radiative property setup !!",
            )

        if self.isubcsw == 0 and self.iovrsw > 2:
            if me == 0:
                warnings.warn(
                    f"*** IOVRSW={self.iovrsw} is not available for",
                    " ISUBCSW=0 setting!!",
                )
                warnings.warn(
                    "The program will use maximum/random overlap", " instead."
                )
            self.iovrsw = 1

        #  --- ...  setup constant factors for heating rate
        #           the 1.0e-2 is to convert pressure from mb to N/m**2

        if iswrate == 1:
            self.heatfac = con_g * 864.0 / con_cp  #   (in k/day)
        else:
            self.heatfac = con_g * 1.0e-2 / con_cp  #   (in k/second)

        #  --- ...  define exponential lookup tables for transmittance. tau is
        #           computed as a function of the tau transition function, and
        #           transmittance is calculated as a function of tau.  all tables
        #           are computed at intervals of 0.0001.  the inverse of the
        #           constant used in the Pade approximation to the tau transition
        #           function is set to bpade.

        self.exp_tbl = np.zeros(ntbmx + 1)
        self.exp_tbl[0] = 1.0
        self.exp_tbl[ntbmx] = expeps

        for i in range(ntbmx - 1):
            tfn = i / (ntbmx - i)
            tau = self.bpade * tfn
            self.exp_tbl[i] = np.exp(-tau)

        self.exp_tbl = np.tile(self.exp_tbl[None, None, None, :], (npts, 1, nlp1, 1))

        self.exp_tbl = create_storage_from_array(
            self.exp_tbl, backend, shape_nlp1, type_ntbmx
        )

        self.NGB = np.tile(np.array(NGB)[None, None, None, :], (npts, 1, nlp1, 1))
        self.ngs = np.tile(np.array(ngs)[None, None, :], (npts, 1, 1))

        self.NGB = create_storage_from_array(self.NGB, backend, shape_nlp1, type_ngptsw)
        self.ngs = create_storage_from_array(
            self.ngs, backend, shape_2D, type_nbandssw_int
        )

        self.idxsfc - np.tile(
            np.array(self.idxsfc)[None, None, None, :], (npts, 1, nlp1, 1)
        )
        self.idxebc - np.tile(
            np.array(self.idxebc)[None, None, None, :], (npts, 1, nlp1, 1)
        )
        self.nspa - np.tile(np.array(self.nspa)[None, None, :], (npts, 1, 1))
        self.nspb - np.tile(np.array(self.nspb)[None, None, :], (npts, 1, 1))

        self.idxsfc = create_storage_from_array(
            self.idxsfc, backend, shape_nlp1, (DTYPE_INT, (nbandssw,))
        )

        self.idxebc = create_storage_from_array(
            self.idxebc, backend, shape_nlp1, (DTYPE_INT, (nbandssw,))
        )
        self.nspa = create_storage_from_array(
            self.nspa, backend, shape_2D, (DTYPE_INT, (nbandssw,))
        )

        self.nspb = create_storage_from_array(
            self.nspb, backend, shape_2D, (DTYPE_INT, (nbandssw,))
        )

        layind = np.arange(nlay, dtype=np.int32)
        layind = np.insert(layind, 0, 0)
        layind = np.tile(layind[None, None, :], (npts, 1, 1))
        self.layind = create_storage_from_array(layind, backend, shape_nlp1, DTYPE_INT)

        self._load_lookup_table_data()

    def return_initdata(self):
        outdict = {"heatfac": self.heatfac, "exp_tbl": self.exp_tbl}
        return outdict

    def create_input_data(self, rank):

        self.serializer2 = ser.Serializer(
            ser.OpenModeKind.Read, SW_SERIALIZED_DIR, "Serialized_rank" + str(rank)
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
            "faersw": {"shape": (npts, nlay, nbdsw, 3), "type": type_nbandssw3},
            "sfcalb": {"shape": (npts, 4), "type": (DTYPE_FLT, (4,))},
            "dz": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "delp": {"shape": (npts, nlay), "type": DTYPE_FLT},
            "de_lgth": {"shape": (npts,), "type": DTYPE_FLT},
            "coszen": {"shape": (npts,), "type": DTYPE_FLT},
            "solcon": {"shape": (), "type": DTYPE_FLT},
            "nday": {"shape": (), "type": DTYPE_INT},
            "idxday": {"shape": (npts,), "type": DTYPE_BOOL},
            "im": {"shape": (), "type": DTYPE_INT},
            "lmk": {"shape": (), "type": DTYPE_INT},
            "lmp": {"shape": (), "type": DTYPE_INT},
            "lprnt": {"shape": (), "type": DTYPE_BOOL},
        }

        self._indict = read_data(
            os.path.join(FORTRANDATA_DIR, "SW"), "swrad", rank, 0, True, invars
        )
        if rank == 1:  # serialized idxday data has a spurious value on rank 1
            indict_gt4py = numpy_dict_to_gt4py_dict(
                self._indict, invars, rank1_flag=True
            )
        else:
            indict_gt4py = numpy_dict_to_gt4py_dict(self._indict, invars)

        outvars = {
            "upfxc_t": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "dnfxc_t": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "upfx0_t": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "upfxc_s": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "dnfxc_s": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "upfx0_s": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "dnfx0_s": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "htswc": {
                "shape": shape_nlp1,
                "type": DTYPE_FLT,
                "fortran_shape": (npts, nlay),
            },
            "htsw0": {
                "shape": shape_nlp1,
                "type": DTYPE_FLT,
                "fortran_shape": (npts, nlay),
            },
            "uvbf0": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "uvbfc": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "nirbm": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "nirdf": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "visbm": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "visdf": {"shape": shape_2D, "type": DTYPE_FLT, "fortran_shape": (npts,)},
            "cldtausw": {
                "shape": shape_nlp1,
                "type": DTYPE_FLT,
                "fortran_shape": (npts, nlay),
            },
        }

        outdict_gt4py = create_gt4py_dict_zeros(outvars)

        locvars = {
            "cosz1": {"shape": shape_2D, "type": DTYPE_FLT},
            "sntz1": {"shape": shape_2D, "type": DTYPE_FLT},
            "ssolar": {"shape": shape_2D, "type": DTYPE_FLT},
            "albbm": {"shape": shape_nlp1, "type": (DTYPE_FLT, (2,))},
            "albdf": {"shape": shape_nlp1, "type": (DTYPE_FLT, (2,))},
            "pavel": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tavel": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "h2ovmr": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "o3vmr": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "coldry": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "temcol": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "colamt": {"shape": shape_nlp1, "type": type_maxgas},
            "colmol": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tauae": {"shape": shape_nlp1, "type": type_nbdsw},
            "ssaae": {"shape": shape_nlp1, "type": type_nbdsw},
            "asyae": {"shape": shape_nlp1, "type": type_nbdsw},
            "cfrac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cliqp": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "reliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cicep": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "reice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cdat1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cdat2": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cdat3": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cdat4": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "zcf0": {"shape": shape_2D, "type": DTYPE_FLT},
            "zcf1": {"shape": shape_2D, "type": DTYPE_FLT},
            "tauliq": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "tauice": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "ssaliq": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "ssaice": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "ssaran": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "ssasnw": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "asyliq": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "asyice": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "asyran": {"shape": shape_nlp1, "type": type_nbandssw_flt},
            "asysnw": {"shape": shape_nlp1, "type": type_nbandssw_flt},
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
            "cldran": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "cldsnw": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "refsnw": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "extcoliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ssacoliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "asycoliq": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "extcoice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ssacoice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "asycoice": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "dgesnw": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "lcloudy": {"shape": shape_nlp1, "type": type_ngptsw_bool},
            "index": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ia": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jb": {"shape": shape_nlp1, "type": DTYPE_INT},
            "cldfmc": {"shape": shape_nlp1, "type": type_ngptsw},
            "taucw": {"shape": shape_nlp1, "type": type_nbdsw},
            "ssacw": {"shape": shape_nlp1, "type": type_nbdsw},
            "asycw": {"shape": shape_nlp1, "type": type_nbdsw},
            "cldfrc": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "plog": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fp": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fp1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ft": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "ft1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "jp1": {"shape": shape_nlp1, "type": DTYPE_INT},
            "fac00": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac01": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac10": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fac11": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "selffac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "selffrac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "forfac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "forfrac": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "indself": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indfor": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jp": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jt": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jt1": {"shape": shape_nlp1, "type": DTYPE_INT},
            "laytrop": {"shape": shape_nlp1, "type": DTYPE_BOOL},
            "id0": {"shape": shape_nlp1, "type": type_nbandssw_int},
            "id1": {"shape": shape_nlp1, "type": type_nbandssw_int},
            "ind01": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind02": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind03": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind04": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind11": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind12": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind13": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ind14": {"shape": shape_nlp1, "type": DTYPE_INT},
            "inds": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indsp": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indf": {"shape": shape_nlp1, "type": DTYPE_INT},
            "indfp": {"shape": shape_nlp1, "type": DTYPE_INT},
            "fs": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "js": {"shape": shape_nlp1, "type": DTYPE_INT},
            "jsa": {"shape": shape_nlp1, "type": DTYPE_INT},
            "colm1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "colm2": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "sfluxzen": {"shape": shape_2D, "type": type_ngptsw},
            "taug": {"shape": shape_nlp1, "type": type_ngptsw},
            "taur": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztaus": {"shape": shape_nlp1, "type": type_ngptsw},
            "zssas": {"shape": shape_nlp1, "type": type_ngptsw},
            "zasys": {"shape": shape_nlp1, "type": type_ngptsw},
            "zldbt0": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrefb": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrefd": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztrab": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztrad": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztdbt": {"shape": shape_nlp1, "type": type_ngptsw},
            "zldbt": {"shape": shape_nlp1, "type": type_ngptsw},
            "zfu": {"shape": shape_nlp1, "type": type_ngptsw},
            "zfd": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztau1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zssa1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zasy1": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztau0": {"shape": shape_nlp1, "type": type_ngptsw},
            "zssa0": {"shape": shape_nlp1, "type": type_ngptsw},
            "zasy0": {"shape": shape_nlp1, "type": type_ngptsw},
            "zasy3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zssaw": {"shape": shape_nlp1, "type": type_ngptsw},
            "zasyw": {"shape": shape_nlp1, "type": type_ngptsw},
            "zgam1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zgam2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zgam3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zgam4": {"shape": shape_nlp1, "type": type_ngptsw},
            "za1": {"shape": shape_nlp1, "type": type_ngptsw},
            "za2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zb1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zb2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrk": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrk2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrp": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrp1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrm1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrpp": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrkg1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrkg3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrkg4": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexp1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexm1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexp2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexm2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zden1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexp3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zexp4": {"shape": shape_nlp1, "type": type_ngptsw},
            "ze1r45": {"shape": shape_nlp1, "type": type_ngptsw},
            "ftind": {"shape": shape_nlp1, "type": type_ngptsw},
            "zsolar": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztdbt0": {"shape": shape_nlp1, "type": type_ngptsw},
            "zr1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zr2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zr3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zr4": {"shape": shape_nlp1, "type": type_ngptsw},
            "zr5": {"shape": shape_nlp1, "type": type_ngptsw},
            "zt1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zt2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zt3": {"shape": shape_nlp1, "type": type_ngptsw},
            "zf1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zf2": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrpp1": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrupb": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrupd": {"shape": shape_nlp1, "type": type_ngptsw},
            "ztdn": {"shape": shape_nlp1, "type": type_ngptsw},
            "zrdnd": {"shape": shape_nlp1, "type": type_ngptsw},
            "jb": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ib": {"shape": shape_nlp1, "type": DTYPE_INT},
            "ibd": {"shape": shape_nlp1, "type": DTYPE_INT},
            "itind": {"shape": shape_nlp1, "type": DTYPE_INT},
            "zb11": {"shape": shape_2D, "type": type_ngptsw},
            "zb22": {"shape": shape_2D, "type": type_ngptsw},
            "fxupc": {"shape": shape_nlp1, "type": type_nbdsw},
            "fxdnc": {"shape": shape_nlp1, "type": type_nbdsw},
            "fxup0": {"shape": shape_nlp1, "type": type_nbdsw},
            "fxdn0": {"shape": shape_nlp1, "type": type_nbdsw},
            "ftoauc": {"shape": shape_2D, "type": DTYPE_FLT},
            "ftoau0": {"shape": shape_2D, "type": DTYPE_FLT},
            "ftoadc": {"shape": shape_2D, "type": DTYPE_FLT},
            "fsfcuc": {"shape": shape_2D, "type": DTYPE_FLT},
            "fsfcu0": {"shape": shape_2D, "type": DTYPE_FLT},
            "fsfcdc": {"shape": shape_2D, "type": DTYPE_FLT},
            "fsfcd0": {"shape": shape_2D, "type": DTYPE_FLT},
            "sfbmc": {"shape": shape_2D, "type": (DTYPE_FLT, (2,))},
            "sfdfc": {"shape": shape_2D, "type": (DTYPE_FLT, (2,))},
            "sfbm0": {"shape": shape_2D, "type": (DTYPE_FLT, (2,))},
            "sfdf0": {"shape": shape_2D, "type": (DTYPE_FLT, (2,))},
            "suvbfc": {"shape": shape_2D, "type": DTYPE_FLT},
            "suvbf0": {"shape": shape_2D, "type": DTYPE_FLT},
            "flxuc": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "flxdc": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "flxu0": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "flxd0": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fnet": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fnetc": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "fnetb": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "rfdelp": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tem0": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tem1": {"shape": shape_nlp1, "type": DTYPE_FLT},
            "tem2": {"shape": shape_nlp1, "type": DTYPE_FLT},
        }

        locdict_gt4py = create_gt4py_dict_zeros(locvars)

        self.indict_gt4py = indict_gt4py
        self.locdict_gt4py = locdict_gt4py
        self.outdict_gt4py = outdict_gt4py
        self.outvars = outvars

    def _load_lookup_table_data(self):
        """
        Read in lookup table data from netcdf data that has been serialized out from
        radsw_datatb.F
        """

        lookupdict_gt4py = loadlookupdata("cldprtb", "radsw")

        # Load lookup data for setcoef
        ds = xr.open_dataset("../lookupdata/radsw_ref_data.nc")
        preflog = ds["preflog"].data
        preflog = np.tile(preflog[None, None, None, :], (npts, 1, nlp1, 1))
        tref = ds["tref"].data
        tref = np.tile(tref[None, None, None, :], (npts, 1, nlp1, 1))

        lookupdict_gt4py["preflog"] = create_storage_from_array(
            preflog, backend, shape_nlp1, (DTYPE_FLT, (59,))
        )
        lookupdict_gt4py["tref"] = create_storage_from_array(
            tref, backend, shape_nlp1, (DTYPE_FLT, (59,))
        )

        self.lookupdict_ref = loadlookupdata("sflux", "radsw")
        self.lookupdict16 = loadlookupdata("kgb16", "radsw")
        self.lookupdict17 = loadlookupdata("kgb17", "radsw")
        self.lookupdict18 = loadlookupdata("kgb18", "radsw")
        self.lookupdict19 = loadlookupdata("kgb19", "radsw")
        self.lookupdict20 = loadlookupdata("kgb20", "radsw")
        self.lookupdict21 = loadlookupdata("kgb21", "radsw")
        self.lookupdict22 = loadlookupdata("kgb22", "radsw")
        self.lookupdict23 = loadlookupdata("kgb23", "radsw")
        self.lookupdict24 = loadlookupdata("kgb24", "radsw")
        self.lookupdict25 = loadlookupdata("kgb25", "radsw")
        self.lookupdict26 = loadlookupdata("kgb26", "radsw")
        self.lookupdict27 = loadlookupdata("kgb27", "radsw")
        self.lookupdict28 = loadlookupdata("kgb28", "radsw")
        self.lookupdict29 = loadlookupdata("kgb29", "radsw")

        # Subtract one from indexing variables for Fortran -> Python conversion
        self.lookupdict_ref["ix1"] = self.lookupdict_ref["ix1"] - 1
        self.lookupdict_ref["ix2"] = self.lookupdict_ref["ix2"] - 1
        self.lookupdict_ref["ibx"] = self.lookupdict_ref["ibx"] - 1

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
            os.path.join(LOOKUP_DIR, "rand2d_tile" + str(rank) + "_sw.nc")
        )
        rand2d = ds["rand2d"].data
        cdfunc = np.zeros((npts, nlay, ngptsw))
        idxday = self._indict["idxday"]
        for n in range(npts):
            myind = idxday[n]
            if rank == 1:
                if myind > 1 and myind < 25:
                    cdfunc[myind - 1, :, :] = np.reshape(
                        rand2d[n, :], (nlay, ngptsw), order="F"
                    )
            else:
                if myind > 0 and myind < 25:
                    cdfunc[myind - 1, :, :] = np.reshape(
                        rand2d[n, :], (nlay, ngptsw), order="F"
                    )

        cdfunc = np.tile(cdfunc[:, None, :, :], (1, 1, 1, 1))
        cdfunc = np.insert(cdfunc, 0, 0, axis=2)

        self.lookupdict_gt4py["cdfunc"] = create_storage_from_array(
            cdfunc, backend, shape_nlp1, type_ngptsw
        )

    def swrad(self, rank, do_subtest=False):
        start = time.time()
        firstloop(
            self.indict_gt4py["plyr"],
            self.indict_gt4py["plvl"],
            self.indict_gt4py["tlyr"],
            self.indict_gt4py["tlvl"],
            self.indict_gt4py["qlyr"],
            self.indict_gt4py["olyr"],
            self.indict_gt4py["gasvmr"],
            self.indict_gt4py["clouds"],
            self.indict_gt4py["faersw"],
            self.indict_gt4py["sfcalb"],
            self.indict_gt4py["dz"],
            self.indict_gt4py["delp"],
            self.indict_gt4py["de_lgth"],
            self.indict_gt4py["coszen"],
            self.indict_gt4py["idxday"],
            self.indict_gt4py["solcon"],
            self.locdict_gt4py["cosz1"],
            self.locdict_gt4py["sntz1"],
            self.locdict_gt4py["ssolar"],
            self.locdict_gt4py["albbm"],
            self.locdict_gt4py["albdf"],
            self.locdict_gt4py["pavel"],
            self.locdict_gt4py["tavel"],
            self.locdict_gt4py["h2ovmr"],
            self.locdict_gt4py["o3vmr"],
            self.locdict_gt4py["coldry"],
            self.locdict_gt4py["temcol"],
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["tauae"],
            self.locdict_gt4py["ssaae"],
            self.locdict_gt4py["asyae"],
            self.locdict_gt4py["cfrac"],
            self.locdict_gt4py["cliqp"],
            self.locdict_gt4py["reliq"],
            self.locdict_gt4py["cicep"],
            self.locdict_gt4py["reice"],
            self.locdict_gt4py["cdat1"],
            self.locdict_gt4py["cdat2"],
            self.locdict_gt4py["cdat3"],
            self.locdict_gt4py["cdat4"],
            self.locdict_gt4py["zcf0"],
            self.locdict_gt4py["zcf1"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:
            outvars_firstloop = {
                "cosz1": {"fortran_shape": (npts,)},
                "sntz1": {"fortran_shape": (npts,)},
                "ssolar": {"fortran_shape": (npts,)},
                "albbm": {"fortran_shape": (npts, 2)},
                "albdf": {"fortran_shape": (npts, 2)},
                "pavel": {"fortran_shape": (npts, nlay)},
                "tavel": {"fortran_shape": (npts, nlay)},
                "h2ovmr": {"fortran_shape": (npts, nlay)},
                "o3vmr": {"fortran_shape": (npts, nlay)},
                "coldry": {"fortran_shape": (npts, nlay)},
                "temcol": {"fortran_shape": (npts, nlay)},
                "colamt": {"fortran_shape": (npts, nlay, maxgas)},
                "colmol": {"fortran_shape": (npts, nlay)},
                "tauae": {"fortran_shape": (npts, nlay, nbdsw)},
                "ssaae": {"fortran_shape": (npts, nlay, nbdsw)},
                "asyae": {"fortran_shape": (npts, nlay, nbdsw)},
                "cfrac": {"fortran_shape": (npts, nlay)},
                "cliqp": {"fortran_shape": (npts, nlay)},
                "reliq": {"fortran_shape": (npts, nlay)},
                "cicep": {"fortran_shape": (npts, nlay)},
                "reice": {"fortran_shape": (npts, nlay)},
                "cdat1": {"fortran_shape": (npts, nlay)},
                "cdat2": {"fortran_shape": (npts, nlay)},
                "cdat3": {"fortran_shape": (npts, nlay)},
                "cdat4": {"fortran_shape": (npts, nlay)},
                "zcf0": {"fortran_shape": (npts,)},
                "zcf1": {"fortran_shape": (npts,)},
            }

            outdict_firstloop = convert_gt4py_output_for_validation(
                self.locdict_gt4py, outvars_firstloop
            )
            valdict_firstloop = read_intermediate_data(
                SW_SERIALIZED_DIR, "swrad", rank, 0, "firstloop", outvars_firstloop
            )

            compare_data(outdict_firstloop, valdict_firstloop)

        self._load_random_numbers(rank)

        cldprop(
            self.locdict_gt4py["cfrac"],
            self.locdict_gt4py["cliqp"],
            self.locdict_gt4py["reliq"],
            self.locdict_gt4py["cicep"],
            self.locdict_gt4py["reice"],
            self.locdict_gt4py["cdat1"],
            self.locdict_gt4py["cdat2"],
            self.locdict_gt4py["cdat3"],
            self.locdict_gt4py["cdat4"],
            self.locdict_gt4py["zcf1"],
            self.indict_gt4py["dz"],
            self.indict_gt4py["de_lgth"],
            self.indict_gt4py["idxday"],
            self.locdict_gt4py["cldfmc"],
            self.locdict_gt4py["taucw"],
            self.locdict_gt4py["ssacw"],
            self.locdict_gt4py["asycw"],
            self.locdict_gt4py["cldfrc"],
            self.outdict_gt4py["cldtausw"],
            self.locdict_gt4py["tauliq"],
            self.locdict_gt4py["tauice"],
            self.locdict_gt4py["ssaliq"],
            self.locdict_gt4py["ssaice"],
            self.locdict_gt4py["ssaran"],
            self.locdict_gt4py["ssasnw"],
            self.locdict_gt4py["asyliq"],
            self.locdict_gt4py["asyice"],
            self.locdict_gt4py["asyran"],
            self.locdict_gt4py["asysnw"],
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
            self.locdict_gt4py["cldran"],
            self.locdict_gt4py["cldsnw"],
            self.locdict_gt4py["refsnw"],
            self.locdict_gt4py["extcoliq"],
            self.locdict_gt4py["ssacoliq"],
            self.locdict_gt4py["asycoliq"],
            self.locdict_gt4py["extcoice"],
            self.locdict_gt4py["ssacoice"],
            self.locdict_gt4py["asycoice"],
            self.locdict_gt4py["dgesnw"],
            self.locdict_gt4py["lcloudy"],
            self.locdict_gt4py["index"],
            self.locdict_gt4py["ia"],
            self.locdict_gt4py["jb"],
            self.idxebc,
            self.lookupdict_gt4py["cdfunc"],
            self.lookupdict_gt4py["extliq1"],
            self.lookupdict_gt4py["extliq2"],
            self.lookupdict_gt4py["ssaliq1"],
            self.lookupdict_gt4py["ssaliq2"],
            self.lookupdict_gt4py["asyliq1"],
            self.lookupdict_gt4py["asyliq2"],
            self.lookupdict_gt4py["extice2"],
            self.lookupdict_gt4py["ssaice2"],
            self.lookupdict_gt4py["asyice2"],
            self.lookupdict_gt4py["extice3"],
            self.lookupdict_gt4py["ssaice3"],
            self.lookupdict_gt4py["asyice3"],
            self.lookupdict_gt4py["fdlice3"],
            self.lookupdict_gt4py["abari"],
            self.lookupdict_gt4py["bbari"],
            self.lookupdict_gt4py["cbari"],
            self.lookupdict_gt4py["dbari"],
            self.lookupdict_gt4py["ebari"],
            self.lookupdict_gt4py["fbari"],
            self.lookupdict_gt4py["b0s"],
            self.lookupdict_gt4py["b1s"],
            self.lookupdict_gt4py["c0s"],
            self.lookupdict_gt4py["b0r"],
            self.lookupdict_gt4py["c0r"],
            self.lookupdict_gt4py["a0r"],
            self.lookupdict_gt4py["a1r"],
            self.lookupdict_gt4py["a0s"],
            self.lookupdict_gt4py["a1s"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:
            outvars_cldprop = {
                "cldfmc": {"fortran_shape": (npts, nlay, ngptsw)},
                "taucw": {"fortran_shape": (npts, nlay, nbdsw)},
                "ssacw": {"fortran_shape": (npts, nlay, nbdsw)},
                "asycw": {"fortran_shape": (npts, nlay, nbdsw)},
                "cldfrc": {"fortran_shape": (npts, nlay)},
            }

            outdict_cldprop = convert_gt4py_output_for_validation(
                self.locdict_gt4py, outvars_cldprop
            )
            valdict_cldprop = read_intermediate_data(
                SW_SERIALIZED_DIR, "swrad", rank, 0, "cldprop", outvars_cldprop
            )

            compare_data(outdict_cldprop, valdict_cldprop)

        setcoef(
            self.locdict_gt4py["pavel"],
            self.locdict_gt4py["tavel"],
            self.locdict_gt4py["h2ovmr"],
            self.indict_gt4py["idxday"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
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
            self.locdict_gt4py["plog"],
            self.locdict_gt4py["fp"],
            self.locdict_gt4py["fp1"],
            self.locdict_gt4py["ft"],
            self.locdict_gt4py["ft1"],
            self.locdict_gt4py["tem1"],
            self.locdict_gt4py["tem2"],
            self.locdict_gt4py["jp1"],
            self.lookupdict_gt4py["preflog"],
            self.lookupdict_gt4py["tref"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:
            outvars_setcoef = {
                "fac00": {"fortran_shape": (npts, nlay)},
                "fac01": {"fortran_shape": (npts, nlay)},
                "fac10": {"fortran_shape": (npts, nlay)},
                "fac11": {"fortran_shape": (npts, nlay)},
                "selffac": {"fortran_shape": (npts, nlay)},
                "selffrac": {"fortran_shape": (npts, nlay)},
                "forfac": {"fortran_shape": (npts, nlay)},
                "forfrac": {"fortran_shape": (npts, nlay)},
                "indself": {"fortran_shape": (npts, nlay)},
                "indfor": {"fortran_shape": (npts, nlay)},
                "jp": {"fortran_shape": (npts, nlay)},
                "jt": {"fortran_shape": (npts, nlay)},
                "jt1": {"fortran_shape": (npts, nlay)},
                "laytrop": {"fortran_shape": (npts,)},
            }

            outdict_setcoef = convert_gt4py_output_for_validation(
                self.locdict_gt4py, outvars_setcoef
            )
            valdict_setcoef = read_intermediate_data(
                SW_SERIALIZED_DIR, "swrad", rank, 0, "setcoef", outvars_setcoef
            )

            compare_data(outdict_setcoef, valdict_setcoef)

        # Compute integer indices of troposphere height
        laytropind = (
            self.locdict_gt4py["laytrop"]
            .view(np.ndarray)
            .astype(int)
            .squeeze()
            .sum(axis=1)
        )
        self.locdict_gt4py["laytropind"] = create_storage_from_array(
            laytropind[:, None] - 1, backend, shape_2D, DTYPE_INT
        )

        taumolsetup(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["jp"],
            self.locdict_gt4py["jt"],
            self.locdict_gt4py["jt1"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["laytropind"],
            self.indict_gt4py["idxday"],
            self.locdict_gt4py["sfluxzen"],
            self.layind,
            self.nspa,
            self.nspb,
            self.ngs,
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["fs"],
            self.locdict_gt4py["js"],
            self.locdict_gt4py["jsa"],
            self.locdict_gt4py["colm1"],
            self.locdict_gt4py["colm2"],
            self.lookupdict_ref["sfluxref01"],
            self.lookupdict_ref["sfluxref02"],
            self.lookupdict_ref["sfluxref03"],
            self.lookupdict_ref["layreffr"],
            self.lookupdict_ref["ix1"],
            self.lookupdict_ref["ix2"],
            self.lookupdict_ref["ibx"],
            self.lookupdict_ref["strrat"],
            self.lookupdict_ref["specwt"],
            self.lookupdict_ref["scalekur"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol16(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.lookupdict_ref["strrat"],
            self.lookupdict16["selfref"],
            self.lookupdict16["forref"],
            self.lookupdict16["absa"],
            self.lookupdict16["absb"],
            self.lookupdict16["rayl"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind03"],
            self.locdict_gt4py["ind04"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            self.locdict_gt4py["ind13"],
            self.locdict_gt4py["ind14"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol17(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.lookupdict_ref["strrat"],
            self.lookupdict17["selfref"],
            self.lookupdict17["forref"],
            self.lookupdict17["absa"],
            self.lookupdict17["absb"],
            self.lookupdict17["rayl"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind03"],
            self.locdict_gt4py["ind04"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            self.locdict_gt4py["ind13"],
            self.locdict_gt4py["ind14"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol18(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.lookupdict_ref["strrat"],
            self.lookupdict18["selfref"],
            self.lookupdict18["forref"],
            self.lookupdict18["absa"],
            self.lookupdict18["absb"],
            self.lookupdict18["rayl"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind03"],
            self.locdict_gt4py["ind04"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            self.locdict_gt4py["ind13"],
            self.locdict_gt4py["ind14"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol19(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.lookupdict_ref["strrat"],
            self.lookupdict19["selfref"],
            self.lookupdict19["forref"],
            self.lookupdict19["absa"],
            self.lookupdict19["absb"],
            self.lookupdict19["rayl"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind03"],
            self.locdict_gt4py["ind04"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            self.locdict_gt4py["ind13"],
            self.locdict_gt4py["ind14"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol20(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.lookupdict20["selfref"],
            self.lookupdict20["forref"],
            self.lookupdict20["absa"],
            self.lookupdict20["absb"],
            self.lookupdict20["absch4"],
            self.lookupdict20["rayl"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol21(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.lookupdict_ref["strrat"],
            self.lookupdict21["selfref"],
            self.lookupdict21["forref"],
            self.lookupdict21["absa"],
            self.lookupdict21["absb"],
            self.lookupdict21["rayl"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind03"],
            self.locdict_gt4py["ind04"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            self.locdict_gt4py["ind13"],
            self.locdict_gt4py["ind14"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol22(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.lookupdict_ref["strrat"],
            self.lookupdict22["selfref"],
            self.lookupdict22["forref"],
            self.lookupdict22["absa"],
            self.lookupdict22["absb"],
            self.lookupdict22["rayl"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind03"],
            self.locdict_gt4py["ind04"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            self.locdict_gt4py["ind13"],
            self.locdict_gt4py["ind14"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol23(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.lookupdict23["selfref"],
            self.lookupdict23["forref"],
            self.lookupdict23["absa"],
            self.lookupdict23["rayl"],
            self.lookupdict23["givfac"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol24(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.lookupdict_ref["strrat"],
            self.lookupdict24["selfref"],
            self.lookupdict24["forref"],
            self.lookupdict24["absa"],
            self.lookupdict24["absb"],
            self.lookupdict24["rayla"],
            self.lookupdict24["raylb"],
            self.lookupdict24["abso3a"],
            self.lookupdict24["abso3b"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind03"],
            self.locdict_gt4py["ind04"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            self.locdict_gt4py["ind13"],
            self.locdict_gt4py["ind14"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            self.locdict_gt4py["js"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol25(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.lookupdict25["absa"],
            self.lookupdict25["rayl"],
            self.lookupdict25["abso3a"],
            self.lookupdict25["abso3b"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol26(
            self.locdict_gt4py["colmol"],
            self.lookupdict26["rayl"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol27(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.lookupdict27["absa"],
            self.lookupdict27["absb"],
            self.lookupdict27["rayl"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol28(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.lookupdict_ref["strrat"],
            self.lookupdict28["absa"],
            self.lookupdict28["absb"],
            self.lookupdict28["rayl"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind03"],
            self.locdict_gt4py["ind04"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            self.locdict_gt4py["ind13"],
            self.locdict_gt4py["ind14"],
            self.locdict_gt4py["js"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        taumol29(
            self.locdict_gt4py["colamt"],
            self.locdict_gt4py["colmol"],
            self.locdict_gt4py["fac00"],
            self.locdict_gt4py["fac01"],
            self.locdict_gt4py["fac10"],
            self.locdict_gt4py["fac11"],
            self.locdict_gt4py["laytrop"],
            self.locdict_gt4py["forfac"],
            self.locdict_gt4py["forfrac"],
            self.locdict_gt4py["indfor"],
            self.locdict_gt4py["selffac"],
            self.locdict_gt4py["selffrac"],
            self.locdict_gt4py["indself"],
            self.lookupdict29["forref"],
            self.lookupdict29["absa"],
            self.lookupdict29["absb"],
            self.lookupdict29["selfref"],
            self.lookupdict29["absh2o"],
            self.lookupdict29["absco2"],
            self.lookupdict29["rayl"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["id0"],
            self.locdict_gt4py["id1"],
            self.locdict_gt4py["ind01"],
            self.locdict_gt4py["ind02"],
            self.locdict_gt4py["ind11"],
            self.locdict_gt4py["ind12"],
            self.locdict_gt4py["inds"],
            self.locdict_gt4py["indsp"],
            self.locdict_gt4py["indf"],
            self.locdict_gt4py["indfp"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:
            outvars_taumol = {
                "taug": {"fortran_shape": (npts, nlay, ngptsw)},
                "taur": {"fortran_shape": (npts, nlay, ngptsw)},
                "sfluxzen": {"fortran_shape": (npts, ngptsw)},
            }

            outdict_taumol = convert_gt4py_output_for_validation(
                self.locdict_gt4py, outvars_taumol
            )
            valdict_taumol = read_intermediate_data(
                SW_SERIALIZED_DIR, "swrad", rank, 0, "taumol", outvars_taumol
            )

            compare_data(outdict_taumol, valdict_taumol)

        spcvrtm_clearsky(
            self.locdict_gt4py["ssolar"],
            self.locdict_gt4py["cosz1"],
            self.locdict_gt4py["sntz1"],
            self.locdict_gt4py["albbm"],
            self.locdict_gt4py["albdf"],
            self.locdict_gt4py["sfluxzen"],
            self.locdict_gt4py["cldfmc"],
            self.locdict_gt4py["zcf1"],
            self.locdict_gt4py["zcf0"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["tauae"],
            self.locdict_gt4py["ssaae"],
            self.locdict_gt4py["asyae"],
            self.locdict_gt4py["taucw"],
            self.locdict_gt4py["ssacw"],
            self.locdict_gt4py["asycw"],
            self.exp_tbl,
            self.locdict_gt4py["ztaus"],
            self.locdict_gt4py["zssas"],
            self.locdict_gt4py["zasys"],
            self.locdict_gt4py["zldbt0"],
            self.locdict_gt4py["zrefb"],
            self.locdict_gt4py["zrefd"],
            self.locdict_gt4py["ztrab"],
            self.locdict_gt4py["ztrad"],
            self.locdict_gt4py["ztdbt"],
            self.locdict_gt4py["zldbt"],
            self.locdict_gt4py["zfu"],
            self.locdict_gt4py["zfd"],
            self.locdict_gt4py["ztau1"],
            self.locdict_gt4py["zssa1"],
            self.locdict_gt4py["zasy1"],
            self.locdict_gt4py["ztau0"],
            self.locdict_gt4py["zssa0"],
            self.locdict_gt4py["zasy0"],
            self.locdict_gt4py["zasy3"],
            self.locdict_gt4py["zssaw"],
            self.locdict_gt4py["zasyw"],
            self.locdict_gt4py["zgam1"],
            self.locdict_gt4py["zgam2"],
            self.locdict_gt4py["zgam3"],
            self.locdict_gt4py["zgam4"],
            self.locdict_gt4py["za1"],
            self.locdict_gt4py["za2"],
            self.locdict_gt4py["zb1"],
            self.locdict_gt4py["zb2"],
            self.locdict_gt4py["zrk"],
            self.locdict_gt4py["zrk2"],
            self.locdict_gt4py["zrp"],
            self.locdict_gt4py["zrp1"],
            self.locdict_gt4py["zrm1"],
            self.locdict_gt4py["zrpp"],
            self.locdict_gt4py["zrkg1"],
            self.locdict_gt4py["zrkg3"],
            self.locdict_gt4py["zrkg4"],
            self.locdict_gt4py["zexp1"],
            self.locdict_gt4py["zexm1"],
            self.locdict_gt4py["zexp2"],
            self.locdict_gt4py["zexm2"],
            self.locdict_gt4py["zden1"],
            self.locdict_gt4py["zexp3"],
            self.locdict_gt4py["zexp4"],
            self.locdict_gt4py["ze1r45"],
            self.locdict_gt4py["ftind"],
            self.locdict_gt4py["zsolar"],
            self.locdict_gt4py["ztdbt0"],
            self.locdict_gt4py["zr1"],
            self.locdict_gt4py["zr2"],
            self.locdict_gt4py["zr3"],
            self.locdict_gt4py["zr4"],
            self.locdict_gt4py["zr5"],
            self.locdict_gt4py["zt1"],
            self.locdict_gt4py["zt2"],
            self.locdict_gt4py["zt3"],
            self.locdict_gt4py["zf1"],
            self.locdict_gt4py["zf2"],
            self.locdict_gt4py["zrpp1"],
            self.locdict_gt4py["zrupd"],
            self.locdict_gt4py["zrupb"],
            self.locdict_gt4py["ztdn"],
            self.locdict_gt4py["zrdnd"],
            self.locdict_gt4py["zb11"],
            self.locdict_gt4py["zb22"],
            self.locdict_gt4py["jb"],
            self.locdict_gt4py["ib"],
            self.locdict_gt4py["ibd"],
            self.NGB,
            self.idxsfc,
            self.locdict_gt4py["itind"],
            self.locdict_gt4py["fxupc"],
            self.locdict_gt4py["fxdnc"],
            self.locdict_gt4py["fxup0"],
            self.locdict_gt4py["fxdn0"],
            self.locdict_gt4py["ftoauc"],
            self.locdict_gt4py["ftoau0"],
            self.locdict_gt4py["ftoadc"],
            self.locdict_gt4py["fsfcuc"],
            self.locdict_gt4py["fsfcu0"],
            self.locdict_gt4py["fsfcdc"],
            self.locdict_gt4py["fsfcd0"],
            self.locdict_gt4py["sfbmc"],
            self.locdict_gt4py["sfdfc"],
            self.locdict_gt4py["sfbm0"],
            self.locdict_gt4py["sfdf0"],
            self.locdict_gt4py["suvbfc"],
            self.locdict_gt4py["suvbf0"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        spcvrtm_allsky(
            self.locdict_gt4py["ssolar"],
            self.locdict_gt4py["cosz1"],
            self.locdict_gt4py["sntz1"],
            self.locdict_gt4py["albbm"],
            self.locdict_gt4py["albdf"],
            self.locdict_gt4py["sfluxzen"],
            self.locdict_gt4py["cldfmc"],
            self.locdict_gt4py["zcf1"],
            self.locdict_gt4py["zcf0"],
            self.locdict_gt4py["taug"],
            self.locdict_gt4py["taur"],
            self.locdict_gt4py["tauae"],
            self.locdict_gt4py["ssaae"],
            self.locdict_gt4py["asyae"],
            self.locdict_gt4py["taucw"],
            self.locdict_gt4py["ssacw"],
            self.locdict_gt4py["asycw"],
            self.exp_tbl,
            self.locdict_gt4py["ztaus"],
            self.locdict_gt4py["zssas"],
            self.locdict_gt4py["zasys"],
            self.locdict_gt4py["zldbt0"],
            self.locdict_gt4py["zrefb"],
            self.locdict_gt4py["zrefd"],
            self.locdict_gt4py["ztrab"],
            self.locdict_gt4py["ztrad"],
            self.locdict_gt4py["ztdbt"],
            self.locdict_gt4py["zldbt"],
            self.locdict_gt4py["zfu"],
            self.locdict_gt4py["zfd"],
            self.locdict_gt4py["ztau1"],
            self.locdict_gt4py["zssa1"],
            self.locdict_gt4py["zasy1"],
            self.locdict_gt4py["ztau0"],
            self.locdict_gt4py["zssa0"],
            self.locdict_gt4py["zasy0"],
            self.locdict_gt4py["zasy3"],
            self.locdict_gt4py["zssaw"],
            self.locdict_gt4py["zasyw"],
            self.locdict_gt4py["zgam1"],
            self.locdict_gt4py["zgam2"],
            self.locdict_gt4py["zgam3"],
            self.locdict_gt4py["zgam4"],
            self.locdict_gt4py["za1"],
            self.locdict_gt4py["za2"],
            self.locdict_gt4py["zb1"],
            self.locdict_gt4py["zb2"],
            self.locdict_gt4py["zrk"],
            self.locdict_gt4py["zrk2"],
            self.locdict_gt4py["zrp"],
            self.locdict_gt4py["zrp1"],
            self.locdict_gt4py["zrm1"],
            self.locdict_gt4py["zrpp"],
            self.locdict_gt4py["zrkg1"],
            self.locdict_gt4py["zrkg3"],
            self.locdict_gt4py["zrkg4"],
            self.locdict_gt4py["zexp1"],
            self.locdict_gt4py["zexm1"],
            self.locdict_gt4py["zexp2"],
            self.locdict_gt4py["zexm2"],
            self.locdict_gt4py["zden1"],
            self.locdict_gt4py["zexp3"],
            self.locdict_gt4py["zexp4"],
            self.locdict_gt4py["ze1r45"],
            self.locdict_gt4py["ftind"],
            self.locdict_gt4py["zsolar"],
            self.locdict_gt4py["ztdbt0"],
            self.locdict_gt4py["zr1"],
            self.locdict_gt4py["zr2"],
            self.locdict_gt4py["zr3"],
            self.locdict_gt4py["zr4"],
            self.locdict_gt4py["zr5"],
            self.locdict_gt4py["zt1"],
            self.locdict_gt4py["zt2"],
            self.locdict_gt4py["zt3"],
            self.locdict_gt4py["zf1"],
            self.locdict_gt4py["zf2"],
            self.locdict_gt4py["zrpp1"],
            self.locdict_gt4py["zrupd"],
            self.locdict_gt4py["zrupb"],
            self.locdict_gt4py["ztdn"],
            self.locdict_gt4py["zrdnd"],
            self.locdict_gt4py["zb11"],
            self.locdict_gt4py["zb22"],
            self.locdict_gt4py["jb"],
            self.locdict_gt4py["ib"],
            self.locdict_gt4py["ibd"],
            self.NGB,
            self.idxsfc,
            self.locdict_gt4py["itind"],
            self.locdict_gt4py["fxupc"],
            self.locdict_gt4py["fxdnc"],
            self.locdict_gt4py["fxup0"],
            self.locdict_gt4py["fxdn0"],
            self.locdict_gt4py["ftoauc"],
            self.locdict_gt4py["ftoau0"],
            self.locdict_gt4py["ftoadc"],
            self.locdict_gt4py["fsfcuc"],
            self.locdict_gt4py["fsfcu0"],
            self.locdict_gt4py["fsfcdc"],
            self.locdict_gt4py["fsfcd0"],
            self.locdict_gt4py["sfbmc"],
            self.locdict_gt4py["sfdfc"],
            self.locdict_gt4py["sfbm0"],
            self.locdict_gt4py["sfdf0"],
            self.locdict_gt4py["suvbfc"],
            self.locdict_gt4py["suvbf0"],
            domain=shape_nlp1,
            origin=default_origin,
            validate_args=validate,
        )

        if do_subtest:
            outvars_spcvrtm = {
                "fxupc": {"fortran_shape": (npts, nlp1, nbdsw)},
                "fxdnc": {"fortran_shape": (npts, nlp1, nbdsw)},
                "fxup0": {"fortran_shape": (npts, nlp1, nbdsw)},
                "fxdn0": {"fortran_shape": (npts, nlp1, nbdsw)},
                "ftoauc": {"fortran_shape": (npts,)},
                "ftoau0": {"fortran_shape": (npts,)},
                "ftoadc": {"fortran_shape": (npts,)},
                "fsfcuc": {"fortran_shape": (npts,)},
                "fsfcu0": {"fortran_shape": (npts,)},
                "fsfcdc": {"fortran_shape": (npts,)},
                "fsfcd0": {"fortran_shape": (npts,)},
                "sfbmc": {"fortran_shape": (npts, 2)},
                "sfdfc": {"fortran_shape": (npts, 2)},
                "sfbm0": {"fortran_shape": (npts, 2)},
                "sfdf0": {"fortran_shape": (npts, 2)},
                "suvbfc": {"fortran_shape": (npts,)},
                "suvbf0": {"fortran_shape": (npts,)},
            }

            outdict_spcvrtm = convert_gt4py_output_for_validation(
                self.locdict_gt4py, outvars_spcvrtm
            )
            valdict_spcvrtm = read_intermediate_data(
                SW_SERIALIZED_DIR, "swrad", rank, 0, "spcvrtm", outvars_spcvrtm
            )

            compare_data(outdict_spcvrtm, valdict_spcvrtm)

        finalloop(
            self.indict_gt4py["idxday"],
            self.indict_gt4py["delp"],
            self.locdict_gt4py["fxupc"],
            self.locdict_gt4py["fxdnc"],
            self.locdict_gt4py["fxup0"],
            self.locdict_gt4py["fxdn0"],
            self.locdict_gt4py["suvbf0"],
            self.locdict_gt4py["suvbfc"],
            self.locdict_gt4py["sfbmc"],
            self.locdict_gt4py["sfdfc"],
            self.locdict_gt4py["ftoauc"],
            self.locdict_gt4py["ftoadc"],
            self.locdict_gt4py["ftoau0"],
            self.locdict_gt4py["fsfcuc"],
            self.locdict_gt4py["fsfcdc"],
            self.locdict_gt4py["fsfcu0"],
            self.locdict_gt4py["fsfcd0"],
            self.outdict_gt4py["upfxc_t"],
            self.outdict_gt4py["dnfxc_t"],
            self.outdict_gt4py["upfx0_t"],
            self.outdict_gt4py["upfxc_s"],
            self.outdict_gt4py["dnfxc_s"],
            self.outdict_gt4py["upfx0_s"],
            self.outdict_gt4py["dnfx0_s"],
            self.outdict_gt4py["htswc"],
            self.outdict_gt4py["htsw0"],
            self.outdict_gt4py["uvbf0"],
            self.outdict_gt4py["uvbfc"],
            self.outdict_gt4py["nirbm"],
            self.outdict_gt4py["nirdf"],
            self.outdict_gt4py["visbm"],
            self.outdict_gt4py["visdf"],
            self.locdict_gt4py["rfdelp"],
            self.locdict_gt4py["fnet"],
            self.locdict_gt4py["fnetc"],
            self.locdict_gt4py["fnetb"],
            self.locdict_gt4py["flxuc"],
            self.locdict_gt4py["flxdc"],
            self.locdict_gt4py["flxu0"],
            self.locdict_gt4py["flxd0"],
            self.heatfac,
        )

        end = time.time()
        print(f"Elapsed time = {end-start}")

        outdict_final = convert_gt4py_output_for_validation(
            self.outdict_gt4py, self.outvars
        )
        valdict_final = read_data(
            os.path.join(FORTRANDATA_DIR, "SW"),
            "swrad",
            rank,
            0,
            False,
            self.outvars,
        )

        compare_data(outdict_final, valdict_final)
