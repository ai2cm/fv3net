import numpy as np
import xarray as xr
import os
import sys
from gt4py.gtscript import (
    stencil,
    computation,
    interval,
    PARALLEL,
    FORWARD,
    BACKWARD,
    mod,
    log,
    sqrt,
)

sys.path.insert(0, "..")
from util import compare_data, create_storage_from_array, create_storage_zeros
from config import *
from phys_const import con_g, con_avgd, con_amd, con_amw, amdw, amdo3
from radphysparam import iswrgas, iswcliq, iovrsw, iswcice, iswmode

from radsw.radsw_param import (
    ftiny,
    oneminus,
    s0,
    nbandssw,
    nblow,
    stpfac,
    bpade,
    flimit,
    zcrit,
    zsr3,
    od_lo,
    eps1,
    eps,
    nuvb,
    NG16,
    NG17,
    NG18,
    NG19,
    NG20,
    NG21,
    NG22,
    NG23,
    NG24,
    NG25,
    NG26,
    NG27,
    NG28,
    NG29,
    NS16,
    NS17,
    NS18,
    NS19,
    NS20,
    NS21,
    NS22,
    NS23,
    NS24,
    NS25,
    NS26,
    NS27,
    NS28,
    NS29,
)

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

# Eventually these will be provided by rad_initialize
isubcsw = 2
lhsw0 = True
lhswb = False
lflxprf = False
lfdncmp = True


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "s0": s0,
        "con_g": con_g,
        "con_avgd": con_avgd,
        "con_amd": con_amd,
        "con_amw": con_amw,
        "amdw": amdw,
        "amdo3": amdo3,
        "iswrgas": iswrgas,
        "iswcliq": iswcliq,
        "iovrsw": iovrsw,
        "nbdsw": nbdsw,
        "ftiny": ftiny,
        "oneminus": oneminus,
    },
)
def firstloop(
    plyr: FIELD_FLT,
    plvl: FIELD_FLT,
    tlyr: FIELD_FLT,
    tlvl: FIELD_FLT,
    qlyr: FIELD_FLT,
    olyr: FIELD_FLT,
    gasvmr: Field[type_10],
    clouds: Field[type_9],
    aerosols: Field[(DTYPE_FLT, (nbdsw, 3))],
    sfcalb: Field[(DTYPE_FLT, (4,))],
    dzlyr: FIELD_FLT,
    delpin: FIELD_FLT,
    de_lgth: FIELD_2D,
    cosz: FIELD_2D,
    idxday: FIELD_2DBOOL,
    solcon: float,
    cosz1: FIELD_2D,
    sntz1: FIELD_2D,
    ssolar: FIELD_2D,
    albbm: Field[(DTYPE_FLT, (2,))],
    albdf: Field[(DTYPE_FLT, (2,))],
    pavel: FIELD_FLT,
    tavel: FIELD_FLT,
    h2ovmr: FIELD_FLT,
    o3vmr: FIELD_FLT,
    coldry: FIELD_FLT,
    temcol: FIELD_FLT,
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    tauae: Field[type_nbdsw],
    ssaae: Field[type_nbdsw],
    asyae: Field[type_nbdsw],
    cfrac: FIELD_FLT,
    cliqp: FIELD_FLT,
    reliq: FIELD_FLT,
    cicep: FIELD_FLT,
    reice: FIELD_FLT,
    cdat1: FIELD_FLT,
    cdat2: FIELD_FLT,
    cdat3: FIELD_FLT,
    cdat4: FIELD_FLT,
    zcf0: FIELD_2D,
    zcf1: FIELD_2D,
):
    from __externals__ import (
        s0,
        con_g,
        con_avgd,
        con_amd,
        con_amw,
        amdw,
        amdo3,
        iswrgas,
        iswcliq,
        iovrsw,
        nbdsw,
        ftiny,
        oneminus,
    )

    with computation(FORWARD):
        with interval(0, 1):

            s0fac = solcon / s0

            if idxday:
                cosz1 = cosz
                sntz1 = 1.0 / cosz
                ssolar = s0fac * cosz

                # Prepare surface albedo: bm,df - dir,dif; 1,2 - nir,uvv.
                albbm[0, 0, 0][0] = sfcalb[0, 0, 0][0]
                albdf[0, 0, 0][0] = sfcalb[0, 0, 0][1]
                albbm[0, 0, 0][1] = sfcalb[0, 0, 0][2]
                albdf[0, 0, 0][1] = sfcalb[0, 0, 0][3]

                zcf0 = 1.0
                zcf1 = 1.0

        with interval(1, None):
            if idxday:
                tem1 = 100.0 * con_g
                tem2 = 1.0e-20 * 1.0e3 * con_avgd

                pavel = plyr
                tavel = tlyr

                h2ovmr = max(0.0, qlyr * amdw / (1.0 - qlyr))  # input specific humidity
                o3vmr = max(0.0, olyr * amdo3)  # input mass mixing ratio

                tem0 = (1.0 - h2ovmr) * con_amd + h2ovmr * con_amw
                coldry = tem2 * delpin / (tem1 * tem0 * (1.0 + h2ovmr))
                temcol = 1.0e-12 * coldry

                colamt[0, 0, 0][0] = max(0.0, coldry * h2ovmr)  # h2o
                colamt[0, 0, 0][1] = max(temcol, coldry * gasvmr[0, 0, 0][0])  # co2
                colamt[0, 0, 0][2] = max(0.0, coldry * o3vmr)  # o3
                colmol = coldry + colamt[0, 0, 0][0]

                #  --- ...  set up gas column amount, convert from volume mixing ratio
                #           to molec/cm2 based on coldry (scaled to 1.0e-20)

                if iswrgas > 0:
                    colamt[0, 0, 0][3] = max(temcol, coldry * gasvmr[0, 0, 0][1])  # n2o
                    colamt[0, 0, 0][4] = max(temcol, coldry * gasvmr[0, 0, 0][2])  # ch4
                    colamt[0, 0, 0][5] = max(temcol, coldry * gasvmr[0, 0, 0][3])  # o2
                else:
                    colamt[0, 0, 0][3] = temcol  # n2o
                    colamt[0, 0, 0][4] = temcol  # ch4
                    colamt[0, 0, 0][5] = temcol

                #  --- ...  set aerosol optical properties
                for ib in range(nbdsw):
                    tauae[0, 0, 0][ib] = aerosols[0, 0, 0][ib, 0]
                    ssaae[0, 0, 0][ib] = aerosols[0, 0, 0][ib, 1]
                    asyae[0, 0, 0][ib] = aerosols[0, 0, 0][ib, 2]

                if iswcliq > 0:  # use prognostic cloud method
                    cfrac = clouds[0, 0, 0][0]  # cloud fraction
                    cliqp = clouds[0, 0, 0][1]  # cloud liq path
                    reliq = clouds[0, 0, 0][2]  # liq partical effctive radius
                    cicep = clouds[0, 0, 0][3]  # cloud ice path
                    reice = clouds[0, 0, 0][4]  # ice partical effctive radius
                    cdat1 = clouds[0, 0, 0][5]  # cloud rain drop path
                    cdat2 = clouds[0, 0, 0][6]  # rain partical effctive radius
                    cdat3 = clouds[0, 0, 0][7]  # cloud snow path
                    cdat4 = clouds[0, 0, 0][8]  # snow partical effctive radius
                else:  # use diagnostic cloud method
                    cfrac = clouds[0, 0, 0][0]  # cloud fraction
                    cdat1 = clouds[0, 0, 0][1]  # cloud optical depth
                    cdat2 = clouds[0, 0, 0][2]  # cloud single scattering albedo
                    cdat3 = clouds[0, 0, 0][3]  # cloud asymmetry factor

                # -# Compute fractions of clear sky view:
                #    - random overlapping
                #    - max/ran overlapping
                #    - maximum overlapping

                if iovrsw == 0:
                    zcf0 = zcf0 * (1.0 - cfrac)
                elif iovrsw == 1:
                    if cfrac > ftiny:  # cloudy layer
                        zcf1 = min(zcf1, 1.0 - cfrac)
                    elif zcf1 < 1.0:  # clear layer
                        zcf0 = zcf0 * zcf1
                        zcf1 = 1.0

                elif iovrsw >= 2:
                    zcf0 = min(zcf0, 1.0 - cfrac)  # used only as clear/cloudy indicator

    with computation(FORWARD), interval(0, 1):
        if idxday:
            if iovrsw == 1:
                zcf0 = zcf0 * zcf1

            if zcf0 <= ftiny:
                zcf0 = 0.0
            if zcf0 > oneminus:
                zcf0 = 1.0

            zcf1 = 1.0 - zcf0


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "ftiny": ftiny,
        "iswcliq": iswcliq,
        "iswcice": iswcice,
        "isubcsw": isubcsw,
        "nbands": nbandssw,
        "nblow": nblow,
        "ngptsw": ngptsw,
    },
)
def cldprop(
    cfrac: FIELD_FLT,
    cliqp: FIELD_FLT,
    reliq: FIELD_FLT,
    cicep: FIELD_FLT,
    reice: FIELD_FLT,
    cdat1: FIELD_FLT,
    cdat2: FIELD_FLT,
    cdat3: FIELD_FLT,
    cdat4: FIELD_FLT,
    zcf1: FIELD_2D,
    dz: FIELD_FLT,
    delgth: FIELD_FLT,
    idxday: FIELD_2DBOOL,
    cldfmc: Field[type_ngptsw],
    taucw: Field[type_nbdsw],
    ssacw: Field[type_nbdsw],
    asycw: Field[type_nbdsw],
    cldfrc: FIELD_FLT,
    cldtau: FIELD_FLT,
    tauliq: Field[type_nbandssw_flt],
    tauice: Field[type_nbandssw_flt],
    ssaliq: Field[type_nbandssw_flt],
    ssaice: Field[type_nbandssw_flt],
    ssaran: Field[type_nbandssw_flt],
    ssasnw: Field[type_nbandssw_flt],
    asyliq: Field[type_nbandssw_flt],
    asyice: Field[type_nbandssw_flt],
    asyran: Field[type_nbandssw_flt],
    asysnw: Field[type_nbandssw_flt],
    cldf: FIELD_FLT,
    dgeice: FIELD_FLT,
    factor: FIELD_FLT,
    fint: FIELD_FLT,
    tauran: FIELD_FLT,
    tausnw: FIELD_FLT,
    cldliq: FIELD_FLT,
    refliq: FIELD_FLT,
    cldice: FIELD_FLT,
    refice: FIELD_FLT,
    cldran: FIELD_FLT,
    cldsnw: FIELD_FLT,
    refsnw: FIELD_FLT,
    extcoliq: FIELD_FLT,
    ssacoliq: FIELD_FLT,
    asycoliq: FIELD_FLT,
    extcoice: FIELD_FLT,
    ssacoice: FIELD_FLT,
    asycoice: FIELD_FLT,
    dgesnw: FIELD_FLT,
    lcloudy: Field[type_ngptsw_bool],
    index: FIELD_INT,
    ia: FIELD_INT,
    jb: FIELD_INT,
    idxebc: Field[type_nbandssw_int],
    cdfunc: Field[type_ngptsw],
    extliq1: Field[(DTYPE_FLT, (58, nbands))],
    extliq2: Field[(DTYPE_FLT, (58, nbands))],
    ssaliq1: Field[(DTYPE_FLT, (58, nbands))],
    ssaliq2: Field[(DTYPE_FLT, (58, nbands))],
    asyliq1: Field[(DTYPE_FLT, (58, nbands))],
    asyliq2: Field[(DTYPE_FLT, (58, nbands))],
    extice2: Field[(DTYPE_FLT, (43, nbands))],
    ssaice2: Field[(DTYPE_FLT, (43, nbands))],
    asyice2: Field[(DTYPE_FLT, (43, nbands))],
    extice3: Field[(DTYPE_FLT, (46, nbands))],
    ssaice3: Field[(DTYPE_FLT, (46, nbands))],
    asyice3: Field[(DTYPE_FLT, (46, nbands))],
    fdlice3: Field[(DTYPE_FLT, (46, nbands))],
    abari: Field[(DTYPE_FLT, (5,))],
    bbari: Field[(DTYPE_FLT, (5,))],
    cbari: Field[(DTYPE_FLT, (5,))],
    dbari: Field[(DTYPE_FLT, (5,))],
    ebari: Field[(DTYPE_FLT, (5,))],
    fbari: Field[(DTYPE_FLT, (5,))],
    b0s: Field[(DTYPE_FLT, (nbands,))],
    b1s: Field[(DTYPE_FLT, (nbands,))],
    c0s: Field[(DTYPE_FLT, (nbands,))],
    b0r: Field[(DTYPE_FLT, (nbands,))],
    c0r: Field[(DTYPE_FLT, (nbands,))],
    a0r: float,
    a1r: float,
    a0s: float,
    a1s: float,
):
    from __externals__ import (
        ftiny,
        iswcliq,
        iswcice,
        nbands,
        nblow,
        isubcsw,
        ngptsw,
    )

    with computation(PARALLEL), interval(1, None):
        for nb in range(nbdsw):
            if idxday and zcf1 > 0:
                ssacw[0, 0, 0][nb] = 1.0

        # Compute cloud radiative properties for a cloudy column.
        if iswcliq > 0:
            if cfrac > ftiny:
                #    - Compute optical properties for rain and snow.
                #      For rain: tauran/ssaran/asyran
                #      For snow: tausnw/ssasnw/asysnw
                #    - Calculation of absorption coefficients due to water clouds
                #      For water clouds: tauliq/ssaliq/asyliq
                #    - Calculation of absorption coefficients due to ice clouds
                #      For ice clouds: tauice/ssaice/asyice
                #    - For Prognostic cloud scheme: sum up the cloud optical property:
                #          taucw=tauliq+tauice+tauran+tausnw
                #          ssacw=ssaliq+ssaice+ssaran+ssasnw
                #          asycw=asyliq+asyice+asyran+asysnw

                cldran = cdat1
                cldsnw = cdat3
                refsnw = cdat4

                dgesnw = 1.0315 * refsnw  # for fu's snow formula

                tauran = cldran * a0r

                #  ---  if use fu's formula it needs to be normalized by snow/ice density
                #       !not use snow density = 0.1 g/cm**3 = 0.1 g/(mu * m**2)
                #       use ice density = 0.9167 g/cm**3 = 0.9167 g/(mu * m**2)
                #       1/0.9167 = 1.09087
                #       factor 1.5396=8/(3*sqrt(3)) converts reff to generalized ice particle size
                #       use newer factor value 1.0315

                if cldsnw > 0.0 and refsnw > 10.0:
                    tausnw = cldsnw * 1.09087 * (a0s + a1s / dgesnw)  # fu's formula
                else:
                    tausnw = 0.0

                for ib in range(nbands):
                    ssaran[0, 0, 0][ib] = tauran * (1.0 - b0r[0, 0, 0][ib])
                    ssasnw[0, 0, 0][ib] = tausnw * (
                        1.0 - (b0s[0, 0, 0][ib] + b1s[0, 0, 0][ib] * dgesnw)
                    )
                    asyran[0, 0, 0][ib] = ssaran[0, 0, 0][ib] * c0r[0, 0, 0][ib]
                    asysnw[0, 0, 0][ib] = ssasnw[0, 0, 0][ib] * c0s[0, 0, 0][ib]

                cldliq = cliqp
                cldice = cicep
                refliq = reliq
                refice = reice

                #  --- ...  calculation of absorption coefficients due to water clouds.

                if cldliq <= 0.0:
                    for ib2 in range(nbands):
                        tauliq[0, 0, 0][ib2] = 0.0
                        ssaliq[0, 0, 0][ib2] = 0.0
                        asyliq[0, 0, 0][ib2] = 0.0

                else:
                    factor = refliq - 1.5
                    index = max(1, min(57, factor)) - 1
                    fint = factor - (index + 1)

                    if iswcliq == 1:
                        for ib3 in range(nbands):
                            extcoliq = max(
                                0.0,
                                extliq1[0, 0, 0][index, ib3]
                                + fint
                                * (
                                    extliq1[0, 0, 0][index + 1, ib3]
                                    - extliq1[0, 0, 0][index, ib3]
                                ),
                            )
                            ssacoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaliq1[0, 0, 0][index, ib3]
                                    + fint
                                    * (
                                        ssaliq1[0, 0, 0][index + 1, ib3]
                                        - ssaliq1[0, 0, 0][index, ib3]
                                    ),
                                ),
                            )

                            asycoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    asyliq1[0, 0, 0][index, ib3]
                                    + fint
                                    * (
                                        asyliq1[0, 0, 0][index + 1, ib3]
                                        - asyliq1[0, 0, 0][index, ib3]
                                    ),
                                ),
                            )

                            tauliq[0, 0, 0][ib3] = cldliq * extcoliq
                            ssaliq[0, 0, 0][ib3] = tauliq[0, 0, 0][ib3] * ssacoliq
                            asyliq[0, 0, 0][ib3] = ssaliq[0, 0, 0][ib3] * asycoliq
                    elif iswcliq == 2:
                        for ib4 in range(nbands):
                            extcoliq = max(
                                0.0,
                                extliq2[0, 0, 0][index, ib4]
                                + fint
                                * (
                                    extliq2[0, 0, 0][index + 1, ib4]
                                    - extliq2[0, 0, 0][index, ib4]
                                ),
                            )
                            ssacoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaliq2[0, 0, 0][index, ib4]
                                    + fint
                                    * (
                                        ssaliq2[0, 0, 0][index + 1, ib4]
                                        - ssaliq2[0, 0, 0][index, ib4]
                                    ),
                                ),
                            )

                            asycoliq = max(
                                0.0,
                                min(
                                    1.0,
                                    asyliq2[0, 0, 0][index, ib4]
                                    + fint
                                    * (
                                        asyliq2[0, 0, 0][index + 1, ib4]
                                        - asyliq2[0, 0, 0][index, ib4]
                                    ),
                                ),
                            )

                            tauliq[0, 0, 0][ib4] = cldliq * extcoliq
                            ssaliq[0, 0, 0][ib4] = tauliq[0, 0, 0][ib4] * ssacoliq
                            asyliq[0, 0, 0][ib4] = ssaliq[0, 0, 0][ib4] * asycoliq

                #  --- ...  calculation of absorption coefficients due to ice clouds.
                if cldice <= 0.0:
                    for ib5 in range(nbands):
                        tauice[0, 0, 0][ib5] = 0.0
                        ssaice[0, 0, 0][ib5] = 0.0
                        asyice[0, 0, 0][ib5] = 0.0
                else:
                    #  --- ...  ebert and curry approach for all particle sizes though somewhat
                    #           unjustified for large ice particles

                    if iswcice == 1:
                        refice = min(130.0, max(13.0, refice))

                        for ib6 in range(nbands):
                            ia = (
                                idxebc[0, 0, 0][ib6] - 1
                            )  # eb_&_c band index for ice cloud coeff

                            extcoice = max(
                                0.0, abari[0, 0, 0][ia] + bbari[0, 0, 0][ia] / refice
                            )
                            ssacoice = max(
                                0.0,
                                min(
                                    1.0,
                                    1.0
                                    - cbari[0, 0, 0][ia]
                                    - dbari[0, 0, 0][ia] * refice,
                                ),
                            )
                            asycoice = max(
                                0.0,
                                min(
                                    1.0,
                                    ebari[0, 0, 0][ia] + fbari[0, 0, 0][ia] * refice,
                                ),
                            )

                            tauice[0, 0, 0][ib6] = cldice * extcoice
                            ssaice[0, 0, 0][ib6] = tauice[0, 0, 0][ib6] * ssacoice
                            asyice[0, 0, 0][ib6] = ssaice[0, 0, 0][ib6] * asycoice

                    #  --- ...  streamer approach for ice effective radius between 5.0 and 131.0 microns
                    elif iswcice == 2:
                        refice = min(131.0, max(5.0, refice))

                        factor = (refice - 2.0) / 3.0
                        index = max(1, min(42, factor)) - 1
                        fint = factor - (index + 1)

                        for ib7 in range(nbands):
                            extcoice = max(
                                0.0,
                                extice2[0, 0, 0][index, ib7]
                                + fint
                                * (
                                    extice2[0, 0, 0][index + 1, ib7]
                                    - extice2[0, 0, 0][index, ib7]
                                ),
                            )
                            ssacoice = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaice2[0, 0, 0][index, ib7]
                                    + fint
                                    * (
                                        ssaice2[0, 0, 0][index + 1, ib7]
                                        - ssaice2[0, 0, 0][index, ib7]
                                    ),
                                ),
                            )
                            asycoice = max(
                                0.0,
                                min(
                                    1.0,
                                    asyice2[0, 0, 0][index, ib7]
                                    + fint
                                    * (
                                        asyice2[0, 0, 0][index + 1, ib7]
                                        - asyice2[0, 0, 0][index, ib7]
                                    ),
                                ),
                            )

                            tauice[0, 0, 0][ib7] = cldice * extcoice
                            ssaice[0, 0, 0][ib7] = tauice[0, 0, 0][ib7] * ssacoice
                            asyice[0, 0, 0][ib7] = ssaice[0, 0, 0][ib7] * asycoice

                    #  --- ...  fu's approach for ice effective radius between 4.8 and 135 microns
                    #           (generalized effective size from 5 to 140 microns)
                    elif iswcice == 3:
                        dgeice = max(5.0, min(140.0, 1.0315 * refice))

                        factor = (dgeice - 2.0) / 3.0
                        index = max(1, min(45, factor)) - 1
                        fint = factor - (index + 1)

                        for ib8 in range(nbands):
                            extcoice = max(
                                0.0,
                                extice3[0, 0, 0][index, ib8]
                                + fint
                                * (
                                    extice3[0, 0, 0][index + 1, ib8]
                                    - extice3[0, 0, 0][index, ib8]
                                ),
                            )
                            ssacoice = max(
                                0.0,
                                min(
                                    1.0,
                                    ssaice3[0, 0, 0][index, ib8]
                                    + fint
                                    * (
                                        ssaice3[0, 0, 0][index + 1, ib8]
                                        - ssaice3[0, 0, 0][index, ib8]
                                    ),
                                ),
                            )
                            asycoice = max(
                                0.0,
                                min(
                                    1.0,
                                    asyice3[0, 0, 0][index, ib8]
                                    + fint
                                    * (
                                        asyice3[0, 0, 0][index + 1, ib8]
                                        - asyice3[0, 0, 0][index, ib8]
                                    ),
                                ),
                            )

                            tauice[0, 0, 0][ib8] = cldice * extcoice
                            ssaice[0, 0, 0][ib8] = tauice[0, 0, 0][ib8] * ssacoice
                            asyice[0, 0, 0][ib8] = ssaice[0, 0, 0][ib8] * asycoice

                for ib9 in range(nbdsw):
                    jb = nblow + ib9 - 16
                    taucw[0, 0, 0][ib9] = (
                        tauliq[0, 0, 0][jb] + tauice[0, 0, 0][jb] + tauran + tausnw
                    )
                    ssacw[0, 0, 0][ib9] = (
                        ssaliq[0, 0, 0][jb]
                        + ssaice[0, 0, 0][jb]
                        + ssaran[0, 0, 0][jb]
                        + ssasnw[0, 0, 0][jb]
                    )
                    asycw[0, 0, 0][ib9] = (
                        asyliq[0, 0, 0][jb]
                        + asyice[0, 0, 0][jb]
                        + asyran[0, 0, 0][jb]
                        + asysnw[0, 0, 0][jb]
                    )
        else:
            if cfrac > ftiny:
                for ib10 in range(nbdsw):
                    taucw[0, 0, 0][ib10] = cdat1
                    ssacw[0, 0, 0][ib10] = cdat1 * cdat2
                    asycw[0, 0, 0][ib10] = ssacw[0, 0, 0][ib10] * cdat3

        # if physparam::isubcsw > 0, call mcica_subcol() to distribute
        # cloud properties to each g-point.

        if isubcsw > 0:
            cldf = 0.0 if cfrac < ftiny else cfrac

    # This section builds mcica_subcol from the fortran into cldprop.
    # Here I've read in the generated random numbers until we figure out
    # what to do with them. This will definitely need to change in future.
    # Only the iovrlw = 1 option is ported from Fortran
    with computation(FORWARD), interval(2, None):
        tem1 = 1.0 - cldf[0, 0, -1]

        for n in range(ngptsw):
            if cdfunc[0, 0, -1][n] > tem1:
                cdfunc[0, 0, 0][n] = cdfunc[0, 0, -1][n]
            else:
                cdfunc[0, 0, 0][n] = cdfunc[0, 0, 0][n] * tem1

    with computation(PARALLEL), interval(1, None):
        tem1 = 1.0 - cldf[0, 0, 0]

        for n2 in range(ngptsw):
            if cdfunc[0, 0, 0][n2] >= tem1:
                lcloudy[0, 0, 0][n2] = 1
            else:
                lcloudy[0, 0, 0][n2] = 0

        for n3 in range(ngptsw):
            if lcloudy[0, 0, 0][n3] == 1:
                cldfmc[0, 0, 0][n3] = 1.0
            else:
                cldfmc[0, 0, 0][n3] = 0.0

        if zcf1 > 0:
            cldtau = taucw[0, 0, 0][9]


@stencil(backend=backend, rebuild=rebuild, externals={"stpfac": stpfac})
def setcoef(
    pavel: FIELD_FLT,
    tavel: FIELD_FLT,
    h2ovmr: FIELD_FLT,
    idxday: FIELD_2DBOOL,
    laytrop: FIELD_BOOL,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    plog: FIELD_FLT,
    fp: FIELD_FLT,
    fp1: FIELD_FLT,
    ft: FIELD_FLT,
    ft1: FIELD_FLT,
    tem1: FIELD_FLT,
    tem2: FIELD_FLT,
    jp1: FIELD_INT,
    preflog: Field[(DTYPE_FLT, (59,))],
    tref: Field[(DTYPE_FLT, (59,))],
):
    from __externals__ import stpfac

    with computation(PARALLEL), interval(1, None):
        if idxday:
            forfac = pavel * stpfac / (tavel * (1.0 + h2ovmr))

            #  --- ...  find the two reference pressures on either side of the
            #           layer pressure.  store them in jp and jp1.  store in fp the
            #           fraction of the difference (in ln(pressure)) between these
            #           two values that the layer pressure lies.

            plog = log(pavel)
            jp = max(1, min(58, 36.0 - 5.0 * (plog + 0.04))) - 1
            jp1 = jp + 1
            fp = 5.0 * (preflog[0, 0, 0][jp] - plog)

            #  --- ...  determine, for each reference pressure (jp and jp1), which
            #          reference temperature (these are different for each reference
            #          pressure) is nearest the layer temperature but does not exceed it.
            #          store these indices in jt and jt1, resp. store in ft (resp. ft1)
            #          the fraction of the way between jt (jt1) and the next highest
            #          reference temperature that the layer temperature falls.

            tem1 = (tavel - tref[0, 0, 0][jp]) / 15.0
            tem2 = (tavel - tref[0, 0, 0][jp1]) / 15.0
            jt = max(1, min(4, 3.0 + tem1)) - 1
            jt1 = max(1, min(4, 3.0 + tem2)) - 1
            ft = tem1 - (jt - 2)
            ft1 = tem2 - (jt1 - 2)

            #  --- ...  we have now isolated the layer ln pressure and temperature,
            #           between two reference pressures and two reference temperatures
            #           (for each reference pressure).  we multiply the pressure
            #           fraction fp with the appropriate temperature fractions to get
            #           the factors that will be needed for the interpolation that yields
            #           the optical depths (performed in routines taugbn for band n).

            fp1 = 1.0 - fp
            fac10 = fp1 * ft
            fac00 = fp1 * (1.0 - ft)
            fac11 = fp * ft1
            fac01 = fp * (1.0 - ft1)

            #  --- ...  if the pressure is less than ~100mb, perform a different
            #           set of species interpolations.

            if plog > 4.56:

                laytrop = True  # Flag for being in the troposphere

                #  --- ...  set up factors needed to separately include the water vapor
                #           foreign-continuum in the calculation of absorption coefficient.

                tem1 = (332.0 - tavel) / 36.0
                indfor = min(2, max(1, tem1))
                forfrac = tem1 - indfor

                #  --- ...  set up factors needed to separately include the water vapor
                #           self-continuum in the calculation of absorption coefficient.

                tem2 = (tavel - 188.0) / 7.2
                indself = min(9, max(1, tem2 - 7))
                selffrac = tem2 - (indself + 7)
                selffac = h2ovmr * forfac

            else:

                #  --- ...  set up factors needed to separately include the water vapor
                #           foreign-continuum in the calculation of absorption coefficient.

                tem1 = (tavel - 188.0) / 36.0
                indfor = 3
                forfrac = tem1 - 1.0

                indself = 0
                selffrac = 0.0
                selffac = 0.0

            # Add one to indices for consistency with Fortran
            jp += 1
            jt += 1
            jt1 += 1


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nbandssw": nbandssw,
        "NG16": NG16,
        "NG17": NG17,
        "NG18": NG18,
        "NG19": NG19,
        "NG20": NG20,
        "NG21": NG21,
        "NG22": NG22,
        "NG23": NG23,
        "NG24": NG24,
        "NG25": NG25,
        "NG26": NG26,
        "NG27": NG27,
        "NG28": NG28,
        "NG29": NG29,
        "oneminus": oneminus,
    },
)
def taumolsetup(
    colamt: Field[type_maxgas],
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    laytrop: FIELD_BOOL,
    laytropind: FIELD_2DINT,
    idxday: FIELD_2DBOOL,
    sfluxzen: Field[gtscript.IJ, type_ngptsw],
    layind: FIELD_INT,
    nspa: Field[gtscript.IJ, type_nbandssw_int],
    nspb: Field[gtscript.IJ, type_nbandssw_int],
    ngs: Field[gtscript.IJ, type_nbandssw_int],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    fs: FIELD_FLT,
    js: FIELD_INT,
    jsa: FIELD_INT,
    colm1: FIELD_FLT,
    colm2: FIELD_FLT,
    sfluxref01: Field[(DTYPE_FLT, (16, 1, 7))],
    sfluxref02: Field[(DTYPE_FLT, (16, 5, 2))],
    sfluxref03: Field[(DTYPE_FLT, (16, 9, 5))],
    layreffr: Field[type_nbandssw_int],
    ix1: Field[type_nbandssw_int],
    ix2: Field[type_nbandssw_int],
    ibx: Field[type_nbandssw_int],
    strrat: Field[type_nbandssw_flt],
    specwt: Field[type_nbandssw_flt],
    scalekur: float,
):
    from __externals__ import (
        nbandssw,
        NG16,
        NG17,
        NG18,
        NG19,
        NG20,
        NG21,
        NG22,
        NG23,
        NG24,
        NG25,
        NG26,
        NG27,
        NG28,
        NG29,
        oneminus,
    )

    with computation(FORWARD), interval(1, None):
        if idxday:
            for jb in range(nbandssw):
                if laytrop:
                    id0[0, 0, 0][jb] = ((jp - 1) * 5 + (jt - 1)) * nspa[0, 0][jb]
                    id1[0, 0, 0][jb] = (jp * 5 + (jt1 - 1)) * nspa[0, 0][jb]
                else:
                    id0[0, 0, 0][jb] = ((jp - 13) * 5 + (jt - 1)) * nspb[0, 0][jb]
                    id1[0, 0, 0][jb] = ((jp - 12) * 5 + (jt1 - 1)) * nspb[0, 0][jb]

            for j in range(NG16):
                sfluxzen[0, 0][ngs[0, 0][0] + j] = sfluxref01[0, 0, 0][
                    j, 0, ibx[0, 0, 0][0]
                ]
            for j2 in range(NG20):
                sfluxzen[0, 0][ngs[0, 0][4] + j2] = sfluxref01[0, 0, 0][
                    j2, 0, ibx[0, 0, 0][4]
                ]
            for j3 in range(NG23):
                sfluxzen[0, 0][ngs[0, 0][7] + j3] = sfluxref01[0, 0, 0][
                    j3, 0, ibx[0, 0, 0][7]
                ]
            for j4 in range(NG25):
                sfluxzen[0, 0][ngs[0, 0][9] + j4] = sfluxref01[0, 0, 0][
                    j4, 0, ibx[0, 0, 0][9]
                ]
            for j5 in range(NG26):
                sfluxzen[0, 0][ngs[0, 0][10] + j5] = sfluxref01[0, 0, 0][
                    j5, 0, ibx[0, 0, 0][10]
                ]
            for j6 in range(NG29):
                sfluxzen[0, 0][ngs[0, 0][13] + j6] = sfluxref01[0, 0, 0][
                    j6, 0, ibx[0, 0, 0][13]
                ]

            for j7 in range(NG27):
                sfluxzen[0, 0][ngs[0, 0][11] + j7] = (
                    scalekur * sfluxref01[0, 0, 0][j7, 0, ibx[0, 0, 0][11]]
                )

    with computation(FORWARD), interval(1, -1):
        if idxday:
            if not laytrop:
                # case default
                cond = jp < layreffr[0, 0, 0][1] and jp[0, 0, 1] >= layreffr[0, 0, 0][1]

                if cond:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][1]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][1]]
                elif sfluxzen[0, 0][ngs[0, 0][1]] == 0.0 and layind == 61:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][1]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][1]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][1] * colm2
                    specmult = specwt[0, 0, 0][1] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj in range(NG17):
                        sfluxzen[0, 0][ngs[0, 0][1] + jj] = sfluxref02[0, 0, 0][
                            jj, js, ibx[0, 0, 0][1]
                        ] + fs * (
                            sfluxref02[0, 0, 0][jj, js + 1, ibx[0, 0, 0][1]]
                            - sfluxref02[0, 0, 0][jj, js, ibx[0, 0, 0][1]]
                        )

                colm1 = 0.0
                colm2 = 0.0

                if jp < layreffr[0, 0, 0][12] and jp[0, 0, 1] >= layreffr[0, 0, 0][12]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][12]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][12]]
                elif sfluxzen[0, 0][ngs[0, 0][12]] == 0.0 and layind == 61:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][12]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][12]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][12] * colm2
                    specmult = specwt[0, 0, 0][12] * min(oneminus, colm1 / speccomb)
                    jsa = specmult
                    fsa = mod(specmult, 1.0)

                    for jj2 in range(NG28):
                        sfluxzen[0, 0][ngs[0, 0][12] + jj2] = sfluxref02[0, 0, 0][
                            jj2, jsa, ibx[0, 0, 0][12]
                        ] + fsa * (
                            sfluxref02[0, 0, 0][jj2, jsa + 1, ibx[0, 0, 0][12]]
                            - sfluxref02[0, 0, 0][jj2, jsa, ibx[0, 0, 0][12]]
                        )

            if laytrop:
                if jp < layreffr[0, 0, 0][2] and jp[0, 0, 1] >= layreffr[0, 0, 0][2]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][2]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][2]]
                if layind == laytropind and sfluxzen[0, 0][ngs[0, 0][2]] == 0.0:
                    colm1 = colamt[0, 0, 0][ix1[0, 0, 0][2]]
                    colm2 = colamt[0, 0, 0][ix2[0, 0, 0][2]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][2] * colm2
                    specmult = specwt[0, 0, 0][2] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj3 in range(NG18):
                        sfluxzen[0, 0][ngs[0, 0][2] + jj3] = sfluxref03[0, 0, 0][
                            jj3, js, ibx[0, 0, 0][2]
                        ] + fs * (
                            sfluxref03[0, 0, 0][jj3, js + 1, ibx[0, 0, 0][2]]
                            - sfluxref03[0, 0, 0][jj3, js, ibx[0, 0, 0][2]]
                        )

                colm1 = 0.0
                colm2 = 0.0

                if jp < layreffr[0, 0, 0][3] and jp[0, 0, 1] >= layreffr[0, 0, 0][3]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][3]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][3]]
                elif layind == laytropind and sfluxzen[0, 0][ngs[0, 0][3]] == 0.0:
                    colm1 = colamt[0, 0, 0][ix1[0, 0, 0][3]]
                    colm2 = colamt[0, 0, 0][ix2[0, 0, 0][3]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][3] * colm2
                    specmult = specwt[0, 0, 0][3] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj4 in range(NG19):
                        sfluxzen[0, 0][ngs[0, 0][3] + jj4] = sfluxref03[0, 0, 0][
                            jj4, js, ibx[0, 0, 0][3]
                        ] + fs * (
                            sfluxref03[0, 0, 0][jj4, js + 1, ibx[0, 0, 0][3]]
                            - sfluxref03[0, 0, 0][jj4, js, ibx[0, 0, 0][3]]
                        )

                colm1 = 0.0
                colm2 = 0.0

                if jp < layreffr[0, 0, 0][5] and jp[0, 0, 1] >= layreffr[0, 0, 0][5]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][5]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][5]]
                elif layind == laytropind and sfluxzen[0, 0][ngs[0, 0][5]] == 0.0:
                    colm1 = colamt[0, 0, 0][ix1[0, 0, 0][5]]
                    colm2 = colamt[0, 0, 0][ix2[0, 0, 0][5]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][5] * colm2
                    specmult = specwt[0, 0, 0][5] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj5 in range(NG21):
                        sfluxzen[0, 0][ngs[0, 0][5] + jj5] = sfluxref03[0, 0, 0][
                            jj5, js, ibx[0, 0, 0][5]
                        ] + fs * (
                            sfluxref03[0, 0, 0][jj5, js + 1, ibx[0, 0, 0][5]]
                            - sfluxref03[0, 0, 0][jj5, js, ibx[0, 0, 0][5]]
                        )

                colm1 = 0.0
                colm2 = 0.0

                if jp < layreffr[0, 0, 0][6] and jp[0, 0, 1] >= layreffr[0, 0, 0][6]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][6]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][6]]
                elif layind == laytropind and sfluxzen[0, 0][ngs[0, 0][6]] == 0.0:
                    colm1 = colamt[0, 0, 0][ix1[0, 0, 0][6]]
                    colm2 = colamt[0, 0, 0][ix2[0, 0, 0][6]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][6] * colm2
                    specmult = specwt[0, 0, 0][6] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj6 in range(NG22):
                        sfluxzen[0, 0][ngs[0, 0][6] + jj6] = sfluxref03[0, 0, 0][
                            jj6, js, ibx[0, 0, 0][6]
                        ] + fs * (
                            sfluxref03[0, 0, 0][jj6, js + 1, ibx[0, 0, 0][6]]
                            - sfluxref03[0, 0, 0][jj6, js, ibx[0, 0, 0][6]]
                        )

                colm1 = 0.0
                colm2 = 0.0

                if jp < layreffr[0, 0, 0][8] and jp[0, 0, 1] >= layreffr[0, 0, 0][8]:
                    colm1 = colamt[0, 0, 1][ix1[0, 0, 0][8]]
                    colm2 = colamt[0, 0, 1][ix2[0, 0, 0][8]]
                elif layind == laytropind and sfluxzen[0, 0][ngs[0, 0][8]] == 0.0:
                    colm1 = colamt[0, 0, 0][ix1[0, 0, 0][8]]
                    colm2 = colamt[0, 0, 0][ix2[0, 0, 0][8]]

                if colm1 != 0.0:
                    speccomb = colm1 + strrat[0, 0, 0][8] * colm2
                    specmult = specwt[0, 0, 0][8] * min(oneminus, colm1 / speccomb)
                    js = specmult
                    fs = mod(specmult, 1.0)

                    for jj7 in range(NG24):
                        sfluxzen[0, 0][ngs[0, 0][8] + jj7] = sfluxref03[0, 0, 0][
                            jj7, js, ibx[0, 0, 0][8]
                        ] + fs * (
                            sfluxref03[0, 0, 0][jj7, js + 1, ibx[0, 0, 0][8]]
                            - sfluxref03[0, 0, 0][jj7, js, ibx[0, 0, 0][8]]
                        )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "oneminus": oneminus,
        "NG16": NG16,
        "NS16": NS16,
    },
)
def taumol16(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG16))],
    forref: Field[(DTYPE_FLT, (3, NG16))],
    absa: Field[(DTYPE_FLT, (585, NG16))],
    absb: Field[(DTYPE_FLT, (235, NG16))],
    rayl: float,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import oneminus, NG16, NS16

    with computation(PARALLEL), interval(1, None):

        tauray = colmol * rayl

        for j in range(NG16):
            taur[0, 0, 0][NS16 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][0] * colamt[0, 0, 0][4]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs

            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][0] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][0] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10
            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG16):
                taug[0, 0, 0][NS16 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                ) + colamt[0, 0, 0][0] * (
                    selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )

        else:
            ind01 = id0[0, 0, 0][0]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][0]
            ind12 = ind11 + 1

            for j3 in range(NG16):
                taug[0, 0, 0][NS16 + j3] = colamt[0, 0, 0][4] * (
                    fac00 * absb[0, 0, 0][ind01, j3]
                    + fac10 * absb[0, 0, 0][ind02, j3]
                    + fac01 * absb[0, 0, 0][ind11, j3]
                    + fac11 * absb[0, 0, 0][ind12, j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "oneminus": oneminus,
        "NG17": NG17,
        "NS17": NS17,
    },
)
def taumol17(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG17))],
    forref: Field[(DTYPE_FLT, (3, NG17))],
    absa: Field[(DTYPE_FLT, (585, NG17))],
    absb: Field[(DTYPE_FLT, (235, NG17))],
    rayl: float,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import oneminus, NG17, NS17

    with computation(PARALLEL), interval(1, None):

        tauray = colmol * rayl

        for j in range(NG17):
            taur[0, 0, 0][NS17 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][1] * colamt[0, 0, 0][1]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][1] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][1] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG17):
                taug[0, 0, 0][NS17 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                ) + colamt[0, 0, 0][0] * (
                    selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )
        else:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][1] * colamt[0, 0, 0][1]
            specmult = 4.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][1] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 5
            ind04 = ind01 + 6
            ind11 = id1[0, 0, 0][1] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 5
            ind14 = ind11 + 6

            indf = indfor - 1
            indfp = indf + 1

            for j3 in range(NG17):
                taug[0, 0, 0][NS17 + j3] = speccomb * (
                    fac000 * absb[0, 0, 0][ind01, j3]
                    + fac100 * absb[0, 0, 0][ind02, j3]
                    + fac010 * absb[0, 0, 0][ind03, j3]
                    + fac110 * absb[0, 0, 0][ind04, j3]
                    + fac001 * absb[0, 0, 0][ind11, j3]
                    + fac101 * absb[0, 0, 0][ind12, j3]
                    + fac011 * absb[0, 0, 0][ind13, j3]
                    + fac111 * absb[0, 0, 0][ind14, j3]
                ) + colamt[0, 0, 0][0] * forfac * (
                    forref[0, 0, 0][indf, j3]
                    + forfrac * (forref[0, 0, 0][indfp, j3] - forref[0, 0, 0][indf, j3])
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "oneminus": oneminus,
        "NG18": NG18,
        "NS18": NS18,
    },
)
def taumol18(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG18))],
    forref: Field[(DTYPE_FLT, (3, NG18))],
    absa: Field[(DTYPE_FLT, (585, NG18))],
    absb: Field[(DTYPE_FLT, (235, NG18))],
    rayl: float,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import oneminus, NG18, NS18

    with computation(PARALLEL), interval(1, None):

        tauray = colmol * rayl

        for j in range(NG18):
            taur[0, 0, 0][NS18 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][2] * colamt[0, 0, 0][4]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][2] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][2] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG18):
                taug[0, 0, 0][NS18 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                ) + colamt[0, 0, 0][0] * (
                    selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )
        else:
            ind01 = id0[0, 0, 0][2]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][2]
            ind12 = ind11 + 1

            for j3 in range(NG18):
                taug[0, 0, 0][NS18 + j3] = colamt[0, 0, 0][4] * (
                    fac00 * absb[0, 0, 0][ind01, j3]
                    + fac10 * absb[0, 0, 0][ind02, j3]
                    + fac01 * absb[0, 0, 0][ind11, j3]
                    + fac11 * absb[0, 0, 0][ind12, j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "oneminus": oneminus,
        "NG19": NG19,
        "NS19": NS19,
    },
)
def taumol19(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG19))],
    forref: Field[(DTYPE_FLT, (3, NG19))],
    absa: Field[(DTYPE_FLT, (585, NG19))],
    absb: Field[(DTYPE_FLT, (235, NG19))],
    rayl: float,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import oneminus, NG19, NS19

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG19):
            taur[0, 0, 0][NS19 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][3] * colamt[0, 0, 0][1]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][3] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][3] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG19):
                taug[0, 0, 0][NS19 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                ) + colamt[0, 0, 0][0] * (
                    selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )
        else:
            ind01 = id0[0, 0, 0][3]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][3]
            ind12 = ind11 + 1

            for j3 in range(NG19):
                taug[0, 0, 0][NS19 + j3] = colamt[0, 0, 0][1] * (
                    fac00 * absb[0, 0, 0][ind01, j3]
                    + fac10 * absb[0, 0, 0][ind02, j3]
                    + fac01 * absb[0, 0, 0][ind11, j3]
                    + fac11 * absb[0, 0, 0][ind12, j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG20": NG20,
        "NS20": NS20,
    },
)
def taumol20(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    selfref: Field[(DTYPE_FLT, (10, NG20))],
    forref: Field[(DTYPE_FLT, (4, NG20))],
    absa: Field[(DTYPE_FLT, (65, NG20))],
    absb: Field[(DTYPE_FLT, (235, NG20))],
    absch4: Field[(DTYPE_FLT, (NG20,))],
    rayl: float,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import NG20, NS20

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG20):
            taur[0, 0, 0][NS20 + j] = tauray

        if laytrop:
            ind01 = id0[0, 0, 0][4]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][4]
            ind12 = ind11 + 1

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG20):
                taug[0, 0, 0][NS20 + j2] = (
                    colamt[0, 0, 0][0]
                    * (
                        (
                            fac00 * absa[0, 0, 0][ind01, j2]
                            + fac10 * absa[0, 0, 0][ind02, j2]
                            + fac01 * absa[0, 0, 0][ind11, j2]
                            + fac11 * absa[0, 0, 0][ind12, j2]
                        )
                        + selffac
                        * (
                            selfref[0, 0, 0][inds, j2]
                            + selffrac
                            * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                        )
                        + forfac
                        * (
                            forref[0, 0, 0][indf, j2]
                            + forfrac
                            * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                        )
                    )
                    + colamt[0, 0, 0][4] * absch4[0, 0, 0][j2]
                )
        else:
            ind01 = id0[0, 0, 0][4]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][4]
            ind12 = ind11 + 1

            indf = indfor - 1
            indfp = indf + 1

            for j3 in range(NG20):
                taug[0, 0, 0][NS20 + j3] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absb[0, 0, 0][ind01, j3]
                        + fac10 * absb[0, 0, 0][ind02, j3]
                        + fac01 * absb[0, 0, 0][ind11, j3]
                        + fac11 * absb[0, 0, 0][ind12, j3]
                        + forfac
                        * (
                            forref[0, 0, 0][indf, j3]
                            + forfrac
                            * (forref[0, 0, 0][indfp, j3] - forref[0, 0, 0][indf, j3])
                        )
                    )
                    + colamt[0, 0, 0][4] * absch4[0, 0, 0][j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "oneminus": oneminus,
        "NG21": NG21,
        "NS21": NS21,
    },
)
def taumol21(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG21))],
    forref: Field[(DTYPE_FLT, (4, NG21))],
    absa: Field[(DTYPE_FLT, (65, NG21))],
    absb: Field[(DTYPE_FLT, (235, NG21))],
    rayl: float,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import oneminus, NG21, NS21

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG21):
            taur[0, 0, 0][NS21 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][5] * colamt[0, 0, 0][1]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][5] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][5] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG21):
                taug[0, 0, 0][NS21 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                ) + colamt[0, 0, 0][0] * (
                    selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )
        else:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][5] * colamt[0, 0, 0][1]
            specmult = 4.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][5] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 5
            ind04 = ind01 + 6
            ind11 = id1[0, 0, 0][5] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 5
            ind14 = ind11 + 6

            indf = indfor - 1
            indfp = indf + 1

            for j3 in range(NG21):
                taug[0, 0, 0][NS21 + j3] = speccomb * (
                    fac000 * absb[0, 0, 0][ind01, j3]
                    + fac100 * absb[0, 0, 0][ind02, j3]
                    + fac010 * absb[0, 0, 0][ind03, j3]
                    + fac110 * absb[0, 0, 0][ind04, j3]
                    + fac001 * absb[0, 0, 0][ind11, j3]
                    + fac101 * absb[0, 0, 0][ind12, j3]
                    + fac011 * absb[0, 0, 0][ind13, j3]
                    + fac111 * absb[0, 0, 0][ind14, j3]
                ) + colamt[0, 0, 0][0] * forfac * (
                    forref[0, 0, 0][indf, j3]
                    + forfrac * (forref[0, 0, 0][indfp, j3] - forref[0, 0, 0][indf, j3])
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "oneminus": oneminus,
        "NG22": NG22,
        "NS22": NS22,
    },
)
def taumol22(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG22))],
    forref: Field[(DTYPE_FLT, (4, NG22))],
    absa: Field[(DTYPE_FLT, (65, NG22))],
    absb: Field[(DTYPE_FLT, (235, NG22))],
    rayl: float,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import oneminus, NG22, NS22

    with computation(PARALLEL), interval(1, None):

        #  --- ...  the following factor is the ratio of total o2 band intensity (lines
        #           and mate continuum) to o2 band intensity (line only). it is needed
        #           to adjust the optical depths since the k's include only lines.

        o2adj = 1.6
        o2tem = 4.35e-4 / (350.0 * 2.0)

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG22):
            taur[0, 0, 0][NS22 + j] = tauray

        if laytrop:
            o2cont = o2tem * colamt[0, 0, 0][5]
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][6] * colamt[0, 0, 0][5]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][6] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][6] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG22):
                taug[0, 0, 0][NS22 + j2] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ind01, j2]
                        + fac100 * absa[0, 0, 0][ind02, j2]
                        + fac010 * absa[0, 0, 0][ind03, j2]
                        + fac110 * absa[0, 0, 0][ind04, j2]
                        + fac001 * absa[0, 0, 0][ind11, j2]
                        + fac101 * absa[0, 0, 0][ind12, j2]
                        + fac011 * absa[0, 0, 0][ind13, j2]
                        + fac111 * absa[0, 0, 0][ind14, j2]
                    )
                    + colamt[0, 0, 0][0]
                    * (
                        selffac
                        * (
                            selfref[0, 0, 0][inds, j2]
                            + selffrac
                            * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                        )
                        + forfac
                        * (
                            forref[0, 0, 0][indf, j2]
                            + forfrac
                            * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                        )
                    )
                    + o2cont
                )
        else:
            o2cont = o2tem * colamt[0, 0, 0][5]

            ind01 = id0[0, 0, 0][6]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][6]
            ind12 = ind11 + 1

            for j3 in range(NG22):
                taug[0, 0, 0][NS22 + j3] = (
                    colamt[0, 0, 0][5]
                    * o2adj
                    * (
                        fac00 * absb[0, 0, 0][ind01, j3]
                        + fac10 * absb[0, 0, 0][ind02, j3]
                        + fac01 * absb[0, 0, 0][ind11, j3]
                        + fac11 * absb[0, 0, 0][ind12, j3]
                    )
                    + o2cont
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG23": NG23,
        "NS23": NS23,
    },
)
def taumol23(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    selfref: Field[(DTYPE_FLT, (10, NG23))],
    forref: Field[(DTYPE_FLT, (3, NG23))],
    absa: Field[(DTYPE_FLT, (65, NG23))],
    rayl: Field[(DTYPE_FLT, (NG23,))],
    givfac: float,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import NG23, NS23

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for j in range(NG23):
            taur[0, 0, 0][NS23 + j] = colmol * rayl[0, 0, 0][j]

        if laytrop:
            ind01 = id0[0, 0, 0][7]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][7]
            ind12 = ind11 + 1

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG23):
                taug[0, 0, 0][NS23 + j2] = colamt[0, 0, 0][0] * (
                    givfac
                    * (
                        fac00 * absa[0, 0, 0][ind01, j2]
                        + fac10 * absa[0, 0, 0][ind02, j2]
                        + fac01 * absa[0, 0, 0][ind11, j2]
                        + fac11 * absa[0, 0, 0][ind12, j2]
                    )
                    + selffac
                    * (
                        selfref[0, 0, 0][inds, j2]
                        + selffrac
                        * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                    )
                    + forfac
                    * (
                        forref[0, 0, 0][indf, j2]
                        + forfrac
                        * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                    )
                )
        else:
            for j3 in range(NG23):
                taug[0, 0, 0][NS23 + j3] = 0.0


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "oneminus": oneminus,
        "NG24": NG24,
        "NS24": NS24,
    },
)
def taumol24(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    strrat: Field[type_nbandssw_flt],
    selfref: Field[(DTYPE_FLT, (10, NG24))],
    forref: Field[(DTYPE_FLT, (3, NG24))],
    absa: Field[(DTYPE_FLT, (585, NG24))],
    absb: Field[(DTYPE_FLT, (235, NG24))],
    rayla: Field[(DTYPE_FLT, (NG24, 9))],
    raylb: Field[(DTYPE_FLT, (NG24,))],
    abso3a: Field[(DTYPE_FLT, (NG24,))],
    abso3b: Field[(DTYPE_FLT, (NG24,))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    js: FIELD_INT,
):

    from __externals__ import oneminus, NG24, NS24

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        if laytrop:
            speccomb = colamt[0, 0, 0][0] + strrat[0, 0, 0][8] * colamt[0, 0, 0][5]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][0] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][8] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][8] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG24):
                taug[0, 0, 0][NS24 + j2] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ind01, j2]
                        + fac100 * absa[0, 0, 0][ind02, j2]
                        + fac010 * absa[0, 0, 0][ind03, j2]
                        + fac110 * absa[0, 0, 0][ind04, j2]
                        + fac001 * absa[0, 0, 0][ind11, j2]
                        + fac101 * absa[0, 0, 0][ind12, j2]
                        + fac011 * absa[0, 0, 0][ind13, j2]
                        + fac111 * absa[0, 0, 0][ind14, j2]
                    )
                    + colamt[0, 0, 0][2] * abso3a[0, 0, 0][j2]
                    + colamt[0, 0, 0][0]
                    * (
                        selffac
                        * (
                            selfref[0, 0, 0][inds, j2]
                            + selffrac
                            * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                        )
                        + forfac
                        * (
                            forref[0, 0, 0][indf, j2]
                            + forfrac
                            * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                        )
                    )
                )

                taur[0, 0, 0][NS24 + j2] = colmol * (
                    rayla[0, 0, 0][j2, js - 1]
                    + fs * (rayla[0, 0, 0][j2, js] - rayla[0, 0, 0][j2, js - 1])
                )
        else:
            ind01 = id0[0, 0, 0][8]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][8]
            ind12 = ind11 + 1

            for j3 in range(NG24):
                taug[0, 0, 0][NS24 + j3] = (
                    colamt[0, 0, 0][5]
                    * (
                        fac00 * absb[0, 0, 0][ind01, j3]
                        + fac10 * absb[0, 0, 0][ind02, j3]
                        + fac01 * absb[0, 0, 0][ind11, j3]
                        + fac11 * absb[0, 0, 0][ind12, j3]
                    )
                    + colamt[0, 0, 0][2] * abso3b[0, 0, 0][j3]
                )

                taur[0, 0, 0][NS24 + j3] = colmol * raylb[0, 0, 0][j3]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG25": NG25,
        "NS25": NS25,
    },
)
def taumol25(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    absa: Field[(DTYPE_FLT, (65, NG25))],
    rayl: Field[(DTYPE_FLT, (NG25,))],
    abso3a: Field[(DTYPE_FLT, (NG25,))],
    abso3b: Field[(DTYPE_FLT, (NG25,))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
):

    from __externals__ import NG25, NS25

    with computation(PARALLEL), interval(1, None):
        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for j in range(NG25):
            taur[0, 0, 0][NS25 + j] = colmol * rayl[0, 0, 0][j]

        if laytrop:
            ind01 = id0[0, 0, 0][9]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][9]
            ind12 = ind11 + 1

            for j2 in range(NG25):
                taug[0, 0, 0][NS25 + j2] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ind01, j2]
                        + fac10 * absa[0, 0, 0][ind02, j2]
                        + fac01 * absa[0, 0, 0][ind11, j2]
                        + fac11 * absa[0, 0, 0][ind12, j2]
                    )
                    + colamt[0, 0, 0][2] * abso3a[0, 0, 0][j2]
                )
        else:
            for j3 in range(NG25):
                taug[0, 0, 0][NS25 + j3] = colamt[0, 0, 0][2] * abso3b[0, 0, 0][j3]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG26": NG26,
        "NS26": NS26,
    },
)
def taumol26(
    colmol: FIELD_FLT,
    rayl: Field[(DTYPE_FLT, (NG26,))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
):

    from __externals__ import NG26, NS26

    with computation(PARALLEL), interval(1, None):
        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for j in range(NG26):
            taug[0, 0, 0][NS26 + j] = 0.0
            taur[0, 0, 0][NS26 + j] = colmol * rayl[0, 0, 0][j]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG27": NG27,
        "NS27": NS27,
    },
)
def taumol27(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    absa: Field[(DTYPE_FLT, (65, NG27))],
    absb: Field[(DTYPE_FLT, (235, NG27))],
    rayl: Field[(DTYPE_FLT, (NG27,))],
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
):

    from __externals__ import NG27, NS27

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for j in range(NG27):
            taur[0, 0, 0][NS27 + j] = colmol * rayl[0, 0, 0][j]

        if laytrop:
            ind01 = id0[0, 0, 0][11]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][11]
            ind12 = ind11 + 1

            for j2 in range(NG27):
                taug[0, 0, 0][NS27 + j2] = colamt[0, 0, 0][2] * (
                    fac00 * absa[0, 0, 0][ind01, j2]
                    + fac10 * absa[0, 0, 0][ind02, j2]
                    + fac01 * absa[0, 0, 0][ind11, j2]
                    + fac11 * absa[0, 0, 0][ind12, j2]
                )
        else:
            ind01 = id0[0, 0, 0][11]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][11]
            ind12 = ind11 + 1

            for j3 in range(NG27):
                taug[0, 0, 0][NS27 + j3] = colamt[0, 0, 0][2] * (
                    fac00 * absb[0, 0, 0][ind01, j3]
                    + fac10 * absb[0, 0, 0][ind02, j3]
                    + fac01 * absb[0, 0, 0][ind11, j3]
                    + fac11 * absb[0, 0, 0][ind12, j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG28": NG28,
        "NS28": NS28,
        "oneminus": oneminus,
    },
)
def taumol28(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    strrat: Field[type_nbandssw_flt],
    absa: Field[(DTYPE_FLT, (585, NG28))],
    absb: Field[(DTYPE_FLT, (1175, NG28))],
    rayl: float,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind03: FIELD_INT,
    ind04: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    ind13: FIELD_INT,
    ind14: FIELD_INT,
    js: FIELD_INT,
):

    from __externals__ import NG28, NS28, oneminus

    with computation(PARALLEL), interval(1, None):
        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG28):
            taur[0, 0, 0][NS28 + j] = tauray

        if laytrop:
            speccomb = colamt[0, 0, 0][2] + strrat[0, 0, 0][12] * colamt[0, 0, 0][5]
            specmult = 8.0 * min(oneminus, colamt[0, 0, 0][2] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][12] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[0, 0, 0][12] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            for j2 in range(NG28):
                taug[0, 0, 0][NS28 + j2] = speccomb * (
                    fac000 * absa[0, 0, 0][ind01, j2]
                    + fac100 * absa[0, 0, 0][ind02, j2]
                    + fac010 * absa[0, 0, 0][ind03, j2]
                    + fac110 * absa[0, 0, 0][ind04, j2]
                    + fac001 * absa[0, 0, 0][ind11, j2]
                    + fac101 * absa[0, 0, 0][ind12, j2]
                    + fac011 * absa[0, 0, 0][ind13, j2]
                    + fac111 * absa[0, 0, 0][ind14, j2]
                )

        else:
            speccomb = colamt[0, 0, 0][2] + strrat[0, 0, 0][12] * colamt[0, 0, 0][5]
            specmult = 4.0 * min(oneminus, colamt[0, 0, 0][2] / speccomb)

            js = 1 + specmult
            fs = mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00
            fac010 = fs1 * fac10
            fac100 = fs * fac00
            fac110 = fs * fac10
            fac001 = fs1 * fac01
            fac011 = fs1 * fac11
            fac101 = fs * fac01
            fac111 = fs * fac11

            ind01 = id0[0, 0, 0][12] + js - 1
            ind02 = ind01 + 1
            ind03 = ind01 + 5
            ind04 = ind01 + 6
            ind11 = id1[0, 0, 0][12] + js - 1
            ind12 = ind11 + 1
            ind13 = ind11 + 5
            ind14 = ind11 + 6

            for j3 in range(NG28):
                taug[0, 0, 0][NS28 + j3] = speccomb * (
                    fac000 * absb[0, 0, 0][ind01, j3]
                    + fac100 * absb[0, 0, 0][ind02, j3]
                    + fac010 * absb[0, 0, 0][ind03, j3]
                    + fac110 * absb[0, 0, 0][ind04, j3]
                    + fac001 * absb[0, 0, 0][ind11, j3]
                    + fac101 * absb[0, 0, 0][ind12, j3]
                    + fac011 * absb[0, 0, 0][ind13, j3]
                    + fac111 * absb[0, 0, 0][ind14, j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "NG29": NG29,
        "NS29": NS29,
    },
)
def taumol29(
    colamt: Field[type_maxgas],
    colmol: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    laytrop: FIELD_BOOL,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forref: Field[(DTYPE_FLT, (4, NG29))],
    absa: Field[(DTYPE_FLT, (65, NG29))],
    absb: Field[(DTYPE_FLT, (235, NG29))],
    selfref: Field[(DTYPE_FLT, (10, NG29))],
    absh2o: Field[(DTYPE_FLT, (NG29,))],
    absco2: Field[(DTYPE_FLT, (NG29,))],
    rayl: float,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    id0: Field[type_nbandssw_int],
    id1: Field[type_nbandssw_int],
    ind01: FIELD_INT,
    ind02: FIELD_INT,
    ind11: FIELD_INT,
    ind12: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
):

    from __externals__ import NG29, NS29

    with computation(PARALLEL), interval(1, None):

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        tauray = colmol * rayl

        for j in range(NG29):
            taur[0, 0, 0][NS29 + j] = tauray

        if laytrop:
            ind01 = id0[0, 0, 0][13]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][13]
            ind12 = ind11 + 1

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1

            for j2 in range(NG29):
                taug[0, 0, 0][NS29 + j2] = (
                    colamt[0, 0, 0][0]
                    * (
                        (
                            fac00 * absa[0, 0, 0][ind01, j2]
                            + fac10 * absa[0, 0, 0][ind02, j2]
                            + fac01 * absa[0, 0, 0][ind11, j2]
                            + fac11 * absa[0, 0, 0][ind12, j2]
                        )
                        + selffac
                        * (
                            selfref[0, 0, 0][inds, j2]
                            + selffrac
                            * (selfref[0, 0, 0][indsp, j2] - selfref[0, 0, 0][inds, j2])
                        )
                        + forfac
                        * (
                            forref[0, 0, 0][indf, j2]
                            + forfrac
                            * (forref[0, 0, 0][indfp, j2] - forref[0, 0, 0][indf, j2])
                        )
                    )
                    + colamt[0, 0, 0][1] * absco2[0, 0, 0][j2]
                )
        else:
            ind01 = id0[0, 0, 0][13]
            ind02 = ind01 + 1
            ind11 = id1[0, 0, 0][13]
            ind12 = ind11 + 1

            for j3 in range(NG29):
                taug[0, 0, 0][NS29 + j3] = (
                    colamt[0, 0, 0][1]
                    * (
                        fac00 * absb[0, 0, 0][ind01, j3]
                        + fac10 * absb[0, 0, 0][ind02, j3]
                        + fac01 * absb[0, 0, 0][ind11, j3]
                        + fac11 * absb[0, 0, 0][ind12, j3]
                    )
                    + colamt[0, 0, 0][0] * absh2o[0, 0, 0][j3]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "ngptsw": ngptsw,
        "bpade": bpade,
        "oneminus": oneminus,
        "ftiny": ftiny,
        "flimit": flimit,
        "zcrit": zcrit,
        "zsr3": zsr3,
        "od_lo": od_lo,
        "eps1": eps1,
        "eps": eps,
    },
)
def spcvrtm_clearsky(
    ssolar: FIELD_2D,
    cosz: FIELD_2D,
    sntz: FIELD_2D,
    albbm: Field[(DTYPE_FLT, (2,))],
    albdf: Field[(DTYPE_FLT, (2,))],
    sfluxzen: Field[gtscript.IJ, type_ngptsw],
    cldfmc: Field[type_ngptsw],
    cf1: FIELD_2D,
    cf0: FIELD_2D,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    tauae: Field[type_nbdsw],
    ssaae: Field[type_nbdsw],
    asyae: Field[type_nbdsw],
    taucw: Field[type_nbdsw],
    ssacw: Field[type_nbdsw],
    asycw: Field[type_nbdsw],
    exp_tbl: Field[type_ntbmx],
    ztaus: Field[type_ngptsw],
    zssas: Field[type_ngptsw],
    zasys: Field[type_ngptsw],
    zldbt0: Field[type_ngptsw],
    zrefb: Field[type_ngptsw],
    zrefd: Field[type_ngptsw],
    ztrab: Field[type_ngptsw],
    ztrad: Field[type_ngptsw],
    ztdbt: Field[type_ngptsw],
    zldbt: Field[type_ngptsw],
    zfu: Field[type_ngptsw],
    zfd: Field[type_ngptsw],
    ztau1: Field[type_ngptsw],
    zssa1: Field[type_ngptsw],
    zasy1: Field[type_ngptsw],
    ztau0: Field[type_ngptsw],
    zssa0: Field[type_ngptsw],
    zasy0: Field[type_ngptsw],
    zasy3: Field[type_ngptsw],
    zssaw: Field[type_ngptsw],
    zasyw: Field[type_ngptsw],
    zgam1: Field[type_ngptsw],
    zgam2: Field[type_ngptsw],
    zgam3: Field[type_ngptsw],
    zgam4: Field[type_ngptsw],
    za1: Field[type_ngptsw],
    za2: Field[type_ngptsw],
    zb1: Field[type_ngptsw],
    zb2: Field[type_ngptsw],
    zrk: Field[type_ngptsw],
    zrk2: Field[type_ngptsw],
    zrp: Field[type_ngptsw],
    zrp1: Field[type_ngptsw],
    zrm1: Field[type_ngptsw],
    zrpp: Field[type_ngptsw],
    zrkg1: Field[type_ngptsw],
    zrkg3: Field[type_ngptsw],
    zrkg4: Field[type_ngptsw],
    zexp1: Field[type_ngptsw],
    zexm1: Field[type_ngptsw],
    zexp2: Field[type_ngptsw],
    zexm2: Field[type_ngptsw],
    zden1: Field[type_ngptsw],
    zexp3: Field[type_ngptsw],
    zexp4: Field[type_ngptsw],
    ze1r45: Field[type_ngptsw],
    ftind: Field[type_ngptsw],
    zsolar: Field[type_ngptsw],
    ztdbt0: Field[type_ngptsw],
    zr1: Field[type_ngptsw],
    zr2: Field[type_ngptsw],
    zr3: Field[type_ngptsw],
    zr4: Field[type_ngptsw],
    zr5: Field[type_ngptsw],
    zt1: Field[type_ngptsw],
    zt2: Field[type_ngptsw],
    zt3: Field[type_ngptsw],
    zf1: Field[type_ngptsw],
    zf2: Field[type_ngptsw],
    zrpp1: Field[type_ngptsw],
    zrupd: Field[type_ngptsw],
    zrupb: Field[type_ngptsw],
    ztdn: Field[type_ngptsw],
    zrdnd: Field[type_ngptsw],
    zb11: Field[gtscript.IJ, (DTYPE_FLT, (ngptsw,))],
    zb22: Field[gtscript.IJ, (DTYPE_FLT, (ngptsw,))],
    jb: FIELD_INT,
    ib: FIELD_INT,
    ibd: FIELD_INT,
    NGB: Field[type_ngptsw],
    idxsfc: Field[(DTYPE_INT, (14,))],
    itind: FIELD_INT,
    fxupc: Field[type_nbdsw],
    fxdnc: Field[type_nbdsw],
    fxup0: Field[type_nbdsw],
    fxdn0: Field[type_nbdsw],
    ftoauc: FIELD_2D,
    ftoau0: FIELD_2D,
    ftoadc: FIELD_2D,
    fsfcuc: FIELD_2D,
    fsfcu0: FIELD_2D,
    fsfcdc: FIELD_2D,
    fsfcd0: FIELD_2D,
    sfbmc: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfdfc: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfbm0: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfdf0: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    suvbfc: FIELD_2D,
    suvbf0: FIELD_2D,
):
    from __externals__ import (
        ngptsw,
        bpade,
        oneminus,
        ftiny,
        zcrit,
        zsr3,
        od_lo,
        eps1,
        eps,
    )

    with computation(PARALLEL), interval(...):
        #  --- ...  loop over all g-points in each band
        for jg in range(ngptsw):
            jb = NGB[0, 0, 0][jg] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1  # spectral band index

            zsolar[0, 0, 0][jg] = ssolar * sfluxzen[0, 0][jg]
            ztdbt0[0, 0, 0][jg] = 1.0

    with computation(PARALLEL), interval(-1, None):
        for n in range(ngptsw):
            ztdbt[0, 0, 0][n] = 1.0
    with computation(PARALLEL), interval(0, 1):
        for n2 in range(ngptsw):
            jb = NGB[0, 0, 0][n2] - 1
            ibd = idxsfc[0, 0, 0][jb - 15] - 1

            zldbt[0, 0, 0][n2] = 0.0
            if ibd != -1:
                zrefb[0, 0, 0][n2] = albbm[0, 0, 0][ibd]
                zrefd[0, 0, 0][n2] = albdf[0, 0, 0][ibd]
            else:
                zrefb[0, 0, 0][n2] = 0.5 * (albbm[0, 0, 0][0] + albbm[0, 0, 0][1])
                zrefd[0, 0, 0][n2] = 0.5 * (albdf[0, 0, 0][0] + albdf[0, 0, 0][1])
            ztrab[0, 0, 0][n2] = 0.0
            ztrad[0, 0, 0][n2] = 0.0

    with computation(PARALLEL), interval(0, -1):
        # Compute clear-sky optical parameters, layer reflectance and
        #    transmittance.
        #    - Set up toa direct beam and surface values (beam and diff)
        #    - Delta scaling for clear-sky condition
        #    - General two-stream expressions for physparam::iswmode
        #    - Compute homogeneous reflectance and transmittance for both
        #      conservative and non-conservative scattering
        #    - Pre-delta-scaling clear and cloudy direct beam transmittance
        #    - Call swflux() to compute the upward and downward radiation fluxes

        for n3 in range(ngptsw):
            jb = NGB[0, 0, 0][n3] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1  # spectral band index

            ztau0[0, 0, 0][n3] = max(
                ftiny, taur[0, 0, 1][n3] + taug[0, 0, 1][n3] + tauae[0, 0, 1][ib]
            )
            zssa0[0, 0, 0][n3] = (
                taur[0, 0, 1][n3] + tauae[0, 0, 1][ib] * ssaae[0, 0, 1][ib]
            )
            zasy0[0, 0, 0][n3] = (
                asyae[0, 0, 1][ib] * ssaae[0, 0, 1][ib] * tauae[0, 0, 1][ib]
            )
            zssaw[0, 0, 0][n3] = min(oneminus, zssa0[0, 0, 0][n3] / ztau0[0, 0, 0][n3])
            zasyw[0, 0, 0][n3] = zasy0[0, 0, 0][n3] / max(ftiny, zssa0[0, 0, 0][n3])

            #  --- ...  saving clear-sky quantities for later total-sky usage
            ztaus[0, 0, 0][n3] = ztau0[0, 0, 0][n3]
            zssas[0, 0, 0][n3] = zssa0[0, 0, 0][n3]
            zasys[0, 0, 0][n3] = zasy0[0, 0, 0][n3]

            #  --- ...  delta scaling for clear-sky condition
            za1[0, 0, 0][n3] = zasyw[0, 0, 0][n3] * zasyw[0, 0, 0][n3]
            za2[0, 0, 0][n3] = zssaw[0, 0, 0][n3] * za1[0, 0, 0][n3]

            ztau1[0, 0, 0][n3] = (1.0 - za2[0, 0, 0][n3]) * ztau0[0, 0, 0][n3]
            zssa1[0, 0, 0][n3] = (zssaw[0, 0, 0][n3] - za2[0, 0, 0][n3]) / (
                1.0 - za2[0, 0, 0][n3]
            )
            zasy1[0, 0, 0][n3] = zasyw[0, 0, 0][n3] / (
                1.0 + zasyw[0, 0, 0][n3]
            )  # to reduce truncation error
            zasy3[0, 0, 0][n3] = 0.75 * zasy1[0, 0, 0][n3]

            #  --- ...  general two-stream expressions
            if iswmode == 1:
                zgam1[0, 0, 0][n3] = 1.75 - zssa1[0, 0, 0][n3] * (
                    1.0 + zasy3[0, 0, 0][n3]
                )
                zgam2[0, 0, 0][n3] = -0.25 + zssa1[0, 0, 0][n3] * (
                    1.0 - zasy3[0, 0, 0][n3]
                )
                zgam3[0, 0, 0][n3] = 0.5 - zasy3[0, 0, 0][n3] * cosz
            elif iswmode == 2:  # pifm
                zgam1[0, 0, 0][n3] = 2.0 - zssa1[0, 0, 0][n3] * (
                    1.25 + zasy3[0, 0, 0][n3]
                )
                zgam2[0, 0, 0][n3] = (
                    0.75 * zssa1[0, 0, 0][n3] * (1.0 - zasy1[0, 0, 0][n3])
                )
                zgam3[0, 0, 0][n3] = 0.5 - zasy3[0, 0, 0][n3] * cosz
            elif iswmode == 3:  # discrete ordinates
                zgam1[0, 0, 0][n3] = (
                    zsr3 * (2.0 - zssa1[0, 0, 0][n3] * (1.0 + zasy1[0, 0, 0][n3])) * 0.5
                )
                zgam2[0, 0, 0][n3] = (
                    zsr3 * zssa1[0, 0, 0][n3] * (1.0 - zasy1[0, 0, 0][n3]) * 0.5
                )
                zgam3[0, 0, 0][n3] = (1.0 - zsr3 * zasy1[0, 0, 0][n3] * cosz) * 0.5

            zgam4[0, 0, 0][n3] = 1.0 - zgam3[0, 0, 0][n3]

            #  --- ...  compute homogeneous reflectance and transmittance

            if zssaw[0, 0, 0][n3] >= zcrit:  # for conservative scattering
                za1[0, 0, 0][n3] = zgam1[0, 0, 0][n3] * cosz - zgam3[0, 0, 0][n3]
                za2[0, 0, 0][n3] = zgam1[0, 0, 0][n3] * ztau1[0, 0, 0][n3]

                #  --- ...  use exponential lookup table for transmittance, or expansion
                #           of exponential for low optical depth

                zb1[0, 0, 0][n3] = min(ztau1[0, 0, 0][n3] * sntz, 500.0)
                if zb1[0, 0, 0][n3] <= od_lo:
                    zb2[0, 0, 0][n3] = (
                        1.0
                        - zb1[0, 0, 0][n3]
                        + 0.5 * zb1[0, 0, 0][n3] * zb1[0, 0, 0][n3]
                    )
                else:
                    ftind[0, 0, 0][n3] = zb1[0, 0, 0][n3] / (bpade + zb1[0, 0, 0][n3])
                    itind = ftind[0, 0, 0][n3] * ntbmx + 0.5
                    zb2[0, 0, 0][n3] = exp_tbl[0, 0, 0][itind]

    with computation(FORWARD), interval(1, None):
        for nn3 in range(ngptsw):
            jb = NGB[0, 0, 0][nn3] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1

            if zssaw[0, 0, -1][nn3] >= zcrit:
                # collimated beam
                zrefb[0, 0, 0][nn3] = max(
                    0.0,
                    min(
                        1.0,
                        (
                            za2[0, 0, -1][nn3]
                            - za1[0, 0, -1][nn3] * (1.0 - zb2[0, 0, -1][nn3])
                        )
                        / (1.0 + za2[0, 0, -1][nn3]),
                    ),
                )
                ztrab[0, 0, 0][nn3] = max(0.0, min(1.0, 1.0 - zrefb[0, 0, 0][nn3]))

                #      ...  isotropic incidence
                zrefd[0, 0, 0][nn3] = max(
                    0.0, min(1.0, za2[0, 0, -1][nn3] / (1.0 + za2[0, 0, -1][nn3]))
                )
                ztrad[0, 0, 0][nn3] = max(0.0, min(1.0, 1.0 - zrefd[0, 0, 0][nn3]))

    with computation(BACKWARD), interval(0, -1):
        for n4 in range(ngptsw):
            jb = NGB[0, 0, 0][n4] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1

            if zssaw[0, 0, 0][n4] < zcrit:  # for non-conservative scattering
                za1[0, 0, 0][n4] = (
                    zgam1[0, 0, 0][n4] * zgam4[0, 0, 0][n4]
                    + zgam2[0, 0, 0][n4] * zgam3[0, 0, 0][n4]
                )
                za2[0, 0, 0][n4] = (
                    zgam1[0, 0, 0][n4] * zgam3[0, 0, 0][n4]
                    + zgam2[0, 0, 0][n4] * zgam4[0, 0, 0][n4]
                )
                zrk[0, 0, 0][n4] = sqrt(
                    (zgam1[0, 0, 0][n4] - zgam2[0, 0, 0][n4])
                    * (zgam1[0, 0, 0][n4] + zgam2[0, 0, 0][n4])
                )
                zrk2[0, 0, 0][n4] = 2.0 * zrk[0, 0, 0][n4]

                zrp[0, 0, 0][n4] = zrk[0, 0, 0][n4] * cosz
                zrp1[0, 0, 0][n4] = 1.0 + zrp[0, 0, 0][n4]
                zrm1[0, 0, 0][n4] = 1.0 - zrp[0, 0, 0][n4]
                zrpp1[0, 0, 0][n4] = 1.0 - zrp[0, 0, 0][n4] * zrp[0, 0, 0][n4]
                tmp = max(flimit, abs(zrpp1[0, 0, 0][n4]))
                zrpp[0, 0, 0][n4] = (
                    tmp if zrpp1[0, 0, 0][n4] >= 0 else -tmp
                )  # avoid numerical singularity
                zrkg1[0, 0, 0][n4] = zrk[0, 0, 0][n4] + zgam1[0, 0, 0][n4]
                zrkg3[0, 0, 0][n4] = zrk[0, 0, 0][n4] * zgam3[0, 0, 0][n4]
                zrkg4[0, 0, 0][n4] = zrk[0, 0, 0][n4] * zgam4[0, 0, 0][n4]

                zr1[0, 0, 0][n4] = zrm1[0, 0, 0][n4] * (
                    za2[0, 0, 0][n4] + zrkg3[0, 0, 0][n4]
                )
                zr2[0, 0, 0][n4] = zrp1[0, 0, 0][n4] * (
                    za2[0, 0, 0][n4] - zrkg3[0, 0, 0][n4]
                )
                zr3[0, 0, 0][n4] = zrk2[0, 0, 0][n4] * (
                    zgam3[0, 0, 0][n4] - za2[0, 0, 0][n4] * cosz
                )
                zr4[0, 0, 0][n4] = zrpp[0, 0, 0][n4] * zrkg1[0, 0, 0][n4]
                zr5[0, 0, 0][n4] = zrpp[0, 0, 0][n4] * (
                    zrk[0, 0, 0][n4] - zgam1[0, 0, 0][n4]
                )

                zt1[0, 0, 0][n4] = zrp1[0, 0, 0][n4] * (
                    za1[0, 0, 0][n4] + zrkg4[0, 0, 0][n4]
                )
                zt2[0, 0, 0][n4] = zrm1[0, 0, 0][n4] * (
                    za1[0, 0, 0][n4] - zrkg4[0, 0, 0][n4]
                )
                zt3[0, 0, 0][n4] = zrk2[0, 0, 0][n4] * (
                    zgam4[0, 0, 0][n4] + za1[0, 0, 0][n4] * cosz
                )

                #  --- ...  use exponential lookup table for transmittance, or expansion
                #           of exponential for low optical depth

                zb1[0, 0, 0][n4] = min(zrk[0, 0, 0][n4] * ztau1[0, 0, 0][n4], 500.0)
                if zb1[0, 0, 0][n4] <= od_lo:
                    zexm1[0, 0, 0][n4] = (
                        1.0
                        - zb1[0, 0, 0][n4]
                        + 0.5 * zb1[0, 0, 0][n4] * zb1[0, 0, 0][n4]
                    )
                else:
                    ftind[0, 0, 0][n4] = zb1[0, 0, 0][n4] / (bpade + zb1[0, 0, 0][n4])
                    itind = ftind[0, 0, 0][n4] * ntbmx + 0.5
                    zexm1[0, 0, 0][n4] = exp_tbl[0, 0, 0][itind]

                zexp1[0, 0, 0][n4] = 1.0 / zexm1[0, 0, 0][n4]

                zb2[0, 0, 0][n4] = min(sntz * ztau1[0, 0, 0][n4], 500.0)
                if zb2[0, 0, 0][n4] <= od_lo:
                    zexm2[0, 0, 0][n4] = (
                        1.0
                        - zb2[0, 0, 0][n4]
                        + 0.5 * zb2[0, 0, 0][n4] * zb2[0, 0, 0][n4]
                    )
                else:
                    ftind[0, 0, 0][n4] = zb2[0, 0, 0][n4] / (bpade + zb2[0, 0, 0][n4])
                    itind = ftind[0, 0, 0][n4] * ntbmx + 0.5
                    zexm2[0, 0, 0][n4] = exp_tbl[0, 0, 0][itind]

                zexp2[0, 0, 0][n4] = 1.0 / zexm2[0, 0, 0][n4]
                ze1r45[0, 0, 0][n4] = (
                    zr4[0, 0, 0][n4] * zexp1[0, 0, 0][n4]
                    + zr5[0, 0, 0][n4] * zexm1[0, 0, 0][n4]
                )

    with computation(BACKWARD), interval(1, None):
        for nn4 in range(ngptsw):

            if zssaw[0, 0, -1][nn4] < zcrit:
                #      ...  collimated beam
                if ze1r45[0, 0, -1][nn4] >= -eps1 and ze1r45[0, 0, -1][nn4] <= eps1:
                    zrefb[0, 0, 0][nn4] = eps1
                    ztrab[0, 0, 0][nn4] = zexm2[0, 0, -1][nn4]
                else:
                    zden1[0, 0, 0][nn4] = zssa1[0, 0, -1][nn4] / ze1r45[0, 0, -1][nn4]
                    zrefb[0, 0, 0][nn4] = max(
                        0.0,
                        min(
                            1.0,
                            (
                                zr1[0, 0, -1][nn4] * zexp1[0, 0, -1][nn4]
                                - zr2[0, 0, -1][nn4] * zexm1[0, 0, -1][nn4]
                                - zr3[0, 0, -1][nn4] * zexm2[0, 0, -1][nn4]
                            )
                            * zden1[0, 0, 0][nn4],
                        ),
                    )
                    ztrab[0, 0, 0][nn4] = max(
                        0.0,
                        min(
                            1.0,
                            zexm2[0, 0, -1][nn4]
                            * (
                                1.0
                                - (
                                    zt1[0, 0, -1][nn4] * zexp1[0, 0, -1][nn4]
                                    - zt2[0, 0, -1][nn4] * zexm1[0, 0, -1][nn4]
                                    - zt3[0, 0, -1][nn4] * zexp2[0, 0, -1][nn4]
                                )
                                * zden1[0, 0, 0][nn4]
                            ),
                        ),
                    )

                #      ...  diffuse beam
                zden1[0, 0, 0][nn4] = zr4[0, 0, -1][nn4] / (
                    ze1r45[0, 0, -1][nn4] * zrkg1[0, 0, -1][nn4]
                )
                zrefd[0, 0, 0][nn4] = max(
                    0.0,
                    min(
                        1.0,
                        zgam2[0, 0, -1][nn4]
                        * (zexp1[0, 0, -1][nn4] - zexm1[0, 0, -1][nn4])
                        * zden1[0, 0, 0][nn4],
                    ),
                )
                ztrad[0, 0, 0][nn4] = max(
                    0.0, min(1.0, zrk2[0, 0, -1][nn4] * zden1[0, 0, 0][nn4])
                )

    with computation(BACKWARD), interval(0, -1):

        #  --- ...  direct beam transmittance. use exponential lookup table
        #           for transmittance, or expansion of exponential for low
        #           optical depth

        for n5 in range(ngptsw):
            jb = NGB[0, 0, 0][n5] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1

            zr1[0, 0, 0][n5] = ztau1[0, 0, 0][n5] * sntz
            if zr1[0, 0, 0][n5] <= od_lo:
                zexp3[0, 0, 0][n5] = (
                    1.0 - zr1[0, 0, 0][n5] + 0.5 * zr1[0, 0, 0][n5] * zr1[0, 0, 0][n5]
                )
            else:
                ftind[0, 0, 0][n5] = zr1[0, 0, 0][n5] / (bpade + zr1[0, 0, 0][n5])
                itind = max(0, min(ntbmx, 0.5 + ntbmx * ftind[0, 0, 0][n5]))
                zexp3[0, 0, 0][n5] = exp_tbl[0, 0, 0][itind]

            ztdbt[0, 0, 0][n5] = zexp3[0, 0, 0][n5] * ztdbt[0, 0, 1][n5]

            #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
            #           (must use 'orig', unscaled cloud optical depth)

            zr1[0, 0, 0][n5] = ztau0[0, 0, 0][n5] * sntz
            if zr1[0, 0, 0][n5] <= od_lo:
                zexp4[0, 0, 0][n5] = (
                    1.0 - zr1[0, 0, 0][n5] + 0.5 * zr1[0, 0, 0][n5] * zr1[0, 0, 0][n5]
                )
            else:
                ftind[0, 0, 0][n5] = zr1[0, 0, 0][n5] / (bpade + zr1[0, 0, 0][n5])
                itind = max(0, min(ntbmx, 0.5 + ntbmx * ftind[0, 0, 0][n5]))
                zexp4[0, 0, 0][n5] = exp_tbl[0, 0, 0][itind]

            zldbt0[0, 0, 0][n5] = zexp4[0, 0, 0][n5]
            ztdbt0[0, 0, 0][n5] = zexp4[0, 0, 0][n5] * ztdbt0[0, 0, 1][n5]

    with computation(PARALLEL), interval(1, None):
        for nn5 in range(ngptsw):
            zldbt[0, 0, 0][nn5] = zexp3[0, 0, -1][nn5]

    # Need to incorporate the vrtqdr subroutine here, since I don't know
    # of a way to specify the interval within a gtscript function. Could be pulled
    # out later if we can call a stencil from a stencil

    with computation(FORWARD):
        with interval(0, 1):
            for nnn5 in range(ngptsw):
                # Link lowest layer with surface.
                zrupb[0, 0, 0][nnn5] = zrefb[0, 0, 0][nnn5]  # direct beam
                zrupd[0, 0, 0][nnn5] = zrefd[0, 0, 0][nnn5]  # diffused

        with interval(1, None):
            for n6 in range(ngptsw):
                zden1[0, 0, 0][n6] = 1.0 / (
                    1.0 - zrupd[0, 0, -1][n6] * zrefd[0, 0, 0][n6]
                )

                zrupb[0, 0, 0][n6] = (
                    zrefb[0, 0, 0][n6]
                    + (
                        ztrad[0, 0, 0][n6]
                        * (
                            (ztrab[0, 0, 0][n6] - zldbt[0, 0, 0][n6])
                            * zrupd[0, 0, -1][n6]
                            + zldbt[0, 0, 0][n6] * zrupb[0, 0, -1][n6]
                        )
                    )
                    * zden1[0, 0, 0][n6]
                )
                zrupd[0, 0, 0][n6] = (
                    zrefd[0, 0, 0][n6]
                    + ztrad[0, 0, 0][n6]
                    * ztrad[0, 0, 0][n6]
                    * zrupd[0, 0, -1][n6]
                    * zden1[0, 0, 0][n6]
                )

    with computation(PARALLEL):
        with interval(-2, -1):
            for n7 in range(ngptsw):
                ztdn[0, 0, 0][n7] = ztrab[0, 0, 1][n7]
                zrdnd[0, 0, 0][n7] = zrefd[0, 0, 1][n7]
        with interval(-1, None):
            for n8 in range(ngptsw):
                ztdn[0, 0, 0][n8] = 1.0
                zrdnd[0, 0, 0][n8] = 0.0

    with computation(BACKWARD), interval(0, -1):
        for n9 in range(ngptsw):
            zden1[0, 0, 0][n9] = 1.0 / (1.0 - zrefd[0, 0, 1][n9] * zrdnd[0, 0, 1][n9])
            ztdn[0, 0, 0][n9] = (
                ztdbt[0, 0, 1][n9] * ztrab[0, 0, 1][n9]
                + (
                    ztrad[0, 0, 1][n9]
                    * (
                        (ztdn[0, 0, 1][n9] - ztdbt[0, 0, 1][n9])
                        + ztdbt[0, 0, 1][n9] * zrefb[0, 0, 1][n9] * zrdnd[0, 0, 1][n9]
                    )
                )
                * zden1[0, 0, 0][n9]
            )
            zrdnd[0, 0, 0][n9] = (
                zrefd[0, 0, 1][n9]
                + ztrad[0, 0, 1][n9]
                * ztrad[0, 0, 1][n9]
                * zrdnd[0, 0, 1][n9]
                * zden1[0, 0, 0][n9]
            )

    # Up and down-welling fluxes at levels.
    with computation(PARALLEL), interval(...):
        for n10 in range(ngptsw):
            jb = NGB[0, 0, 0][n10] - 1
            ib = jb + 1 - nblow

            zden1[0, 0, 0][n10] = 1.0 / (
                1.0 - zrdnd[0, 0, 0][n10] * zrupd[0, 0, 0][n10]
            )
            zfu[0, 0, 0][n10] = (
                ztdbt[0, 0, 0][n10] * zrupb[0, 0, 0][n10]
                + (ztdn[0, 0, 0][n10] - ztdbt[0, 0, 0][n10]) * zrupd[0, 0, 0][n10]
            ) * zden1[0, 0, 0][n10]
            zfd[0, 0, 0][n10] = (
                ztdbt[0, 0, 0][n10]
                + (
                    ztdn[0, 0, 0][n10]
                    - ztdbt[0, 0, 0][n10]
                    + ztdbt[0, 0, 0][n10] * zrupb[0, 0, 0][n10] * zrdnd[0, 0, 0][n10]
                )
                * zden1[0, 0, 0][n10]
            )

            #  --- ...  compute upward and downward fluxes at levels
            fxup0[0, 0, 0][ib] = (
                fxup0[0, 0, 0][ib] + zsolar[0, 0, 0][n10] * zfu[0, 0, 0][n10]
            )
            fxdn0[0, 0, 0][ib] = (
                fxdn0[0, 0, 0][ib] + zsolar[0, 0, 0][n10] * zfd[0, 0, 0][n10]
            )

    with computation(FORWARD), interval(0, 1):
        for n11 in range(ngptsw):
            jb = NGB[0, 0, 0][n11] - 1
            ib = jb + 1 - nblow
            ibd = idxsfc[0, 0, 0][jb - 15] - 1

            # --- ...  surface downward beam/diffuse flux components
            zb11[0, 0][n11] = zsolar[0, 0, 0][n11] * ztdbt0[0, 0, 0][n11]
            zb22[0, 0][n11] = zsolar[0, 0, 0][n11] * (
                zfd[0, 0, 0][n11] - ztdbt0[0, 0, 0][n11]
            )

            if ibd != -1:
                sfbm0[0, 0][ibd] = sfbm0[0, 0][ibd] + zb11[0, 0][n11]
                sfdf0[0, 0][ibd] = sfdf0[0, 0][ibd] + zb22[0, 0][n11]
            else:
                zf1[0, 0, 0][n11] = 0.5 * zb11[0, 0][n11]
                zf2[0, 0, 0][n11] = 0.5 * zb22[0, 0][n11]
                sfbm0[0, 0][0] = sfbm0[0, 0][0] + zf1[0, 0, 0][n11]
                sfdf0[0, 0][0] = sfdf0[0, 0][0] + zf2[0, 0, 0][n11]
                sfbm0[0, 0][1] = sfbm0[0, 0][1] + zf1[0, 0, 0][n11]
                sfdf0[0, 0][1] = sfdf0[0, 0][1] + zf2[0, 0, 0][n11]

            zldbt[0, 0, 0][n11] = 0.0


ibd0 = nuvb - nblow


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "ngptsw": ngptsw,
        "bpade": bpade,
        "oneminus": oneminus,
        "ftiny": ftiny,
        "flimit": flimit,
        "zcrit": zcrit,
        "zsr3": zsr3,
        "od_lo": od_lo,
        "eps1": eps1,
        "eps": eps,
        "ibd0": ibd0,
    },
)
def spcvrtm_allsky(
    ssolar: FIELD_2D,
    cosz: FIELD_2D,
    sntz: FIELD_2D,
    albbm: Field[(DTYPE_FLT, (2,))],
    albdf: Field[(DTYPE_FLT, (2,))],
    sfluxzen: Field[gtscript.IJ, type_ngptsw],
    cldfmc: Field[type_ngptsw],
    cf1: FIELD_2D,
    cf0: FIELD_2D,
    taug: Field[type_ngptsw],
    taur: Field[type_ngptsw],
    tauae: Field[type_nbdsw],
    ssaae: Field[type_nbdsw],
    asyae: Field[type_nbdsw],
    taucw: Field[type_nbdsw],
    ssacw: Field[type_nbdsw],
    asycw: Field[type_nbdsw],
    exp_tbl: Field[type_ntbmx],
    ztaus: Field[type_ngptsw],
    zssas: Field[type_ngptsw],
    zasys: Field[type_ngptsw],
    zldbt0: Field[type_ngptsw],
    zrefb: Field[type_ngptsw],
    zrefd: Field[type_ngptsw],
    ztrab: Field[type_ngptsw],
    ztrad: Field[type_ngptsw],
    ztdbt: Field[type_ngptsw],
    zldbt: Field[type_ngptsw],
    zfu: Field[type_ngptsw],
    zfd: Field[type_ngptsw],
    ztau1: Field[type_ngptsw],
    zssa1: Field[type_ngptsw],
    zasy1: Field[type_ngptsw],
    ztau0: Field[type_ngptsw],
    zssa0: Field[type_ngptsw],
    zasy0: Field[type_ngptsw],
    zasy3: Field[type_ngptsw],
    zssaw: Field[type_ngptsw],
    zasyw: Field[type_ngptsw],
    zgam1: Field[type_ngptsw],
    zgam2: Field[type_ngptsw],
    zgam3: Field[type_ngptsw],
    zgam4: Field[type_ngptsw],
    za1: Field[type_ngptsw],
    za2: Field[type_ngptsw],
    zb1: Field[type_ngptsw],
    zb2: Field[type_ngptsw],
    zrk: Field[type_ngptsw],
    zrk2: Field[type_ngptsw],
    zrp: Field[type_ngptsw],
    zrp1: Field[type_ngptsw],
    zrm1: Field[type_ngptsw],
    zrpp: Field[type_ngptsw],
    zrkg1: Field[type_ngptsw],
    zrkg3: Field[type_ngptsw],
    zrkg4: Field[type_ngptsw],
    zexp1: Field[type_ngptsw],
    zexm1: Field[type_ngptsw],
    zexp2: Field[type_ngptsw],
    zexm2: Field[type_ngptsw],
    zden1: Field[type_ngptsw],
    zexp3: Field[type_ngptsw],
    zexp4: Field[type_ngptsw],
    ze1r45: Field[type_ngptsw],
    ftind: Field[type_ngptsw],
    zsolar: Field[type_ngptsw],
    ztdbt0: Field[type_ngptsw],
    zr1: Field[type_ngptsw],
    zr2: Field[type_ngptsw],
    zr3: Field[type_ngptsw],
    zr4: Field[type_ngptsw],
    zr5: Field[type_ngptsw],
    zt1: Field[type_ngptsw],
    zt2: Field[type_ngptsw],
    zt3: Field[type_ngptsw],
    zf1: Field[type_ngptsw],
    zf2: Field[type_ngptsw],
    zrpp1: Field[type_ngptsw],
    zrupd: Field[type_ngptsw],
    zrupb: Field[type_ngptsw],
    ztdn: Field[type_ngptsw],
    zrdnd: Field[type_ngptsw],
    zb11: Field[gtscript.IJ, (DTYPE_FLT, (ngptsw,))],
    zb22: Field[gtscript.IJ, (DTYPE_FLT, (ngptsw,))],
    jb: FIELD_INT,
    ib: FIELD_INT,
    ibd: FIELD_INT,
    NGB: Field[type_ngptsw],
    idxsfc: Field[(DTYPE_INT, (14,))],
    itind: FIELD_INT,
    fxupc: Field[type_nbdsw],
    fxdnc: Field[type_nbdsw],
    fxup0: Field[type_nbdsw],
    fxdn0: Field[type_nbdsw],
    ftoauc: FIELD_2D,
    ftoau0: FIELD_2D,
    ftoadc: FIELD_2D,
    fsfcuc: FIELD_2D,
    fsfcu0: FIELD_2D,
    fsfcdc: FIELD_2D,
    fsfcd0: FIELD_2D,
    sfbmc: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfdfc: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfbm0: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfdf0: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    suvbfc: FIELD_2D,
    suvbf0: FIELD_2D,
):
    from __externals__ import (
        ngptsw,
        bpade,
        oneminus,
        ftiny,
        zcrit,
        zsr3,
        od_lo,
        eps1,
        eps,
    )

    # -# Compute total sky optical parameters, layer reflectance and
    #    transmittance.
    #    - Set up toa direct beam and surface values (beam and diff)
    #    - Delta scaling for total-sky condition
    #    - General two-stream expressions for physparam::iswmode
    #    - Compute homogeneous reflectance and transmittance for
    #      conservative scattering and non-conservative scattering
    #    - Pre-delta-scaling clear and cloudy direct beam transmittance
    #    - Call swflux() to compute the upward and downward radiation fluxes

    with computation(BACKWARD):
        with interval(-1, None):
            if cf1 > eps:
                for m0 in range(ngptsw):
                    ztdbt0[0, 0, 0][m0] = 1.0
        with interval(0, -1):
            if cf1 > eps:
                for m in range(ngptsw):
                    jb = NGB[0, 0, 0][m] - 1
                    ib = jb + 1 - nblow
                    ibd = idxsfc[0, 0, 0][jb - 15] - 1

                    if cldfmc[0, 0, 1][m] > ftiny:
                        ztau0[0, 0, 0][m] = ztaus[0, 0, 0][m] + taucw[0, 0, 1][ib]
                        zssa0[0, 0, 0][m] = zssas[0, 0, 0][m] + ssacw[0, 0, 1][ib]
                        zasy0[0, 0, 0][m] = zasys[0, 0, 0][m] + asycw[0, 0, 1][ib]
                        zssaw[0, 0, 0][m] = min(
                            oneminus, zssa0[0, 0, 0][m] / ztau0[0, 0, 0][m]
                        )
                        zasyw[0, 0, 0][m] = zasy0[0, 0, 0][m] / max(
                            ftiny, zssa0[0, 0, 0][m]
                        )

                        #  --- ...  delta scaling for total-sky condition
                        za1[0, 0, 0][m] = zasyw[0, 0, 0][m] * zasyw[0, 0, 0][m]
                        za2[0, 0, 0][m] = zssaw[0, 0, 0][m] * za1[0, 0, 0][m]

                        ztau1[0, 0, 0][m] = (1.0 - za2[0, 0, 0][m]) * ztau0[0, 0, 0][m]
                        zssa1[0, 0, 0][m] = (zssaw[0, 0, 0][m] - za2[0, 0, 0][m]) / (
                            1.0 - za2[0, 0, 0][m]
                        )
                        zasy1[0, 0, 0][m] = zasyw[0, 0, 0][m] / (
                            1.0 + zasyw[0, 0, 0][m]
                        )
                        zasy3[0, 0, 0][m] = 0.75 * zasy1[0, 0, 0][m]

                        #  --- ...  general two-stream expressions
                        if iswmode == 1:
                            zgam1[0, 0, 0][m] = 1.75 - zssa1[0, 0, 0][m] * (
                                1.0 + zasy3[0, 0, 0][m]
                            )
                            zgam2[0, 0, 0][m] = -0.25 + zssa1[0, 0, 0][m] * (
                                1.0 - zasy3[0, 0, 0][m]
                            )
                            zgam3[0, 0, 0][m] = 0.5 - zasy3[0, 0, 0][m] * cosz
                        elif iswmode == 2:  # pifm
                            zgam1[0, 0, 0][m] = 2.0 - zssa1[0, 0, 0][m] * (
                                1.25 + zasy3[0, 0, 0][m]
                            )
                            zgam2[0, 0, 0][m] = (
                                0.75 * zssa1[0, 0, 0][m] * (1.0 - zasy1[0, 0, 0][m])
                            )
                            zgam3[0, 0, 0][m] = 0.5 - zasy3[0, 0, 0][m] * cosz
                        elif iswmode == 3:  # discrete ordinates
                            zgam1[0, 0, 0][m] = (
                                zsr3
                                * (2.0 - zssa1[0, 0, 0][m] * (1.0 + zasy1[0, 0, 0][m]))
                                * 0.5
                            )
                            zgam2[0, 0, 0][m] = (
                                zsr3
                                * zssa1[0, 0, 0][m]
                                * (1.0 - zasy1[0, 0, 0][m])
                                * 0.5
                            )
                            zgam3[0, 0, 0][m] = (
                                1.0 - zsr3 * zasy1[0, 0, 0][m] * cosz
                            ) * 0.5

                        zgam4[0, 0, 0][m] = 1.0 - zgam3[0, 0, 0][m]

                        #  --- ...  compute homogeneous reflectance and transmittance
                        if zssaw[0, 0, 0][m] >= zcrit:  # for conservative scattering
                            za1[0, 0, 0][m] = (
                                zgam1[0, 0, 0][m] * cosz - zgam3[0, 0, 0][m]
                            )
                            za2[0, 0, 0][m] = zgam1[0, 0, 0][m] * ztau1[0, 0, 0][m]

                            #  --- ...  use exponential lookup table for transmittance, or expansion
                            #           of exponential for low optical depth

                            zb1[0, 0, 0][m] = min(ztau1[0, 0, 0][m] * sntz, 500.0)
                            if zb1[0, 0, 0][m] <= od_lo:
                                zb2[0, 0, 0][m] = (
                                    1.0
                                    - zb1[0, 0, 0][m]
                                    + 0.5 * zb1[0, 0, 0][m] * zb1[0, 0, 0][m]
                                )
                            else:
                                ftind[0, 0, 0][m] = zb1[0, 0, 0][m] / (
                                    bpade + zb1[0, 0, 0][m]
                                )
                                itind = ftind[0, 0, 0][m] * ntbmx + 0.5
                                zb2[0, 0, 0][m] = exp_tbl[0, 0, 0][itind]

    with computation(BACKWARD), interval(1, None):
        if cf1 > eps:
            for m2 in range(ngptsw):
                if cldfmc[0, 0, 0][m2] > ftiny:
                    if zssaw[0, 0, -1][m2] >= zcrit:
                        #      ...  collimated beam
                        zrefb[0, 0, 0][m2] = max(
                            0.0,
                            min(
                                1.0,
                                (
                                    za2[0, 0, -1][m2]
                                    - za1[0, 0, -1][m2] * (1.0 - zb2[0, 0, -1][m2])
                                )
                                / (1.0 + za2[0, 0, -1][m2]),
                            ),
                        )
                        ztrab[0, 0, 0][m2] = max(
                            0.0, min(1.0, 1.0 - zrefb[0, 0, 0][m2])
                        )

                        #      ...  isotropic incidence
                        zrefd[0, 0, 0][m2] = max(
                            0.0, min(1.0, za2[0, 0, -1][m2] / (1.0 + za2[0, 0, -1][m2]))
                        )
                        ztrad[0, 0, 0][m2] = max(
                            0.0, min(1.0, 1.0 - zrefd[0, 0, 0][m2])
                        )

    with computation(BACKWARD), interval(0, -1):
        if cf1 > eps:
            for m3 in range(ngptsw):
                if cldfmc[0, 0, 1][m3] > ftiny:
                    if zssaw[0, 0, 0][m3] < zcrit:  # for non-conservative scattering
                        za1[0, 0, 0][m3] = (
                            zgam1[0, 0, 0][m3] * zgam4[0, 0, 0][m3]
                            + zgam2[0, 0, 0][m3] * zgam3[0, 0, 0][m3]
                        )
                        za2[0, 0, 0][m3] = (
                            zgam1[0, 0, 0][m3] * zgam3[0, 0, 0][m3]
                            + zgam2[0, 0, 0][m3] * zgam4[0, 0, 0][m3]
                        )
                        zrk[0, 0, 0][m3] = sqrt(
                            (zgam1[0, 0, 0][m3] - zgam2[0, 0, 0][m3])
                            * (zgam1[0, 0, 0][m3] + zgam2[0, 0, 0][m3])
                        )
                        zrk2[0, 0, 0][m3] = 2.0 * zrk[0, 0, 0][m3]

                        zrp[0, 0, 0][m3] = zrk[0, 0, 0][m3] * cosz
                        zrp1[0, 0, 0][m3] = 1.0 + zrp[0, 0, 0][m3]
                        zrm1[0, 0, 0][m3] = 1.0 - zrp[0, 0, 0][m3]
                        zrpp1[0, 0, 0][m3] = 1.0 - zrp[0, 0, 0][m3] * zrp[0, 0, 0][m3]
                        tmp = max(flimit, abs(zrpp1[0, 0, 0][m3]))
                        zrpp[0, 0, 0][m3] = (
                            tmp if zrpp1[0, 0, 0][m3] >= 0 else -tmp
                        )  # avoid numerical singularity
                        zrkg1[0, 0, 0][m3] = zrk[0, 0, 0][m3] + zgam1[0, 0, 0][m3]
                        zrkg3[0, 0, 0][m3] = zrk[0, 0, 0][m3] * zgam3[0, 0, 0][m3]
                        zrkg4[0, 0, 0][m3] = zrk[0, 0, 0][m3] * zgam4[0, 0, 0][m3]

                        zr1[0, 0, 0][m3] = zrm1[0, 0, 0][m3] * (
                            za2[0, 0, 0][m3] + zrkg3[0, 0, 0][m3]
                        )
                        zr2[0, 0, 0][m3] = zrp1[0, 0, 0][m3] * (
                            za2[0, 0, 0][m3] - zrkg3[0, 0, 0][m3]
                        )
                        zr3[0, 0, 0][m3] = zrk2[0, 0, 0][m3] * (
                            zgam3[0, 0, 0][m3] - za2[0, 0, 0][m3] * cosz
                        )
                        zr4[0, 0, 0][m3] = zrpp[0, 0, 0][m3] * zrkg1[0, 0, 0][m3]
                        zr5[0, 0, 0][m3] = zrpp[0, 0, 0][m3] * (
                            zrk[0, 0, 0][m3] - zgam1[0, 0, 0][m3]
                        )

                        zt1[0, 0, 0][m3] = zrp1[0, 0, 0][m3] * (
                            za1[0, 0, 0][m3] + zrkg4[0, 0, 0][m3]
                        )
                        zt2[0, 0, 0][m3] = zrm1[0, 0, 0][m3] * (
                            za1[0, 0, 0][m3] - zrkg4[0, 0, 0][m3]
                        )
                        zt3[0, 0, 0][m3] = zrk2[0, 0, 0][m3] * (
                            zgam4[0, 0, 0][m3] + za1[0, 0, 0][m3] * cosz
                        )

                        #  --- ...  use exponential lookup table for transmittance, or expansion
                        #           of exponential for low optical depth

                        zb1[0, 0, 0][m3] = min(
                            zrk[0, 0, 0][m3] * ztau1[0, 0, 0][m3], 500.0
                        )
                        if zb1[0, 0, 0][m3] <= od_lo:
                            zexm1[0, 0, 0][m3] = (
                                1.0
                                - zb1[0, 0, 0][m3]
                                + 0.5 * zb1[0, 0, 0][m3] * zb1[0, 0, 0][m3]
                            )
                        else:
                            ftind[0, 0, 0][m3] = zb1[0, 0, 0][m3] / (
                                bpade + zb1[0, 0, 0][m3]
                            )
                            itind = ftind[0, 0, 0][m3] * ntbmx + 0.5
                            zexm1[0, 0, 0][m3] = exp_tbl[0, 0, 0][itind]

                        zexp1[0, 0, 0][m3] = 1.0 / zexm1[0, 0, 0][m3]

                        zb2[0, 0, 0][m3] = min(ztau1[0, 0, 0][m3] * sntz, 500.0)
                        if zb2[0, 0, 0][m3] <= od_lo:
                            zexm2[0, 0, 0][m3] = (
                                1.0
                                - zb2[0, 0, 0][m3]
                                + 0.5 * zb2[0, 0, 0][m3] * zb2[0, 0, 0][m3]
                            )
                        else:
                            ftind[0, 0, 0][m3] = zb2[0, 0, 0][m3] / (
                                bpade + zb2[0, 0, 0][m3]
                            )
                            itind = ftind[0, 0, 0][m3] * ntbmx + 0.5
                            zexm2[0, 0, 0][m3] = exp_tbl[0, 0, 0][itind]

                        zexp2[0, 0, 0][m3] = 1.0 / zexm2[0, 0, 0][m3]
                        ze1r45[0, 0, 0][m3] = (
                            zr4[0, 0, 0][m3] * zexp1[0, 0, 0][m3]
                            + zr5[0, 0, 0][m3] * zexm1[0, 0, 0][m3]
                        )

    with computation(BACKWARD), interval(1, None):
        if cf1 > eps:
            for mm3 in range(ngptsw):
                if cldfmc[0, 0, 0][mm3] > ftiny:
                    if zssaw[0, 0, -1][mm3] < zcrit:
                        #      ...  collimated beam
                        if (
                            ze1r45[0, 0, -1][mm3] >= -eps1
                            and ze1r45[0, 0, -1][mm3] <= eps1
                        ):
                            zrefb[0, 0, 0][mm3] = eps1
                            ztrab[0, 0, 0][mm3] = zexm2[0, 0, -1][mm3]
                        else:
                            zden1[0, 0, 0][mm3] = (
                                zssa1[0, 0, -1][mm3] / ze1r45[0, 0, -1][mm3]
                            )
                            zrefb[0, 0, 0][mm3] = max(
                                0.0,
                                min(
                                    1.0,
                                    (
                                        zr1[0, 0, -1][mm3] * zexp1[0, 0, -1][mm3]
                                        - zr2[0, 0, -1][mm3] * zexm1[0, 0, -1][mm3]
                                        - zr3[0, 0, -1][mm3] * zexm2[0, 0, -1][mm3]
                                    )
                                    * zden1[0, 0, 0][mm3],
                                ),
                            )
                            ztrab[0, 0, 0][mm3] = max(
                                0.0,
                                min(
                                    1.0,
                                    zexm2[0, 0, -1][mm3]
                                    * (
                                        1.0
                                        - (
                                            zt1[0, 0, -1][mm3] * zexp1[0, 0, -1][mm3]
                                            - zt2[0, 0, -1][mm3] * zexm1[0, 0, -1][mm3]
                                            - zt3[0, 0, -1][mm3] * zexp2[0, 0, -1][mm3]
                                        )
                                        * zden1[0, 0, 0][mm3]
                                    ),
                                ),
                            )

                        #      ...  diffuse beam
                        zden1[0, 0, 0][mm3] = zr4[0, 0, -1][mm3] / (
                            ze1r45[0, 0, -1][mm3] * zrkg1[0, 0, -1][mm3]
                        )
                        zrefd[0, 0, 0][mm3] = max(
                            0.0,
                            min(
                                1.0,
                                zgam2[0, 0, -1][mm3]
                                * (zexp1[0, 0, -1][mm3] - zexm1[0, 0, -1][mm3])
                                * zden1[0, 0, 0][mm3],
                            ),
                        )
                        ztrad[0, 0, 0][mm3] = max(
                            0.0, min(1.0, zrk2[0, 0, -1][mm3] * zden1[0, 0, 0][mm3])
                        )

    with computation(BACKWARD), interval(0, -1):
        if cf1 > eps:
            for m4 in range(ngptsw):
                if cldfmc[0, 0, 1][m4] > ftiny:
                    #  --- ...  direct beam transmittance. use exponential lookup table
                    #           for transmittance, or expansion of exponential for low
                    #           optical depth

                    zr1[0, 0, 0][m4] = ztau1[0, 0, 0][m4] * sntz
                    if zr1[0, 0, 0][m4] <= od_lo:
                        zexp3[0, 0, 0][m4] = (
                            1.0
                            - zr1[0, 0, 0][m4]
                            + 0.5 * zr1[0, 0, 0][m4] * zr1[0, 0, 0][m4]
                        )
                    else:
                        ftind[0, 0, 0][m4] = zr1[0, 0, 0][m4] / (
                            bpade + zr1[0, 0, 0][m4]
                        )
                        itind = max(0, min(ntbmx, 0.5 + ntbmx * ftind[0, 0, 0][m4]))
                        zexp3[0, 0, 0][m4] = exp_tbl[0, 0, 0][itind]

                    ztdbt[0, 0, 0][m4] = zexp3[0, 0, 0][m4] * ztdbt[0, 0, 1][m4]

                    #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                    #           (must use 'orig', unscaled cloud optical depth)

                    zr1[0, 0, 0][m4] = ztau0[0, 0, 0][m4] * sntz
                    if zr1[0, 0, 0][m4] <= od_lo:
                        zexp4[0, 0, 0][m4] = (
                            1.0
                            - zr1[0, 0, 0][m4]
                            + 0.5 * zr1[0, 0, 0][m4] * zr1[0, 0, 0][m4]
                        )
                    else:
                        ftind[0, 0, 0][m4] = zr1[0, 0, 0][m4] / (
                            bpade + zr1[0, 0, 0][m4]
                        )
                        itind = max(0, min(ntbmx, 0.5 + ntbmx * ftind[0, 0, 0][m4]))
                        zexp4[0, 0, 0][m4] = exp_tbl[0, 0, 0][itind]

                    ztdbt0[0, 0, 0][m4] = zexp4[0, 0, 0][m4] * ztdbt0[0, 0, 1][m4]

                else:
                    #  --- ...  direct beam transmittance
                    ztdbt[0, 0, 0][m4] = zldbt[0, 0, 1][m4] * ztdbt[0, 0, 1][m4]

                    #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                    ztdbt0[0, 0, 0][m4] = zldbt0[0, 0, 0][m4] * ztdbt0[0, 0, 1][m4]

    # Need to incorporate the vrtqdr subroutine here, since I don't know
    # of a way to specify the interval within a gtscript function. Could be pulled
    # out later

    with computation(FORWARD):
        with interval(0, 1):
            if cf1 > eps:
                for m5 in range(ngptsw):
                    # Link lowest layer with surface.
                    zrupb[0, 0, 0][m5] = zrefb[0, 0, 0][m5]  # direct beam
                    zrupd[0, 0, 0][m5] = zrefd[0, 0, 0][m5]  # diffused

        with interval(1, None):
            if cf1 > eps:
                for m6 in range(ngptsw):
                    if cldfmc[0, 0, 0][m6] > ftiny:
                        zldbt[0, 0, 0][m6] = zexp3[0, 0, -1][m6]

                    zden1[0, 0, 0][m6] = 1.0 / (
                        1.0 - zrupd[0, 0, -1][m6] * zrefd[0, 0, 0][m6]
                    )

                    zrupb[0, 0, 0][m6] = (
                        zrefb[0, 0, 0][m6]
                        + (
                            ztrad[0, 0, 0][m6]
                            * (
                                (ztrab[0, 0, 0][m6] - zldbt[0, 0, 0][m6])
                                * zrupd[0, 0, -1][m6]
                                + zldbt[0, 0, 0][m6] * zrupb[0, 0, -1][m6]
                            )
                        )
                        * zden1[0, 0, 0][m6]
                    )
                    zrupd[0, 0, 0][m6] = (
                        zrefd[0, 0, 0][m6]
                        + ztrad[0, 0, 0][m6]
                        * ztrad[0, 0, 0][m6]
                        * zrupd[0, 0, -1][m6]
                        * zden1[0, 0, 0][m6]
                    )

                    ztdn[0, 0, 0][m6] = 0.0
                    zrdnd[0, 0, 0][m6] = 0.0

    with computation(PARALLEL):
        with interval(-2, -1):
            if cf1 > eps:
                for m7 in range(ngptsw):
                    ztdn[0, 0, 0][m7] = ztrab[0, 0, 1][m7]
                    zrdnd[0, 0, 0][m7] = zrefd[0, 0, 1][m7]
        with interval(-1, None):
            if cf1 > eps:
                for m8 in range(ngptsw):
                    ztdn[0, 0, 0][m8] = 1.0
                    zrdnd[0, 0, 0][m8] = 0.0

    with computation(BACKWARD), interval(0, -1):
        if cf1 > eps:
            for m9 in range(ngptsw):
                zden1[0, 0, 0][m9] = 1.0 / (
                    1.0 - zrefd[0, 0, 1][m9] * zrdnd[0, 0, 1][m9]
                )
                ztdn[0, 0, 0][m9] = (
                    ztdbt[0, 0, 1][m9] * ztrab[0, 0, 1][m9]
                    + (
                        ztrad[0, 0, 1][m9]
                        * (
                            (ztdn[0, 0, 1][m9] - ztdbt[0, 0, 1][m9])
                            + ztdbt[0, 0, 1][m9]
                            * zrefb[0, 0, 1][m9]
                            * zrdnd[0, 0, 1][m9]
                        )
                    )
                    * zden1[0, 0, 0][m9]
                )
                zrdnd[0, 0, 0][m9] = (
                    zrefd[0, 0, 1][m9]
                    + ztrad[0, 0, 1][m9]
                    * ztrad[0, 0, 1][m9]
                    * zrdnd[0, 0, 1][m9]
                    * zden1[0, 0, 0][m9]
                )

    # Up and down-welling fluxes at levels.
    with computation(FORWARD), interval(...):
        #  -# Process and save outputs.
        # --- ...  surface downward beam/diffused flux components
        if cf1 > eps:
            for m10 in range(ngptsw):
                jb = NGB[0, 0, 0][m10] - 1
                ib = jb + 1 - nblow

                zden1[0, 0, 0][m10] = 1.0 / (
                    1.0 - zrdnd[0, 0, 0][m10] * zrupd[0, 0, 0][m10]
                )
                zfu[0, 0, 0][m10] = (
                    ztdbt[0, 0, 0][m10] * zrupb[0, 0, 0][m10]
                    + (ztdn[0, 0, 0][m10] - ztdbt[0, 0, 0][m10]) * zrupd[0, 0, 0][m10]
                ) * zden1[0, 0, 0][m10]
                zfd[0, 0, 0][m10] = (
                    ztdbt[0, 0, 0][m10]
                    + (
                        ztdn[0, 0, 0][m10]
                        - ztdbt[0, 0, 0][m10]
                        + ztdbt[0, 0, 0][m10]
                        * zrupb[0, 0, 0][m10]
                        * zrdnd[0, 0, 0][m10]
                    )
                    * zden1[0, 0, 0][m10]
                )

                fxupc[0, 0, 0][ib] = (
                    fxupc[0, 0, 0][ib] + zsolar[0, 0, 0][m10] * zfu[0, 0, 0][m10]
                )
                fxdnc[0, 0, 0][ib] = (
                    fxdnc[0, 0, 0][ib] + zsolar[0, 0, 0][m10] * zfd[0, 0, 0][m10]
                )

    # -# Process and save outputs.
    with computation(FORWARD), interval(0, 1):
        if cf1 > eps:
            for m11 in range(ngptsw):
                jb = NGB[0, 0, 0][m11] - 1
                ib = jb + 1 - nblow
                ibd = idxsfc[0, 0, 0][jb - 15] - 1  # spectral band index

                # --- ...  surface downward beam/diffused flux components
                zb11[0, 0][m11] = zsolar[0, 0, 0][m11] * ztdbt0[0, 0, 0][m11]
                zb22[0, 0][m11] = zsolar[0, 0, 0][m11] * (
                    zfd[0, 0, 0][m11] - ztdbt0[0, 0, 0][m11]
                )

                if ibd != -1:
                    sfbmc[0, 0][ibd] = sfbmc[0, 0][ibd] + zb11[0, 0][m11]
                    sfdfc[0, 0][ibd] = sfdfc[0, 0][ibd] + zb22[0, 0][m11]
                else:
                    zf1[0, 0, 0][m11] = 0.5 * zb11[0, 0][m11]
                    zf2[0, 0, 0][m11] = 0.5 * zb22[0, 0][m11]
                    sfbmc[0, 0][0] = sfbmc[0, 0][0] + zf1[0, 0, 0][m11]
                    sfdfc[0, 0][0] = sfdfc[0, 0][0] + zf2[0, 0, 0][m11]
                    sfbmc[0, 0][1] = sfbmc[0, 0][1] + zf1[0, 0, 0][m11]
                    sfdfc[0, 0][1] = sfdfc[0, 0][1] + zf2[0, 0, 0][m11]

    with computation(FORWARD):
        with interval(0, 1):
            for b in range(nbdsw):
                fsfcu0 = fsfcu0 + fxup0[0, 0, 0][b]
                fsfcd0 = fsfcd0 + fxdn0[0, 0, 0][b]

            # --- ...  uv-b surface downward flux
            suvbf0 = fxdn0[0, 0, 0][ibd0]
        with interval(-1, None):
            for bb in range(nbdsw):
                ftoadc = ftoadc + fxdn0[0, 0, 0][bb]
                ftoau0 = ftoau0 + fxup0[0, 0, 0][bb]

    with computation(PARALLEL), interval(...):
        if cf1 <= eps:  # clear column
            for b2 in range(nbdsw):
                fxupc[0, 0, 0][b2] = fxup0[0, 0, 0][b2]
                fxdnc[0, 0, 0][b2] = fxdn0[0, 0, 0][b2]

    with computation(FORWARD):
        with interval(0, 1):
            if cf1 <= eps:
                ftoauc = ftoau0
                fsfcuc = fsfcu0
                fsfcdc = fsfcd0

                # --- ...  surface downward beam/diffused flux components
                sfbmc[0, 0][0] = sfbm0[0, 0][0]
                sfdfc[0, 0][0] = sfdf0[0, 0][0]
                sfbmc[0, 0][1] = sfbm0[0, 0][1]
                sfdfc[0, 0][1] = sfdf0[0, 0][1]

                # --- ...  uv-b surface downward flux
                suvbfc = suvbf0
            else:  # cloudy column, compute total-sky fluxes
                for b3 in range(nbdsw):
                    fsfcuc = fsfcuc + fxupc[0, 0, 0][b3]
                    fsfcdc = fsfcdc + fxdnc[0, 0, 0][b3]

                # --- ...  uv-b surface downward flux
                suvbfc = fxdnc[0, 0, 0][ibd0]

        with interval(-1, None):
            if cf1 > eps:  # cloudy column, compute total-sky fluxes
                for b4 in range(nbdsw):
                    ftoauc = ftoauc + fxupc[0, 0, 0][b4]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nbdsw": nbdsw,
        "lhsw0": lhsw0,
        "lhswb": lhswb,
        "lflxprf": lflxprf,
        "lfdncmp": lfdncmp,
    },
)
def finalloop(
    idxday: FIELD_2DBOOL,
    delp: FIELD_FLT,
    fxupc: Field[type_nbdsw],
    fxdnc: Field[type_nbdsw],
    fxup0: Field[type_nbdsw],
    fxdn0: Field[type_nbdsw],
    suvbf0: FIELD_2D,
    suvbfc: FIELD_2D,
    sfbmc: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    sfdfc: Field[gtscript.IJ, (DTYPE_FLT, (2,))],
    ftoauc: FIELD_2D,
    ftoadc: FIELD_2D,
    ftoau0: FIELD_2D,
    fsfcuc: FIELD_2D,
    fsfcdc: FIELD_2D,
    fsfcu0: FIELD_2D,
    fsfcd0: FIELD_2D,
    upfxc_t: FIELD_2D,
    dnfxc_t: FIELD_2D,
    upfx0_t: FIELD_2D,
    upfxc_s: FIELD_2D,
    dnfxc_s: FIELD_2D,
    upfx0_s: FIELD_2D,
    dnfx0_s: FIELD_2D,
    hswc: FIELD_FLT,
    hsw0: FIELD_FLT,
    uvbf0: FIELD_2D,
    uvbfc: FIELD_2D,
    nirbm: FIELD_2D,
    nirdf: FIELD_2D,
    visbm: FIELD_2D,
    visdf: FIELD_2D,
    rfdelp: FIELD_FLT,
    fnet: FIELD_FLT,
    fnetc: FIELD_FLT,
    fnetb: FIELD_FLT,
    flxuc: FIELD_FLT,
    flxdc: FIELD_FLT,
    flxu0: FIELD_FLT,
    flxd0: FIELD_FLT,
    heatfac: float,
):
    from __externals__ import nbdsw, lhsw0, lhswb, lflxprf, lfdncmp

    with computation(PARALLEL), interval(...):
        if idxday:
            #  --- ...  sum up total spectral fluxes for total-sky
            for ib in range(nbdsw):
                flxuc = flxuc + fxupc[0, 0, 0][ib]
                flxdc = flxdc + fxdnc[0, 0, 0][ib]

                if lhsw0 or lflxprf:
                    flxu0 = flxu0 + fxup0[0, 0, 0][ib]
                    flxd0 = flxd0 + fxdn0[0, 0, 0][ib]

    with computation(FORWARD):
        with interval(0, 1):
            if idxday:
                if lfdncmp:
                    # --- ...  optional uv-b surface downward flux
                    uvbf0 = suvbf0
                    uvbfc = suvbfc

                    # --- ...  optional beam and diffuse sfc fluxes
                    nirbm = sfbmc[0, 0][0]
                    nirdf = sfdfc[0, 0][0]
                    visbm = sfbmc[0, 0][1]
                    visdf = sfdfc[0, 0][1]

                #  --- ...  toa and sfc fluxes

                upfxc_t = ftoauc
                dnfxc_t = ftoadc
                upfx0_t = ftoau0

                upfxc_s = fsfcuc
                dnfxc_s = fsfcdc
                upfx0_s = fsfcu0
                dnfx0_s = fsfcd0

                fnet = flxdc - flxuc
                fnetc = flxd0 - flxu0

                if lhswb:
                    for mb in range(nbdsw):
                        fnetb = fxdnc[0, 0, 0][mb] - fxupc[0, 0, 0][mb]

        with interval(1, None):
            if idxday:
                rfdelp = heatfac / delp

                fnet = flxdc - flxuc
                fnetc = flxd0 - flxu0

                if lhswb:
                    for mb2 in range(nbdsw):
                        fnetb = fxdnc[0, 0, 0][mb2] - fxupc[0, 0, 0][mb2]

    with computation(PARALLEL):
        with interval(1, None):
            if idxday:
                #  --- ...  compute heating rates
                hswc = (fnet - fnet[0, 0, -1]) * rfdelp

                # --- ...  optional clear sky heating rates
                if lhsw0:
                    hsw0 = (fnetc - fnetc[0, 0, -1]) * rfdelp
