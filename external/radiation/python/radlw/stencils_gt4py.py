from gt4py.gtscript import (
    stencil,
    computation,
    interval,
    PARALLEL,
    FORWARD,
    BACKWARD,
    exp,
    log,
    mod,
)
import sys

sys.path.insert(0, "..")
from phys_const import con_amw, con_amd, con_g, con_avgd, con_amo3
from radlw.radlw_param import (
    nbands,
    nplnk,
    nrates,
    eps,
    ngptlw,
    abssnow0,
    absrain,
    cldmin,
    nspa,
    nspb,
    ngb,
    ng01,
    ng02,
    ng03,
    ng04,
    ng05,
    ng06,
    ng07,
    ng08,
    ng09,
    ng10,
    ng11,
    ng12,
    ng13,
    ng14,
    ng15,
    ng16,
    ns02,
    ns03,
    ns04,
    ns05,
    ns06,
    ns07,
    ns08,
    ns09,
    ns10,
    ns11,
    ns12,
    ns13,
    ns14,
    ns15,
    ns16,
    oneminus,
    bpade,
    fluxfac,
    heatfac,
    ntbl,
    wtdiff,
)
from radphysparam import ilwcice, ilwcliq
from config import *

rebuild = False
validate = True
backend = "gtc:gt:cpu_ifirst"

amdw = con_amd / con_amw
amdo3 = con_amd / con_amo3


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nbands": nbands,
        "maxgas": maxgas,
        "ilwcliq": ilwcliq,
        "ilwrgas": ilwrgas,
        "amdw": amdw,
        "amdo3": amdo3,
        "con_avgd": con_avgd,
        "con_g": con_g,
        "con_amd": con_amd,
        "con_amw": con_amw,
        "eps": eps,
    },
)
def firstloop(
    plyr: FIELD_FLT,
    plvl: FIELD_FLT,
    tlyr: FIELD_FLT,
    tlvl: FIELD_FLT,
    qlyr: FIELD_FLT,
    olyr: FIELD_FLT,
    gasvmr: Field[(DTYPE_FLT, (10,))],
    clouds: Field[(DTYPE_FLT, (9,))],
    icseed: FIELD_INT,
    aerosols: Field[(DTYPE_FLT, (nbands, 3))],
    sfemis: FIELD_2D,
    sfgtmp: FIELD_2D,
    dzlyr: FIELD_FLT,
    delpin: FIELD_FLT,
    de_lgth: FIELD_2D,
    cldfrc: FIELD_FLT,
    pavel: FIELD_FLT,
    tavel: FIELD_FLT,
    delp: FIELD_FLT,
    dz: FIELD_FLT,
    h2ovmr: FIELD_FLT,
    o3vmr: FIELD_FLT,
    coldry: FIELD_FLT,
    colbrd: FIELD_FLT,
    colamt: Field[type_maxgas],
    wx: Field[type_maxxsec],
    tauaer: Field[type_nbands],
    semiss0: Field[gtscript.IJ, type_nbands],
    semiss: Field[gtscript.IJ, type_nbands],
    tem11: FIELD_FLT,
    tem22: FIELD_FLT,
    tem00: FIELD_2D,
    summol: FIELD_FLT,
    pwvcm: FIELD_2D,
    clwp: FIELD_FLT,
    relw: FIELD_FLT,
    ciwp: FIELD_FLT,
    reiw: FIELD_FLT,
    cda1: FIELD_FLT,
    cda2: FIELD_FLT,
    cda3: FIELD_FLT,
    cda4: FIELD_FLT,
    secdiff: Field[gtscript.IJ, type_nbands],
    a0: Field[type_nbands],
    a1: Field[type_nbands],
    a2: Field[type_nbands],
):
    from __externals__ import (
        nbands,
        ilwcliq,
        ilwrgas,
        maxgas,
        amdw,
        amdo3,
        con_avgd,
        con_amd,
        con_amw,
        con_g,
        eps,
    )

    with computation(FORWARD):
        with interval(0, 1):
            for j0 in range(nbands):
                semiss0[0, 0][j0] = 1.0

            if sfemis[0, 0] > eps and sfemis[0, 0] <= 1.0:
                for j in range(nbands):
                    semiss[0, 0][j] = sfemis[0, 0]
            else:
                for j2 in range(nbands):
                    semiss[0, 0][j2] = semiss0[0, 0][j2]

    with computation(PARALLEL):
        with interval(1, None):
            pavel = plyr
            delp = delpin
            tavel = tlyr
            dz = dzlyr

            tem1 = 100.0 * con_g
            tem2 = 1.0e-20 * 1.0e3 * con_avgd

            h2ovmr = max(0.0, qlyr * amdw / (1.0 - qlyr))  # input specific humidity
            o3vmr = max(0.0, olyr * amdo3)  # input mass mixing ratio

            tem0 = (1.0 - h2ovmr) * con_amd + h2ovmr * con_amw
            coldry = tem2 * delp / (tem1 * tem0 * (1.0 + h2ovmr))
            temcol = 1.0e-12 * coldry

            colamt[0, 0, 0][0] = max(0.0, coldry * h2ovmr)  # h2o
            colamt[0, 0, 0][1] = max(temcol, coldry * gasvmr[0, 0, 0][0])  # co2
            colamt[0, 0, 0][2] = max(temcol, coldry * o3vmr)  # o3

            if ilwrgas > 0:
                colamt[0, 0, 0][3] = max(temcol, coldry * gasvmr[0, 0, 0][1])  # n2o
                colamt[0, 0, 0][4] = max(temcol, coldry * gasvmr[0, 0, 0][2])  # ch4
                colamt[0, 0, 0][5] = max(0.0, coldry * gasvmr[0, 0, 0][3])  # o2
                colamt[0, 0, 0][6] = max(0.0, coldry * gasvmr[0, 0, 0][4])  # co

                wx[0, 0, 0][0] = max(0.0, coldry * gasvmr[0, 0, 0][8])  # ccl4
                wx[0, 0, 0][1] = max(0.0, coldry * gasvmr[0, 0, 0][5])  # cf11
                wx[0, 0, 0][2] = max(0.0, coldry * gasvmr[0, 0, 0][6])  # cf12
                wx[0, 0, 0][3] = max(0.0, coldry * gasvmr[0, 0, 0][7])  # cf22

            else:
                colamt[0, 0, 0][3] = 0.0  # n2o
                colamt[0, 0, 0][4] = 0.0  # ch4
                colamt[0, 0, 0][5] = 0.0  # o2
                colamt[0, 0, 0][6] = 0.0  # co

                wx[0, 0, 0][0] = 0.0
                wx[0, 0, 0][1] = 0.0
                wx[0, 0, 0][2] = 0.0
                wx[0, 0, 0][3] = 0.0

            for j3 in range(nbands):
                tauaer[0, 0, 0][j3] = aerosols[0, 0, 0][j3, 0] * (
                    1.0 - aerosols[0, 0, 0][j3, 1]
                )

    with computation(PARALLEL):
        with interval(1, None):
            cldfrc = clouds[0, 0, 0][0]

    with computation(PARALLEL):
        with interval(1, None):
            # Workaround for variables first referenced inside if statements
            # Can be removed at next gt4py release
            clwp = clwp
            relw = relw
            ciwp = ciwp
            reiw = reiw
            cda1 = cda1
            cda2 = cda2
            cda3 = cda3
            cda4 = cda4
            clouds = clouds
            if ilwcliq > 0:
                clwp = clouds[0, 0, 0][1]
                relw = clouds[0, 0, 0][2]
                ciwp = clouds[0, 0, 0][3]
                reiw = clouds[0, 0, 0][4]
                cda1 = clouds[0, 0, 0][5]
                cda2 = clouds[0, 0, 0][6]
                cda3 = clouds[0, 0, 0][7]
                cda4 = clouds[0, 0, 0][8]
            else:
                cda1 = clouds[0, 0, 0][1]

    with computation(FORWARD):
        with interval(0, 1):
            cldfrc = 1.0
        with interval(1, 2):
            tem11 = coldry[0, 0, 0] + colamt[0, 0, 0][0]
            tem22 = colamt[0, 0, 0][0]

    with computation(FORWARD):
        with interval(2, None):
            #  --- ...  compute precipitable water vapor for diffusivity angle adjustments
            tem11 = tem11[0, 0, -1] + coldry + colamt[0, 0, 0][0]
            tem22 = tem22[0, 0, -1] + colamt[0, 0, 0][0]

    with computation(FORWARD):
        with interval(-1, None):
            tem00 = 10.0 * tem22 / (amdw * tem11 * con_g)
    with computation(FORWARD):
        with interval(0, 1):
            pwvcm[0, 0] = tem00[0, 0] * plvl[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            tem1 = 1.80
            tem2 = 1.50
            for j4 in range(nbands):
                if j4 == 0 or j4 == 3 or j4 == 9:
                    secdiff[0, 0][j4] = 1.66
                else:
                    secdiff[0, 0][j4] = min(
                        tem1,
                        max(
                            tem2,
                            a0[0, 0, 0][j4]
                            + a1[0, 0, 0][j4] * exp(a2[0, 0, 0][j4] * pwvcm),
                        ),
                    )
        with interval(1, None):
            for m in range(1, maxgas):
                summol += colamt[0, 0, 0][m]
            colbrd = coldry - summol


@gtscript.stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nbands": nbands,
        "ilwcliq": ilwcliq,
        "ngptlw": ngptlw,
        "isubclw": isubclw,
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
    dz: FIELD_FLT,
    cldfmc: Field[type_ngptlw],
    taucld: Field[type_nbands],
    cldtau: FIELD_FLT,
    absliq1: Field[(DTYPE_FLT, (58, nbands))],
    absice1: Field[(DTYPE_FLT, (2, 5))],
    absice2: Field[(DTYPE_FLT, (43, nbands))],
    absice3: Field[(DTYPE_FLT, (46, nbands))],
    ipat: Field[(DTYPE_INT, (nbands,))],
    tauliq: Field[type_nbands],
    tauice: Field[type_nbands],
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
    index: FIELD_INT,
    ia: FIELD_INT,
    lcloudy: Field[(DTYPE_INT, (ngptlw,))],
    cdfunc: Field[type_ngptlw],
    tem1: FIELD_FLT,
    lcf1: FIELD_2DBOOL,
    cldsum: FIELD_FLT,
):
    from __externals__ import nbands, ilwcliq, ngptlw, isubclw

    # Compute flag for whether or not there is cloud in the vertical column
    with computation(FORWARD):
        with interval(0, 1):
            cldsum = cfrac[0, 0, 1]
        with interval(1, -1):
            cldsum = cldsum[0, 0, -1] + cfrac[0, 0, 1]
    with computation(FORWARD), interval(-2, -1):
        lcf1 = cldsum > 0

    with computation(FORWARD), interval(1, None):
        if lcf1:
            if ilwcliq > 0:
                if cfrac > cldmin:
                    tauran = absrain * cdat1
                    if cdat3 > 0.0 and cdat4 > 10.0:
                        tausnw = abssnow0 * 1.05756 * cdat3 / cdat4
                    else:
                        tausnw = 0.0

                    cldliq = cliqp
                    cldice = cicep
                    refliq = reliq
                    refice = reice

                    if cldliq <= 0:
                        for i in range(nbands):
                            tauliq[0, 0, 0][i] = 0.0
                    else:
                        if ilwcliq == 1:
                            factor = refliq - 1.5
                            index = max(1, min(57, factor)) - 1
                            fint = factor - (index + 1)

                            for ib in range(nbands):
                                tmp = cldliq * (
                                    absliq1[0, 0, 0][index, ib]
                                    + fint
                                    * (
                                        absliq1[0, 0, 0][index + 1, ib]
                                        - absliq1[0, 0, 0][index, ib]
                                    )
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauliq[0, 0, 0][ib] = tmp if tmp > 0.0 else 0.0

                    if cldice <= 0.0:
                        for ib2 in range(nbands):
                            tauice[0, 0, 0][ib2] = 0.0
                    else:
                        if ilwcice == 1:
                            refice = min(130.0, max(13.0, refice))

                            for ib3 in range(nbands):
                                ia = ipat[0, 0, 0][ib3] - 1
                                tmp = cldice * (
                                    absice1[0, 0, 0][0, ia]
                                    + absice1[0, 0, 0][1, ia] / refice
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauice[0, 0, 0][ib3] = tmp if tmp > 0.0 else 0.0
                        elif ilwcice == 2:
                            factor = (refice - 2.0) / 3.0
                            index = max(1, min(42, factor)) - 1
                            fint = factor - (index + 1)

                            for ib4 in range(nbands):
                                tmp = cldice * (
                                    absice2[0, 0, 0][index, ib4]
                                    + fint
                                    * (
                                        absice2[0, 0, 0][index + 1, ib4]
                                        - absice2[0, 0, 0][index, ib4]
                                    )
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauice[0, 0, 0][ib4] = tmp if tmp > 0.0 else 0.0

                        elif ilwcice == 3:
                            dgeice = max(5.0, 1.0315 * refice)  # v4.71 value
                            factor = (dgeice - 2.0) / 3.0
                            index = max(1, min(45, factor)) - 1
                            fint = factor - (index + 1)

                            for ib5 in range(nbands):
                                tmp = cldice * (
                                    absice3[0, 0, 0][index, ib5]
                                    + fint
                                    * (
                                        absice3[0, 0, 0][index + 1, ib5]
                                        - absice3[0, 0, 0][index, ib5]
                                    )
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauice[0, 0, 0][ib5] = tmp if tmp > 0.0 else 0.0

                    for ib6 in range(nbands):
                        taucld[0, 0, 0][ib6] = (
                            tauice[0, 0, 0][ib6]
                            + tauliq[0, 0, 0][ib6]
                            + tauran
                            + tausnw
                        )

            else:
                if cfrac > cldmin:
                    for ib7 in range(nbands):
                        taucld[0, 0, 0][ib7] = cdat1

            if isubclw > 0:
                if cfrac < cldmin:
                    cldf = 0.0
                else:
                    cldf = cfrac

    # This section builds mcica_subcol from the fortran into cldprop.
    # Here I've read in the generated random numbers until we figure out
    # what to do with them. This will definitely need to change in future.
    # Only the iovrlw = 1 option is ported from Fortran
    with computation(PARALLEL), interval(2, None):
        if lcf1:
            tem1 = 1.0 - cldf[0, 0, -1]

            for n in range(ngptlw):
                if cdfunc[0, 0, -1][n] > tem1:
                    cdfunc[0, 0, 0][n] = cdfunc[0, 0, -1][n]
                else:
                    cdfunc[0, 0, 0][n] = cdfunc[0, 0, 0][n] * tem1

    with computation(PARALLEL), interval(1, None):
        if lcf1:
            tem1 = 1.0 - cldf[0, 0, 0]

            for n2 in range(ngptlw):
                if cdfunc[0, 0, 0][n2] >= tem1:
                    lcloudy[0, 0, 0][n2] = 1
                else:
                    lcloudy[0, 0, 0][n2] = 0

            for n3 in range(ngptlw):
                if lcloudy[0, 0, 0][n3] == 1:
                    cldfmc[0, 0, 0][n3] = 1.0
                else:
                    cldfmc[0, 0, 0][n3] = 0.0

            cldtau = taucld[0, 0, 0][6]


stpfac = 296.0 / 1013.0


@stencil(
    backend=backend, rebuild=rebuild, externals={"nbands": nbands, "stpfac": stpfac}
)
def setcoef(
    pavel: FIELD_FLT,
    tavel: FIELD_FLT,
    tz: FIELD_FLT,
    stemp: FIELD_2D,
    h2ovmr: FIELD_FLT,
    colamt: Field[type_maxgas],
    coldry: FIELD_FLT,
    colbrd: FIELD_FLT,
    totplnk: Field[(DTYPE_FLT, (nplnk, nbands))],
    pref: Field[(DTYPE_FLT, (59,))],
    preflog: Field[(DTYPE_FLT, (59,))],
    tref: Field[(DTYPE_FLT, (59,))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    delwave: Field[type_nbands],
    laytrop: Field[bool],
    pklay: Field[type_nbands],
    pklev: Field[type_nbands],
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
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
    minorfrac: FIELD_FLT,
    scaleminor: FIELD_FLT,
    scaleminorn2: FIELD_FLT,
    indminor: FIELD_INT,
    tzint: FIELD_INT,
    stempint: FIELD_INT,
    tavelint: FIELD_INT,
    indlay: FIELD_INT,
    indlev: FIELD_INT,
    tlyrfr: FIELD_FLT,
    tlvlfr: FIELD_FLT,
    jp1: FIELD_INT,
    plog: FIELD_FLT,
):
    from __externals__ import nbands, stpfac

    with computation(PARALLEL):
        #  --- ...  calculate information needed by the radiative transfer routine
        #           that is specific to this atmosphere, especially some of the
        #           coefficients and indices needed to compute the optical depths
        #           by interpolating data from stored reference atmospheres.
        with interval(0, 1):
            indlay = min(180, max(1, stemp - 159.0))
            indlev = min(180, max(1, tz - 159.0))
            tzint = tz
            stempint = stemp
            tlyrfr = stemp - stempint
            tlvlfr = tz - tzint

            for i0 in range(nbands):
                tem1 = totplnk[0, 0, 0][indlay, i0] - totplnk[0, 0, 0][indlay - 1, i0]
                tem2 = totplnk[0, 0, 0][indlev, i0] - totplnk[0, 0, 0][indlev - 1, i0]
                pklay[0, 0, 0][i0] = delwave[0, 0, 0][i0] * (
                    totplnk[0, 0, 0][indlay - 1, i0] + tlyrfr * tem1
                )
                pklev[0, 0, 0][i0] = delwave[0, 0, 0][i0] * (
                    totplnk[0, 0, 0][indlev - 1, i0] + tlvlfr * tem2
                )

        #           calculate the integrated Planck functions for each band at the
        #           surface, level, and layer temperatures.
        with interval(1, None):
            indlay = min(180, max(1, tavel - 159.0))
            tavelint = tavel
            tlyrfr = tavel - tavelint

            indlev = min(180, max(1, tz - 159.0))
            tzint = tz
            tlvlfr = tz - tzint

            #  --- ...  begin spectral band loop
            for i in range(nbands):
                pklay[0, 0, 0][i] = delwave[0, 0, 0][i] * (
                    totplnk[0, 0, 0][indlay - 1, i]
                    + tlyrfr
                    * (totplnk[0, 0, 0][indlay, i] - totplnk[0, 0, 0][indlay - 1, i])
                )
                pklev[0, 0, 0][i] = delwave[0, 0, 0][i] * (
                    totplnk[0, 0, 0][indlev - 1, i]
                    + tlvlfr
                    * (totplnk[0, 0, 0][indlev, i] - totplnk[0, 0, 0][indlev - 1, i])
                )

            #  --- ...  find the two reference pressures on either side of the
            #           layer pressure. store them in jp and jp1. store in fp the
            #           fraction of the difference (in ln(pressure)) between these
            #           two values that the layer pressure lies.

            plog = log(pavel)
            jp = max(1, min(58, 36.0 - 5.0 * (plog + 0.04))) - 1
            jp1 = jp + 1
            #  --- ...  limit pressure extrapolation at the top
            fp = max(0.0, min(1.0, 5.0 * (preflog[0, 0, 0][jp] - plog)))

            #  --- ...  determine, for each reference pressure (jp and jp1), which
            #           reference temperature (these are different for each
            #           reference pressure) is nearest the layer temperature but does
            #           not exceed it. store these indices in jt and jt1, resp.
            #           store in ft (resp. ft1) the fraction of the way between jt
            #           (jt1) and the next highest reference temperature that the
            #           layer temperature falls.

            tem1 = (tavel - tref[0, 0, 0][jp]) / 15.0
            tem2 = (tavel - tref[0, 0, 0][jp1]) / 15.0
            jt = max(1, min(4, 3.0 + tem1)) - 1
            jt1 = max(1, min(4, 3.0 + tem2)) - 1
            # --- ...  restrict extrapolation ranges by limiting abs(det t) < 37.5 deg
            ft = max(-0.5, min(1.5, tem1 - (jt - 2)))
            ft1 = max(-0.5, min(1.5, tem2 - (jt1 - 2)))

            #  --- ...  we have now isolated the layer ln pressure and temperature,
            #           between two reference pressures and two reference temperatures
            #           (for each reference pressure).  we multiply the pressure
            #           fraction fp with the appropriate temperature fractions to get
            #           the factors that will be needed for the interpolation that yields
            #           the optical depths (performed in routines taugbn for band n)

            tem1 = 1.0 - fp
            fac10 = tem1 * ft
            fac00 = tem1 * (1.0 - ft)
            fac11 = fp * ft1
            fac01 = fp * (1.0 - ft1)

            forfac = pavel * stpfac / (tavel * (1.0 + h2ovmr))
            selffac = h2ovmr * forfac

            #  --- ...  set up factors needed to separately include the minor gases
            #           in the calculation of absorption coefficient

            scaleminor = pavel / tavel
            scaleminorn2 = (pavel / tavel) * (colbrd / (coldry + colamt[0, 0, 0][0]))

            tem1 = (tavel - 180.8) / 7.2
            indminor = min(18, max(1, tem1))
            minorfrac = tem1 - indminor

            #  --- ...  if the pressure is less than ~100mb, perform a different
            #           set of species interpolations.

            indfor = indfor
            forfrac = forfrac
            indself = indself
            selffrac = selffrac
            rfrate = rfrate
            chi_mls = chi_mls
            laytrop = laytrop

            if plog > 4.56:

                # compute troposphere mask, True in troposphere, False otherwise
                laytrop = True

                tem1 = (332.0 - tavel) / 36.0
                indfor = min(2, max(1, tem1))
                forfrac = tem1 - indfor

                #  --- ...  set up factors needed to separately include the water vapor
                #           self-continuum in the calculation of absorption coefficient.

                tem1 = (tavel - 188.0) / 7.2
                indself = min(9, max(1, tem1 - 7))
                selffrac = tem1 - (indself + 7)

                #  --- ...  setup reference ratio to be used in calculation of binary
                #           species parameter in lower atmosphere.

                rfrate[0, 0, 0][0, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][0, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )
                rfrate[0, 0, 0][1, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][2, jp]
                )
                rfrate[0, 0, 0][1, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][2, jp + 1]
                )
                rfrate[0, 0, 0][2, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][3, jp]
                )
                rfrate[0, 0, 0][2, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][3, jp + 1]
                )
                rfrate[0, 0, 0][3, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][5, jp]
                )
                rfrate[0, 0, 0][3, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][5, jp + 1]
                )
                rfrate[0, 0, 0][4, 0] = (
                    chi_mls[0, 0, 0][3, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][4, 1] = (
                    chi_mls[0, 0, 0][3, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )

            else:
                laytrop = False

                tem1 = (tavel - 188.0) / 36.0
                indfor = 3
                forfrac = tem1 - 1.0

                indself = 0
                selffrac = 0.0

                #  --- ...  setup reference ratio to be used in calculation of binary
                #           species parameter in upper atmosphere.

                rfrate[0, 0, 0][0, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][0, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )
                rfrate[0, 0, 0][5, 0] = (
                    chi_mls[0, 0, 0][2, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][5, 1] = (
                    chi_mls[0, 0, 0][2, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )

            #  --- ...  rescale selffac and forfac for use in taumol

            selffac = colamt[0, 0, 0][0] * selffac
            forfac = colamt[0, 0, 0][0] * forfac

            #  --- ...  add one to computed indices for compatibility with later
            #           subroutines

            jp += 1
            jt += 1
            jt1 += 1


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[0],
        "nspb": nspb[0],
        "ng01": ng01,
    },
)
def taugb01(
    laytrop: FIELD_BOOL,
    pavel: FIELD_FLT,
    colamt: Field[type_maxgas],
    colbrd: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    scaleminorn2: FIELD_FLT,
    indminor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (10, 65))],
    absb: Field[(DTYPE_FLT, (10, 235))],
    selfref: Field[(DTYPE_FLT, (10, 10))],
    forref: Field[(DTYPE_FLT, (10, 4))],
    fracrefa: Field[(DTYPE_FLT, (10,))],
    fracrefb: Field[(DTYPE_FLT, (10,))],
    ka_mn2: Field[(DTYPE_FLT, (10, 19))],
    kb_mn2: Field[(DTYPE_FLT, (10, 19))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    pp: FIELD_FLT,
    corradj: FIELD_FLT,
    scalen2: FIELD_FLT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    taun2: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng01

    with computation(PARALLEL), interval(1, None):
        # Workaround for bug in gt4py
        jp = jp
        jt = jt
        jt1 = jt1
        indself = indself
        indfor = indfor
        indminor = indminor
        pavel = pavel
        colbrd = colbrd
        scaleminorn2 = scaleminorn2

        if laytrop:
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa
            ind1 = (jp * 5 + (jt1 - 1)) * nspa
            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1

            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1

            pp = pavel
            scalen2 = colbrd * scaleminorn2
            if pp < 250.0:
                corradj = 1.0 - 0.15 * (250.0 - pp) / 154.4
            else:
                corradj = 1.0

            for ig in range(ng01):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                taun2 = scalen2 * (
                    ka_mn2[0, 0, 0][ig, indm]
                    + minorfrac
                    * (ka_mn2[0, 0, 0][ig, indmp] - ka_mn2[0, 0, 0][ig, indm])
                )

                taug[0, 0, 0][ig] = corradj * (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ig, ind0]
                        + fac10 * absa[0, 0, 0][ig, ind0p]
                        + fac01 * absa[0, 0, 0][ig, ind1]
                        + fac11 * absa[0, 0, 0][ig, ind1p]
                    )
                    + tauself
                    + taufor
                    + taun2
                )

                fracs[0, 0, 0][ig] = fracrefa[0, 0, 0][ig]

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb
            indf = indfor - 1
            indm = indminor - 1

            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indfp = indf + 1
            indmp = indm + 1

            scalen2 = colbrd * scaleminorn2
            corradj = 1.0 - 0.15 * (pavel / 95.6)

            for ig2 in range(ng01):
                taufor = forfac * (
                    forref[0, 0, 0][ig2, indf]
                    + forfrac
                    * (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])
                )
                taun2 = scalen2 * (
                    kb_mn2[0, 0, 0][ig2, indm]
                    + minorfrac
                    * (kb_mn2[0, 0, 0][ig2, indmp] - kb_mn2[0, 0, 0][ig2, indm])
                )

                taug[0, 0, 0][ig2] = corradj * (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                    + taufor
                    + taun2
                )

                fracs[0, 0, 0][ig2] = fracrefb[0, 0, 0][ig2]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[1],
        "nspb": nspb[1],
        "ng02": ng02,
        "ns02": ns02,
    },
)
def taugb02(
    laytrop: FIELD_BOOL,
    pavel: FIELD_FLT,
    colamt: Field[type_maxgas],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (10, 65))],
    absb: Field[(DTYPE_FLT, (10, 235))],
    selfref: Field[(DTYPE_FLT, (10, 10))],
    forref: Field[(DTYPE_FLT, (10, 4))],
    fracrefa: Field[(DTYPE_FLT, (10,))],
    fracrefb: Field[(DTYPE_FLT, (10,))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    corradj: FIELD_FLT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng02, ns02

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa
            ind1 = (jp * 5 + (jt1 - 1)) * nspa
            inds = indself - 1
            indf = indfor - 1

            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indsp = inds + 1
            indfp = indf + 1

            corradj = 1.0 - 0.05 * (pavel - 100.0) / 900.0

            for ig in range(ng02):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )

                taug[0, 0, 0][ns02 + ig] = corradj * (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ig, ind0]
                        + fac10 * absa[0, 0, 0][ig, ind0p]
                        + fac01 * absa[0, 0, 0][ig, ind1]
                        + fac11 * absa[0, 0, 0][ig, ind1p]
                    )
                    + tauself
                    + taufor
                )

                fracs[0, 0, 0][ns02 + ig] = fracrefa[0, 0, 0][ig]

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb
            indf = indfor - 1

            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indfp = indf + 1

            for ig2 in range(ng02):
                taufor = forfac * (
                    forref[0, 0, 0][ig2, indf]
                    + forfrac
                    * (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])
                )

                taug[0, 0, 0][ns02 + ig2] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                    + taufor
                )

                fracs[0, 0, 0][ns02 + ig2] = fracrefb[0, 0, 0][ig2]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[2],
        "nspb": nspb[2],
        "ng03": ng03,
        "ns03": ns03,
        "oneminus": oneminus,
    },
)
def taugb03(
    laytrop: FIELD_BOOL,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    indminor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng03, 585))],
    absb: Field[(DTYPE_FLT, (ng03, 1175))],
    selfref: Field[(DTYPE_FLT, (ng03, 10))],
    forref: Field[(DTYPE_FLT, (ng03, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng03, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng03, 5))],
    ka_mn2o: Field[(DTYPE_FLT, (ng03, 9, 19))],
    kb_mn2o: Field[(DTYPE_FLT, (ng03, 5, 19))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    js: FIELD_INT,
    js1: FIELD_INT,
    jmn2o: FIELD_INT,
    jmn2op: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
    ratn2o: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng03, ns03, oneminus

    with computation(PARALLEL):
        with interval(...):
            #  --- ...  minor gas mapping levels:
            #     lower - n2o, p = 706.272 mbar, t = 278.94 k
            #     upper - n2o, p = 95.58 mbar, t = 215.7 k

            refrat_planck_a = chi_mls[0, 0, 0][0, 8] / chi_mls[0, 0, 0][1, 8]
            refrat_planck_b = chi_mls[0, 0, 0][0, 12] / chi_mls[0, 0, 0][1, 12]
            refrat_m_a = chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][1, 2]
            refrat_m_b = chi_mls[0, 0, 0][0, 12] / chi_mls[0, 0, 0][1, 12]

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            speccomb_mn2o = colamt[0, 0, 0][0] + refrat_m_a * colamt[0, 0, 0][1]
            specparm_mn2o = colamt[0, 0, 0][0] / speccomb_mn2o
            specmult_mn2o = 8.0 * min(specparm_mn2o, oneminus)
            jmn2o = 1 + specmult_mn2o - 1
            fmn2o = mod(specmult_mn2o, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1
            jmn2op = jmn2o + 1
            jplp = jpl + 1

            #  --- ...  in atmospheres where the amount of n2O is too great to be considered
            #           a minor species, adjust the column amount of n2O by an empirical factor
            #           to obtain the proper contribution.

            p = coldry * chi_mls[0, 0, 0][3, jp]
            ratn2o = colamt[0, 0, 0][3] / p
            if ratn2o > 1.5:
                adjfac = 0.5 + (ratn2o - 0.5) ** 0.65
                adjcoln2o = adjfac * p
            else:
                adjcoln2o = colamt[0, 0, 0][3]

            if specparm < 0.125:
                p = fs - 1.0
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p = -fs
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0 * 1
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk0 = 1.0 - fs
                fk1 = fs
                fk2 = 0.0
                id000 = ind0 * 1
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 * 1
                id210 = ind0 * 1

            fac000 = fk0 * fac00
            fac100 = fk1 * fac00
            fac200 = fk2 * fac00
            fac010 = fk0 * fac10
            fac110 = fk1 * fac10
            fac210 = fk2 * fac10

            if specparm1 < 0.125:
                p = fs1 - 1.0
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p = -fs1
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk0 = 1.0 - fs1
                fk1 = fs1
                fk2 = 0.0
                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk0 * fac01
            fac101 = fk1 * fac01
            fac201 = fk2 * fac01
            fac011 = fk0 * fac11
            fac111 = fk1 * fac11
            fac211 = fk2 * fac11

            for ig in range(ng03):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                n2om1 = ka_mn2o[0, 0, 0][ig, jmn2o, indm] + fmn2o * (
                    ka_mn2o[0, 0, 0][ig, jmn2op, indm]
                    - ka_mn2o[0, 0, 0][ig, jmn2o, indm]
                )
                n2om2 = ka_mn2o[0, 0, 0][ig, jmn2o, indmp] + fmn2o * (
                    ka_mn2o[0, 0, 0][ig, jmn2op, indmp]
                    - ka_mn2o[0, 0, 0][ig, jmn2o, indmp]
                )
                absn2o = n2om1 + minorfrac * (n2om2 - n2om1)

                tau_major = speccomb * (
                    fac000 * absa[0, 0, 0][ig, id000]
                    + fac010 * absa[0, 0, 0][ig, id010]
                    + fac100 * absa[0, 0, 0][ig, id100]
                    + fac110 * absa[0, 0, 0][ig, id110]
                    + fac200 * absa[0, 0, 0][ig, id200]
                    + fac210 * absa[0, 0, 0][ig, id210]
                )

                tau_major1 = speccomb1 * (
                    fac001 * absa[0, 0, 0][ig, id001]
                    + fac011 * absa[0, 0, 0][ig, id011]
                    + fac101 * absa[0, 0, 0][ig, id101]
                    + fac111 * absa[0, 0, 0][ig, id111]
                    + fac201 * absa[0, 0, 0][ig, id201]
                    + fac211 * absa[0, 0, 0][ig, id211]
                )

                taug[0, 0, 0][ns03 + ig] = (
                    tau_major + tau_major1 + tauself + taufor + adjcoln2o * absn2o
                )

                fracs[0, 0, 0][ns03 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:

            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 4.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 4.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb + js1 - 1

            speccomb_mn2o = colamt[0, 0, 0][0] + refrat_m_b * colamt[0, 0, 0][1]
            specparm_mn2o = colamt[0, 0, 0][0] / speccomb_mn2o
            specmult_mn2o = 4.0 * min(specparm_mn2o, oneminus)
            jmn2o = 1 + specmult_mn2o - 1
            fmn2o = mod(specmult_mn2o, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_b * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 4.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            indf = indfor - 1
            indm = indminor - 1
            indfp = indf + 1
            indmp = indm + 1
            jmn2op = jmn2o + 1
            jplp = jpl + 1

            id000 = ind0
            id010 = ind0 + 5
            id100 = ind0 + 1
            id110 = ind0 + 6
            id001 = ind1
            id011 = ind1 + 5
            id101 = ind1 + 1
            id111 = ind1 + 6

            #  --- ...  in atmospheres where the amount of n2o is too great to be considered
            #           a minor species, adjust the column amount of N2O by an empirical factor
            #           to obtain the proper contribution.

            p = coldry * chi_mls[0, 0, 0][3, jp]
            ratn2o = colamt[0, 0, 0][3] / p
            if ratn2o > 1.5:
                adjfac = 0.5 + (ratn2o - 0.5) ** 0.65
                adjcoln2o = adjfac * p
            else:
                adjcoln2o = colamt[0, 0, 0][3]

            fk0 = 1.0 - fs
            fk1 = fs
            fac000 = fk0 * fac00
            fac010 = fk0 * fac10
            fac100 = fk1 * fac00
            fac110 = fk1 * fac10

            fk0 = 1.0 - fs1
            fk1 = fs1
            fac001 = fk0 * fac01
            fac011 = fk0 * fac11
            fac101 = fk1 * fac01
            fac111 = fk1 * fac11

            for ig2 in range(ng03):
                taufor = forfac * (
                    forref[0, 0, 0][ig2, indf]
                    + forfrac
                    * (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])
                )
                n2om1 = kb_mn2o[0, 0, 0][ig2, jmn2o, indm] + fmn2o * (
                    kb_mn2o[0, 0, 0][ig2, jmn2op, indm]
                    - kb_mn2o[0, 0, 0][ig2, jmn2o, indm]
                )
                n2om2 = kb_mn2o[0, 0, 0][ig2, jmn2o, indmp] + fmn2o * (
                    kb_mn2o[0, 0, 0][ig2, jmn2op, indmp]
                    - kb_mn2o[0, 0, 0][ig2, jmn2o, indmp]
                )
                absn2o = n2om1 + minorfrac * (n2om2 - n2om1)

                tau_major = speccomb * (
                    fac000 * absb[0, 0, 0][ig2, id000]
                    + fac010 * absb[0, 0, 0][ig2, id010]
                    + fac100 * absb[0, 0, 0][ig2, id100]
                    + fac110 * absb[0, 0, 0][ig2, id110]
                )

                tau_major1 = speccomb1 * (
                    fac001 * absb[0, 0, 0][ig2, id001]
                    + fac011 * absb[0, 0, 0][ig2, id011]
                    + fac101 * absb[0, 0, 0][ig2, id101]
                    + fac111 * absb[0, 0, 0][ig2, id111]
                )

                taug[0, 0, 0][ns03 + ig2] = (
                    tau_major + tau_major1 + taufor + adjcoln2o * absn2o
                )

                fracs[0, 0, 0][ns03 + ig2] = fracrefb[0, 0, 0][ig2, jpl] + fpl * (
                    fracrefb[0, 0, 0][ig2, jplp] - fracrefb[0, 0, 0][ig2, jpl]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[3],
        "nspb": nspb[3],
        "ng04": ng04,
        "ns04": ns04,
        "oneminus": oneminus,
    },
)
def taugb04(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng04, 585))],
    absb: Field[(DTYPE_FLT, (ng04, 1175))],
    selfref: Field[(DTYPE_FLT, (ng04, 10))],
    forref: Field[(DTYPE_FLT, (ng04, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng04, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng04, 5))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    js: FIELD_INT,
    js1: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng04, ns04, oneminus

    with computation(PARALLEL):
        with interval(...):
            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 10] / chi_mls[0, 0, 0][1, 10]
            )  # P = 142.5940 mb
            refrat_planck_b = (
                chi_mls[0, 0, 0][2, 12] / chi_mls[0, 0, 0][1, 12]
            )  # P = 95.58350 mb

    with computation(PARALLEL), interval(1, None):
        refrat_planck_a = refrat_planck_a
        refrat_planck_b = refrat_planck_b
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1
            jplp = jpl + 1

            # Workaround for bug in gt4py, can be removed at release of next tag (>32)
            id000 = id000
            id010 = id010
            id100 = id100
            id110 = id110
            id200 = id200
            id210 = id210

            if specparm < 0.125:
                p = fs - 1.0
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p = -fs
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk0 = 1.0 - fs
                fk1 = fs
                fk2 = 0.0
                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0
                id210 = ind0

            fac000 = fk0 * fac00
            fac100 = fk1 * fac00
            fac200 = fk2 * fac00
            fac010 = fk0 * fac10
            fac110 = fk1 * fac10
            fac210 = fk2 * fac10

            id001 = id001
            id011 = id011
            id101 = id101
            id111 = id111
            id201 = id201
            id211 = id211

            if specparm1 < 0.125:
                p = fs1 - 1.0
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p = -fs1
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk0 = 1.0 - fs1
                fk1 = fs1
                fk2 = 0.0
                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk0 * fac01
            fac101 = fk1 * fac01
            fac201 = fk2 * fac01
            fac011 = fk0 * fac11
            fac111 = fk1 * fac11
            fac211 = fk2 * fac11

            for ig in range(ng04):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )

                tau_major = speccomb * (
                    fac000 * absa[0, 0, 0][ig, id000]
                    + fac010 * absa[0, 0, 0][ig, id010]
                    + fac100 * absa[0, 0, 0][ig, id100]
                    + fac110 * absa[0, 0, 0][ig, id110]
                    + fac200 * absa[0, 0, 0][ig, id200]
                    + fac210 * absa[0, 0, 0][ig, id210]
                )

                tau_major1 = speccomb1 * (
                    fac001 * absa[0, 0, 0][ig, id001]
                    + fac011 * absa[0, 0, 0][ig, id011]
                    + fac101 * absa[0, 0, 0][ig, id101]
                    + fac111 * absa[0, 0, 0][ig, id111]
                    + fac201 * absa[0, 0, 0][ig, id201]
                    + fac211 * absa[0, 0, 0][ig, id211]
                )

                taug[0, 0, 0][ns04 + ig] = tau_major + tau_major1 + tauself + taufor

                fracs[0, 0, 0][ns04 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            speccomb = colamt[0, 0, 0][2] + rfrate[0, 0, 0][5, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][2] / speccomb
            specmult = 4.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb + js - 1

            speccomb1 = colamt[0, 0, 0][2] + rfrate[0, 0, 0][5, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][2] / speccomb1
            specmult1 = 4.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb + js1 - 1

            speccomb_planck = colamt[0, 0, 0][2] + refrat_planck_b * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][2] / speccomb_planck
            specmult_planck = 4.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)
            jplp = jpl + 1

            id000 = ind0
            id010 = ind0 + 5
            id100 = ind0 + 1
            id110 = ind0 + 6
            id001 = ind1
            id011 = ind1 + 5
            id101 = ind1 + 1
            id111 = ind1 + 6

            fk0 = 1.0 - fs
            fk1 = fs
            fac000 = fk0 * fac00
            fac010 = fk0 * fac10
            fac100 = fk1 * fac00
            fac110 = fk1 * fac10

            fk0 = 1.0 - fs1
            fk1 = fs1
            fac001 = fk0 * fac01
            fac011 = fk0 * fac11
            fac101 = fk1 * fac01
            fac111 = fk1 * fac11

            for ig2 in range(ng04):
                tau_major = speccomb * (
                    fac000 * absb[0, 0, 0][ig2, id000]
                    + fac010 * absb[0, 0, 0][ig2, id010]
                    + fac100 * absb[0, 0, 0][ig2, id100]
                    + fac110 * absb[0, 0, 0][ig2, id110]
                )
                tau_major1 = speccomb1 * (
                    fac001 * absb[0, 0, 0][ig2, id001]
                    + fac011 * absb[0, 0, 0][ig2, id011]
                    + fac101 * absb[0, 0, 0][ig2, id101]
                    + fac111 * absb[0, 0, 0][ig2, id111]
                )

                taug[0, 0, 0][ns04 + ig2] = tau_major + tau_major1

                fracs[0, 0, 0][ns04 + ig2] = fracrefb[0, 0, 0][ig2, jpl] + fpl * (
                    fracrefb[0, 0, 0][ig2, jplp] - fracrefb[0, 0, 0][ig2, jpl]
                )

            taug[0, 0, 0][ns04 + 7] = taug[0, 0, 0][ns04 + 7] * 0.92
            taug[0, 0, 0][ns04 + 8] = taug[0, 0, 0][ns04 + 8] * 0.88
            taug[0, 0, 0][ns04 + 9] = taug[0, 0, 0][ns04 + 9] * 1.07
            taug[0, 0, 0][ns04 + 10] = taug[0, 0, 0][ns04 + 10] * 1.1
            taug[0, 0, 0][ns04 + 11] = taug[0, 0, 0][ns04 + 11] * 0.99
            taug[0, 0, 0][ns04 + 12] = taug[0, 0, 0][ns04 + 12] * 0.88
            taug[0, 0, 0][ns04 + 13] = taug[0, 0, 0][ns04 + 13] * 0.943


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[4],
        "nspb": nspb[4],
        "ng05": ng05,
        "ns05": ns05,
        "oneminus": oneminus,
    },
)
def taugb05(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    wx: Field[type_maxxsec],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    indminor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng05, 585))],
    absb: Field[(DTYPE_FLT, (ng05, 1175))],
    selfref: Field[(DTYPE_FLT, (ng05, 10))],
    forref: Field[(DTYPE_FLT, (ng05, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng05, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng05, 5))],
    ka_mo3: Field[(DTYPE_FLT, (ng05, 9, 19))],
    ccl4: Field[(DTYPE_FLT, (ng05,))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    js: FIELD_INT,
    js1: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    jmo3: FIELD_INT,
    jmo3p: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng05, ns05, oneminus

    with computation(PARALLEL):
        with interval(...):
            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 4] / chi_mls[0, 0, 0][1, 4]
            )  # P = 142.5940 mb
            refrat_planck_b = (
                chi_mls[0, 0, 0][2, 42] / chi_mls[0, 0, 0][1, 42]
            )  # P = 95.58350 mb
            refrat_m_a = chi_mls[0, 0, 0][0, 6] / chi_mls[0, 0, 0][1, 6]

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            speccomb_mo3 = colamt[0, 0, 0][0] + refrat_m_a * colamt[0, 0, 0][1]
            specparm_mo3 = colamt[0, 0, 0][0] / speccomb_mo3
            specmult_mo3 = 8.0 * min(specparm_mo3, oneminus)
            jmo3 = 1 + specmult_mo3 - 1
            fmo3 = mod(specmult_mo3, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1
            jplp = jpl + 1
            jmo3p = jmo3 + 1

            # Workaround for bug in gt4py, can be removed at release of next tag (>32)
            id000 = id000
            id010 = id010
            id100 = id100
            id110 = id110
            id200 = id200
            id210 = id210

            if specparm < 0.125:
                p0 = fs - 1.0
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p0 = -fs
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk00 = 1.0 - fs
                fk10 = fs
                fk20 = 0.0

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0
                id210 = ind0

            id001 = id001
            id011 = id011
            id101 = id101
            id111 = id111
            id201 = id201
            id211 = id211

            fac000 = fk00 * fac00
            fac100 = fk10 * fac00
            fac200 = fk20 * fac00
            fac010 = fk00 * fac10
            fac110 = fk10 * fac10
            fac210 = fk20 * fac10

            if specparm1 < 0.125:
                p1 = fs1 - 1.0
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p1 = -fs1
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk01 = 1.0 - fs1
                fk11 = fs1
                fk21 = 0.0

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk01 * fac01
            fac101 = fk11 * fac01
            fac201 = fk21 * fac01
            fac011 = fk01 * fac11
            fac111 = fk11 * fac11
            fac211 = fk21 * fac11

            for ig in range(ng05):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                o3m1 = ka_mo3[0, 0, 0][ig, jmo3, indm] + fmo3 * (
                    ka_mo3[0, 0, 0][ig, jmo3p, indm] - ka_mo3[0, 0, 0][ig, jmo3, indm]
                )
                o3m2 = ka_mo3[0, 0, 0][ig, jmo3, indmp] + fmo3 * (
                    ka_mo3[0, 0, 0][ig, jmo3p, indmp] - ka_mo3[0, 0, 0][ig, jmo3, indmp]
                )
                abso3 = o3m1 + minorfrac * (o3m2 - o3m1)

                taug[0, 0, 0][ns05 + ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                    + tauself
                    + taufor
                    + abso3 * colamt[0, 0, 0][2]
                    + wx[0, 0, 0][0] * ccl4[0, 0, 0][ig]
                )

                fracs[0, 0, 0][ns05 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            speccomb = colamt[0, 0, 0][2] + rfrate[0, 0, 0][5, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][2] / speccomb
            specmult = 4.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb + js - 1

            speccomb1 = colamt[0, 0, 0][2] + rfrate[0, 0, 0][5, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][2] / speccomb1
            specmult1 = 4.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb + js1 - 1

            speccomb_planck = colamt[0, 0, 0][2] + refrat_planck_b * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][2] / speccomb_planck
            specmult_planck = 4.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)
            jplp = jpl + 1

            id000 = ind0
            id010 = ind0 + 5
            id100 = ind0 + 1
            id110 = ind0 + 6
            id001 = ind1
            id011 = ind1 + 5
            id101 = ind1 + 1
            id111 = ind1 + 6

            fk00 = 1.0 - fs
            fk10 = fs

            fk01 = 1.0 - fs1
            fk11 = fs1

            fac000 = fk00 * fac00
            fac010 = fk00 * fac10
            fac100 = fk10 * fac00
            fac110 = fk10 * fac10

            fac001 = fk01 * fac01
            fac011 = fk01 * fac11
            fac101 = fk11 * fac01
            fac111 = fk11 * fac11

            for ig2 in range(ng05):
                taug[0, 0, 0][ns05 + ig2] = (
                    speccomb
                    * (
                        fac000 * absb[0, 0, 0][ig2, id000]
                        + fac010 * absb[0, 0, 0][ig2, id010]
                        + fac100 * absb[0, 0, 0][ig2, id100]
                        + fac110 * absb[0, 0, 0][ig2, id110]
                    )
                    + speccomb1
                    * (
                        fac001 * absb[0, 0, 0][ig2, id001]
                        + fac011 * absb[0, 0, 0][ig2, id011]
                        + fac101 * absb[0, 0, 0][ig2, id101]
                        + fac111 * absb[0, 0, 0][ig2, id111]
                    )
                    + wx[0, 0, 0][0] * ccl4[0, 0, 0][ig2]
                )

                fracs[0, 0, 0][ns05 + ig2] = fracrefb[0, 0, 0][ig2, jpl] + fpl * (
                    fracrefb[0, 0, 0][ig2, jplp] - fracrefb[0, 0, 0][ig2, jpl]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[5],
        "ng06": ng06,
        "ns06": ns06,
    },
)
def taugb06(
    laytrop: FIELD_BOOL,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    wx: Field[type_maxxsec],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    indminor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng06, 65))],
    selfref: Field[(DTYPE_FLT, (ng06, 10))],
    forref: Field[(DTYPE_FLT, (ng06, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng06,))],
    ka_mco2: Field[(DTYPE_FLT, (ng06, 19))],
    cfc11adj: Field[(DTYPE_FLT, (ng06,))],
    cfc12: Field[(DTYPE_FLT, (ng06,))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    ratco2: FIELD_FLT,
):
    from __externals__ import nspa, ng06, ns06

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa
            ind1 = (jp * 5 + (jt1 - 1)) * nspa

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1

            temp = coldry * chi_mls[0, 0, 0][1, jp]
            ratco2 = colamt[0, 0, 0][1] / temp
            if ratco2 > 3.0:
                adjfac = 2.0 + (ratco2 - 2.0) ** 0.77
                adjcolco2 = adjfac * temp
            else:
                adjcolco2 = colamt[0, 0, 0][1]

            for ig in range(ng06):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                absco2 = ka_mco2[0, 0, 0][ig, indm] + minorfrac * (
                    ka_mco2[0, 0, 0][ig, indmp] - ka_mco2[0, 0, 0][ig, indm]
                )

                taug[0, 0, 0][ns06 + ig] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ig, ind0]
                        + fac10 * absa[0, 0, 0][ig, ind0p]
                        + fac01 * absa[0, 0, 0][ig, ind1]
                        + fac11 * absa[0, 0, 0][ig, ind1p]
                    )
                    + tauself
                    + taufor
                    + adjcolco2 * absco2
                    + wx[0, 0, 0][1] * cfc11adj[0, 0, 0][ig]
                    + wx[0, 0, 0][2] * cfc12[0, 0, 0][ig]
                )

                fracs[0, 0, 0][ns06 + ig] = fracrefa[0, 0, 0][ig]

        else:
            for ig2 in range(ng06):
                taug[0, 0, 0][ns06 + ig2] = (
                    wx[0, 0, 0][1] * cfc11adj[0, 0, 0][ig2]
                    + wx[0, 0, 0][2] * cfc12[0, 0, 0][ig2]
                )

                fracs[0, 0, 0][ns06 + ig2] = fracrefa[0, 0, 0][ig2]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[6],
        "nspb": nspb[6],
        "ng07": ng07,
        "ns07": ns07,
        "oneminus": oneminus,
    },
)
def taugb07(
    laytrop: FIELD_BOOL,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    indminor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng07, 585))],
    absb: Field[(DTYPE_FLT, (ng07, 235))],
    selfref: Field[(DTYPE_FLT, (ng07, 10))],
    forref: Field[(DTYPE_FLT, (ng07, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng07, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng07,))],
    ka_mco2: Field[(DTYPE_FLT, (ng07, 9, 19))],
    kb_mco2: Field[(DTYPE_FLT, (ng07, 19))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    js: FIELD_INT,
    js1: FIELD_INT,
    jmco2: FIELD_INT,
    jmco2p: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
    ratco2: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng07, ns07, oneminus

    with computation(PARALLEL):
        with interval(...):
            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][2, 2]
            )  # P = 706.2620 mb
            refrat_m_a = (
                chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][2, 2]
            )  # P = 706.2720 mb

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][1, 0] * colamt[0, 0, 0][2]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][1, 1] * colamt[0, 0, 0][2]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            speccomb_mco2 = colamt[0, 0, 0][0] + refrat_m_a * colamt[0, 0, 0][2]
            specparm_mco2 = colamt[0, 0, 0][0] / speccomb_mco2
            specmult_mco2 = 8.0 * min(specparm_mco2, oneminus)
            jmco2 = 1 + specmult_mco2 - 1
            fmco2 = mod(specmult_mco2, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][2]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1
            jplp = jpl + 1
            jmco2p = jmco2 + 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1

            temp = coldry * chi_mls[0, 0, 0][1, jp]
            ratco2 = colamt[0, 0, 0][1] / temp
            if ratco2 > 3.0:
                adjfac = 3.0 + (ratco2 - 3.0) ** 0.79
                adjcolco2 = adjfac * temp
            else:
                adjcolco2 = colamt[0, 0, 0][1]

            # Workaround for bug in gt4py, can be removed at release of next tag (>32)
            id000 = id000
            id010 = id010
            id100 = id100
            id110 = id110
            id200 = id200
            id210 = id210

            if specparm < 0.125:
                p0 = fs - 1.0
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p0 = -fs
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk00 = 1.0 - fs
                fk10 = fs
                fk20 = 0.0

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0
                id210 = ind0

            fac000 = fk00 * fac00
            fac100 = fk10 * fac00
            fac200 = fk20 * fac00
            fac010 = fk00 * fac10
            fac110 = fk10 * fac10
            fac210 = fk20 * fac10

            id001 = id001
            id011 = id011
            id101 = id101
            id111 = id111
            id201 = id201
            id211 = id211

            if specparm1 < 0.125:
                p1 = fs1 - 1.0
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p1 = -fs1
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk01 = 1.0 - fs1
                fk11 = fs1
                fk21 = 0.0

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk01 * fac01
            fac101 = fk11 * fac01
            fac201 = fk21 * fac01
            fac011 = fk01 * fac11
            fac111 = fk11 * fac11
            fac211 = fk21 * fac11

            for ig in range(ng07):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                co2m1 = ka_mco2[0, 0, 0][ig, jmco2, indm] + fmco2 * (
                    ka_mco2[0, 0, 0][ig, jmco2p, indm]
                    - ka_mco2[0, 0, 0][ig, jmco2, indm]
                )
                co2m2 = ka_mco2[0, 0, 0][ig, jmco2, indmp] + fmco2 * (
                    ka_mco2[0, 0, 0][ig, jmco2p, indmp]
                    - ka_mco2[0, 0, 0][ig, jmco2, indmp]
                )
                absco2 = co2m1 + minorfrac * (co2m2 - co2m1)

                taug[0, 0, 0][ns07 + ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                    + tauself
                    + taufor
                    + adjcolco2 * absco2
                )

                fracs[0, 0, 0][ns07 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            temp = coldry * chi_mls[0, 0, 0][1, jp]
            ratco2 = colamt[0, 0, 0][1] / temp
            if ratco2 > 3.0:
                adjfac = 2.0 + (ratco2 - 2.0) ** 0.79
                adjcolco2 = adjfac * temp
            else:
                adjcolco2 = colamt[0, 0, 0][1]

            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            indm = indminor - 1
            indmp = indm + 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1

            for ig2 in range(ng07):
                absco2 = kb_mco2[0, 0, 0][ig2, indm] + minorfrac * (
                    kb_mco2[0, 0, 0][ig2, indmp] - kb_mco2[0, 0, 0][ig2, indm]
                )

                taug[0, 0, 0][ns07 + ig2] = (
                    colamt[0, 0, 0][2]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                    + adjcolco2 * absco2
                )

                fracs[0, 0, 0][ns07 + ig2] = fracrefb[0, 0, 0][ig2]

            taug[0, 0, 0][ns07 + 5] = taug[0, 0, 0][ns07 + 5] * 0.92
            taug[0, 0, 0][ns07 + 6] = taug[0, 0, 0][ns07 + 6] * 0.88
            taug[0, 0, 0][ns07 + 7] = taug[0, 0, 0][ns07 + 7] * 1.07
            taug[0, 0, 0][ns07 + 8] = taug[0, 0, 0][ns07 + 8] * 1.1
            taug[0, 0, 0][ns07 + 9] = taug[0, 0, 0][ns07 + 9] * 0.99
            taug[0, 0, 0][ns07 + 10] = taug[0, 0, 0][ns07 + 10] * 0.855


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[7],
        "nspb": nspb[7],
        "ng08": ng08,
        "ns08": ns08,
    },
)
def taugb08(
    laytrop: FIELD_BOOL,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    wx: Field[type_maxxsec],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    indminor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng08, 65))],
    absb: Field[(DTYPE_FLT, (ng08, 235))],
    selfref: Field[(DTYPE_FLT, (ng08, 10))],
    forref: Field[(DTYPE_FLT, (ng08, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng08,))],
    fracrefb: Field[(DTYPE_FLT, (ng08,))],
    ka_mo3: Field[(DTYPE_FLT, (ng08, 19))],
    ka_mco2: Field[(DTYPE_FLT, (ng08, 19))],
    kb_mco2: Field[(DTYPE_FLT, (ng08, 19))],
    cfc12: Field[(DTYPE_FLT, (ng08,))],
    ka_mn2o: Field[(DTYPE_FLT, (ng08, 19))],
    kb_mn2o: Field[(DTYPE_FLT, (ng08, 19))],
    cfc22adj: Field[(DTYPE_FLT, (ng08,))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    ratco2: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng08, ns08

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa
            ind1 = (jp * 5 + (jt1 - 1)) * nspa

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1

            temp = coldry * chi_mls[0, 0, 0][1, jp]
            ratco2 = colamt[0, 0, 0][1] / temp
            if ratco2 > 3.0:
                adjfac = 2.0 + (ratco2 - 2.0) ** 0.65
                adjcolco2 = adjfac * temp
            else:
                adjcolco2 = colamt[0, 0, 0][1]

            for ig in range(ng08):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                absco2 = ka_mco2[0, 0, 0][ig, indm] + minorfrac * (
                    ka_mco2[0, 0, 0][ig, indmp] - ka_mco2[0, 0, 0][ig, indm]
                )
                abso3 = ka_mo3[0, 0, 0][ig, indm] + minorfrac * (
                    ka_mo3[0, 0, 0][ig, indmp] - ka_mo3[0, 0, 0][ig, indm]
                )
                absn2o = ka_mn2o[0, 0, 0][ig, indm] + minorfrac * (
                    ka_mn2o[0, 0, 0][ig, indmp] - ka_mn2o[0, 0, 0][ig, indm]
                )

                taug[0, 0, 0][ns08 + ig] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ig, ind0]
                        + fac10 * absa[0, 0, 0][ig, ind0p]
                        + fac01 * absa[0, 0, 0][ig, ind1]
                        + fac11 * absa[0, 0, 0][ig, ind1p]
                    )
                    + tauself
                    + taufor
                    + adjcolco2 * absco2
                    + colamt[0, 0, 0][2] * abso3
                    + colamt[0, 0, 0][3] * absn2o
                    + wx[0, 0, 0][2] * cfc12[0, 0, 0][ig]
                    + wx[0, 0, 0][3] * cfc22adj[0, 0, 0][ig]
                )

                fracs[0, 0, 0][ns08 + ig] = fracrefa[0, 0, 0][ig]

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            indm = indminor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indmp = indm + 1

            temp = coldry * chi_mls[0, 0, 0][1, jp]
            ratco2 = colamt[0, 0, 0][1] / temp
            if ratco2 > 3.0:
                adjfac = 2.0 + (ratco2 - 2.0) ** 0.65
                adjcolco2 = adjfac * temp
            else:
                adjcolco2 = colamt[0, 0, 0][1]

            for ig2 in range(ng08):
                absco2 = kb_mco2[0, 0, 0][ig2, indm] + minorfrac * (
                    kb_mco2[0, 0, 0][ig2, indmp] - kb_mco2[0, 0, 0][ig2, indm]
                )
                absn2o = kb_mn2o[0, 0, 0][ig2, indm] + minorfrac * (
                    kb_mn2o[0, 0, 0][ig2, indmp] - kb_mn2o[0, 0, 0][ig2, indm]
                )

                taug[0, 0, 0][ns08 + ig2] = (
                    colamt[0, 0, 0][2]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                    + adjcolco2 * absco2
                    + colamt[0, 0, 0][3] * absn2o
                    + wx[0, 0, 0][2] * cfc12[0, 0, 0][ig2]
                    + wx[0, 0, 0][3] * cfc22adj[0, 0, 0][ig2]
                )

                fracs[0, 0, 0][ns08 + ig2] = fracrefb[0, 0, 0][ig2]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[8],
        "nspb": nspb[8],
        "ng09": ng09,
        "ns09": ns09,
        "oneminus": oneminus,
    },
)
def taugb09(
    laytrop: FIELD_BOOL,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    indminor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng09, 585))],
    absb: Field[(DTYPE_FLT, (ng09, 235))],
    selfref: Field[(DTYPE_FLT, (ng09, 10))],
    forref: Field[(DTYPE_FLT, (ng09, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng09, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng09,))],
    ka_mn2o: Field[(DTYPE_FLT, (ng09, 9, 19))],
    kb_mn2o: Field[(DTYPE_FLT, (ng09, 19))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    js: FIELD_INT,
    js1: FIELD_INT,
    jmn2o: FIELD_INT,
    jmn2op: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
    ratn2o: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng09, ns09, oneminus

    with computation(PARALLEL):
        with interval(...):
            #  --- ...  calculate reference ratio to be used in calculation of Planck
            #           fraction in lower/upper atmosphere.

            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 8] / chi_mls[0, 0, 0][5, 8]
            )  # P = 212 mb
            refrat_m_a = (
                chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][5, 2]
            )  # P = 706.272 mb

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][3, 0] * colamt[0, 0, 0][4]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][3, 1] * colamt[0, 0, 0][4]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            speccomb_mn2o = colamt[0, 0, 0][0] + refrat_m_a * colamt[0, 0, 0][4]
            specparm_mn2o = colamt[0, 0, 0][0] / speccomb_mn2o
            specmult_mn2o = 8.0 * min(specparm_mn2o, oneminus)
            jmn2o = 1 + specmult_mn2o - 1
            fmn2o = mod(specmult_mn2o, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][4]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1
            jplp = jpl + 1
            jmn2op = jmn2o + 1

            temp = coldry * chi_mls[0, 0, 0][3, jp]
            ratn2o = colamt[0, 0, 0][3] / temp
            if ratn2o > 1.5:
                adjfac = 0.5 + (ratn2o - 0.5) ** 0.65
                adjcoln2o = adjfac * temp
            else:
                adjcoln2o = colamt[0, 0, 0][3]

            # Workaround for bug in gt4py, can be removed at release of next tag (>32)
            id000 = id000
            id010 = id010
            id100 = id100
            id110 = id110
            id200 = id200
            id210 = id210

            if specparm < 0.125:
                p0 = fs - 1.0
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p0 = -fs
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk00 = 1.0 - fs
                fk10 = fs
                fk20 = 0.0

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0
                id210 = ind0

            fac000 = fk00 * fac00
            fac100 = fk10 * fac00
            fac200 = fk20 * fac00
            fac010 = fk00 * fac10
            fac110 = fk10 * fac10
            fac210 = fk20 * fac10

            id001 = id001
            id011 = id011
            id101 = id101
            id111 = id111
            id201 = id201
            id211 = id211

            if specparm1 < 0.125:
                p1 = fs1 - 1.0
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p1 = -fs1
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk01 = 1.0 - fs1
                fk11 = fs1
                fk21 = 0.0

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk01 * fac01
            fac101 = fk11 * fac01
            fac201 = fk21 * fac01
            fac011 = fk01 * fac11
            fac111 = fk11 * fac11
            fac211 = fk21 * fac11

            for ig in range(ng09):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                n2om1 = ka_mn2o[0, 0, 0][ig, jmn2o, indm] + fmn2o * (
                    ka_mn2o[0, 0, 0][ig, jmn2op, indm]
                    - ka_mn2o[0, 0, 0][ig, jmn2o, indm]
                )
                n2om2 = ka_mn2o[0, 0, 0][ig, jmn2o, indmp] + fmn2o * (
                    ka_mn2o[0, 0, 0][ig, jmn2op, indmp]
                    - ka_mn2o[0, 0, 0][ig, jmn2o, indmp]
                )
                absn2o = n2om1 + minorfrac * (n2om2 - n2om1)

                taug[0, 0, 0][ns09 + ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                    + tauself
                    + taufor
                    + adjcoln2o * absn2o
                )

                fracs[0, 0, 0][ns09 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            indm = indminor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indmp = indm + 1

            temp = coldry * chi_mls[0, 0, 0][3, jp]
            ratn2o = colamt[0, 0, 0][3] / temp
            if ratn2o > 1.5:
                adjfac = 0.5 + (ratn2o - 0.5) ** 0.65
                adjcoln2o = adjfac * temp
            else:
                adjcoln2o = colamt[0, 0, 0][3]

            for ig2 in range(ng09):
                absn2o = kb_mn2o[0, 0, 0][ig2, indm] + minorfrac * (
                    kb_mn2o[0, 0, 0][ig2, indmp] - kb_mn2o[0, 0, 0][ig2, indm]
                )

                taug[0, 0, 0][ns09 + ig2] = (
                    colamt[0, 0, 0][4]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                    + adjcoln2o * absn2o
                )

                fracs[0, 0, 0][ns09 + ig2] = fracrefb[0, 0, 0][ig2]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[9],
        "nspb": nspb[9],
        "ng10": ng10,
        "ns10": ns10,
    },
)
def taugb10(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng10, 65))],
    absb: Field[(DTYPE_FLT, (ng10, 235))],
    selfref: Field[(DTYPE_FLT, (ng10, 10))],
    forref: Field[(DTYPE_FLT, (ng10, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng10,))],
    fracrefb: Field[(DTYPE_FLT, (ng10,))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng10, ns10

    with computation(PARALLEL), interval(1, None):
        inds = inds
        tauself = tauself
        if laytrop:
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa
            ind1 = (jp * 5 + (jt1 - 1)) * nspa

            inds = indself - 1
            indf = indfor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indsp = inds + 1
            indfp = indf + 1

            for ig in range(ng10):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )

                taug[0, 0, 0][ns10 + ig] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ig, ind0]
                        + fac10 * absa[0, 0, 0][ig, ind0p]
                        + fac01 * absa[0, 0, 0][ig, ind1]
                        + fac11 * absa[0, 0, 0][ig, ind1p]
                    )
                    + tauself
                    + taufor
                )

                fracs[0, 0, 0][ns10 + ig] = fracrefa[0, 0, 0][ig]

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            indf = indfor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indfp = indf + 1

            for ig2 in range(ng10):
                taufor = forfac * (
                    forref[0, 0, 0][ig2, indf]
                    + forfrac
                    * (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])
                )

                taug[0, 0, 0][ns10 + ig2] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                    + taufor
                )

                fracs[0, 0, 0][ns10 + ig2] = fracrefb[0, 0, 0][ig2]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[10],
        "nspb": nspb[10],
        "ng11": ng11,
        "ns11": ns11,
    },
)
def taugb11(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    indminor: FIELD_INT,
    scaleminor: FIELD_FLT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng11, 65))],
    absb: Field[(DTYPE_FLT, (ng11, 235))],
    selfref: Field[(DTYPE_FLT, (ng11, 10))],
    forref: Field[(DTYPE_FLT, (ng11, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng11,))],
    fracrefb: Field[(DTYPE_FLT, (ng11,))],
    ka_mo2: Field[(DTYPE_FLT, (ng11, 19))],
    kb_mo2: Field[(DTYPE_FLT, (ng11, 19))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng11, ns11

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa
            ind1 = (jp * 5 + (jt1 - 1)) * nspa

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1

            scaleo2 = colamt[0, 0, 0][5] * scaleminor

            for ig in range(ng11):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                tauo2 = scaleo2 * (
                    ka_mo2[0, 0, 0][ig, indm]
                    + minorfrac
                    * (ka_mo2[0, 0, 0][ig, indmp] - ka_mo2[0, 0, 0][ig, indm])
                )

                taug[0, 0, 0][ns11 + ig] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ig, ind0]
                        + fac10 * absa[0, 0, 0][ig, ind0p]
                        + fac01 * absa[0, 0, 0][ig, ind1]
                        + fac11 * absa[0, 0, 0][ig, ind1p]
                    )
                    + tauself
                    + taufor
                    + tauo2
                )

                fracs[0, 0, 0][ns11 + ig] = fracrefa[0, 0, 0][ig]

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            indf = indfor - 1
            indm = indminor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indfp = indf + 1
            indmp = indm + 1

            scaleo2 = colamt[0, 0, 0][5] * scaleminor

            for ig2 in range(ng11):
                taufor = forfac * (
                    forref[0, 0, 0][ig2, indf]
                    + forfrac
                    * (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])
                )
                tauo2 = scaleo2 * (
                    kb_mo2[0, 0, 0][ig2, indm]
                    + minorfrac
                    * (kb_mo2[0, 0, 0][ig2, indmp] - kb_mo2[0, 0, 0][ig2, indm])
                )

                taug[0, 0, 0][ns11 + ig2] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                    + taufor
                    + tauo2
                )

                fracs[0, 0, 0][ns11 + ig2] = fracrefb[0, 0, 0][ig2]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[11],
        "nspb": nspb[11],
        "ng12": ng12,
        "ns12": ns12,
        "oneminus": oneminus,
    },
)
def taugb12(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng09, 585))],
    selfref: Field[(DTYPE_FLT, (ng09, 10))],
    forref: Field[(DTYPE_FLT, (ng09, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng09, 9))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    js: FIELD_INT,
    js1: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
    specparm_planck: FIELD_FLT,
):
    from __externals__ import nspa, ng12, ns12, oneminus

    with computation(PARALLEL):
        with interval(...):
            refrat_planck_a = chi_mls[0, 0, 0][0, 9] / chi_mls[0, 0, 0][1, 9]

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            if specparm_planck >= oneminus:
                specparm_planck = oneminus
            specmult_planck = 8.0 * specparm_planck
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1
            jplp = jpl + 1

            # Workaround for bug in gt4py, can be removed at release of next tag (>32)
            id000 = id000
            id010 = id010
            id100 = id100
            id110 = id110
            id200 = id200
            id210 = id210

            if specparm < 0.125:
                p0 = fs - 1.0
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p0 = -fs
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk00 = 1.0 - fs
                fk10 = fs
                fk20 = 0.0

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0
                id210 = ind0

            fac000 = fk00 * fac00
            fac100 = fk10 * fac00
            fac200 = fk20 * fac00
            fac010 = fk00 * fac10
            fac110 = fk10 * fac10
            fac210 = fk20 * fac10

            id001 = id001
            id011 = id011
            id101 = id101
            id111 = id111
            id201 = id201
            id211 = id211

            if specparm1 < 0.125:
                p1 = fs1 - 1.0
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p1 = -fs1
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk01 = 1.0 - fs1
                fk11 = fs1
                fk21 = 0.0

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk01 * fac01
            fac101 = fk11 * fac01
            fac201 = fk21 * fac01
            fac011 = fk01 * fac11
            fac111 = fk11 * fac11
            fac211 = fk21 * fac11

            for ig in range(ng12):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )

                taug[0, 0, 0][ns12 + ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                    + tauself
                    + taufor
                )

                fracs[0, 0, 0][ns12 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            for ig2 in range(ng12):
                taug[0, 0, 0][ns12 + ig2] = 0.0
                fracs[0, 0, 0][ns12 + ig2] = 0.0


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[12],
        "nspb": nspb[12],
        "ng13": ng13,
        "ns13": ns13,
        "oneminus": oneminus,
    },
)
def taugb13(
    laytrop: FIELD_BOOL,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    indminor: FIELD_INT,
    minorfrac: FIELD_FLT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng13, 585))],
    selfref: Field[(DTYPE_FLT, (ng13, 10))],
    forref: Field[(DTYPE_FLT, (ng13, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng13, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng13,))],
    ka_mco: Field[(DTYPE_FLT, (ng13, 9, 19))],
    ka_mco2: Field[(DTYPE_FLT, (ng13, 9, 19))],
    kb_mo3: Field[(DTYPE_FLT, (ng13, 19))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    js: FIELD_INT,
    js1: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    jmco: FIELD_INT,
    jmcop: FIELD_INT,
    jmco2: FIELD_INT,
    jmco2p: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
    ratco2: FIELD_FLT,
):
    from __externals__ import nspa, ng13, ns13, oneminus

    with computation(PARALLEL):
        with interval(...):
            #  --- ...  calculate reference ratio to be used in calculation of Planck
            #           fraction in lower/upper atmosphere.

            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 4] / chi_mls[0, 0, 0][3, 4]
            )  # P = 473.420 mb (Level 5)
            refrat_m_a = (
                chi_mls[0, 0, 0][0, 0] / chi_mls[0, 0, 0][3, 0]
            )  # P = 1053. (Level 1)
            refrat_m_a3 = (
                chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][3, 2]
            )  # P = 706. (Level 3)

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][2, 0] * colamt[0, 0, 0][3]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][2, 1] * colamt[0, 0, 0][3]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            speccomb_mco2 = colamt[0, 0, 0][0] + refrat_m_a * colamt[0, 0, 0][3]
            specparm_mco2 = colamt[0, 0, 0][0] / speccomb_mco2
            specmult_mco2 = 8.0 * min(specparm_mco2, oneminus)
            jmco2 = 1 + specmult_mco2 - 1
            fmco2 = mod(specmult_mco2, 1.0)

            speccomb_mco = colamt[0, 0, 0][0] + refrat_m_a3 * colamt[0, 0, 0][3]
            specparm_mco = colamt[0, 0, 0][0] / speccomb_mco
            specmult_mco = 8.0 * min(specparm_mco, oneminus)
            jmco = 1 + specmult_mco - 1
            fmco = mod(specmult_mco, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][3]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1
            jplp = jpl + 1
            jmco2p = jmco2 + 1
            jmcop = jmco + 1

            temp = coldry * 3.55e-4
            ratco2 = colamt[0, 0, 0][1] / temp
            if ratco2 > 3.0:
                adjfac = 2.0 + (ratco2 - 2.0) ** 0.68
                adjcolco2 = adjfac * temp
            else:
                adjcolco2 = colamt[0, 0, 0][1]

            # Workaround for bug in gt4py, can be removed at release of next tag (>32)
            id000 = id000
            id010 = id010
            id100 = id100
            id110 = id110
            id200 = id200
            id210 = id210

            if specparm < 0.125:
                p0 = fs - 1.0
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p0 = -fs
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk00 = 1.0 - fs
                fk10 = fs
                fk20 = 0.0

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0
                id210 = ind0

            fac000 = fk00 * fac00
            fac100 = fk10 * fac00
            fac200 = fk20 * fac00
            fac010 = fk00 * fac10
            fac110 = fk10 * fac10
            fac210 = fk20 * fac10

            id001 = id001
            id011 = id011
            id101 = id101
            id111 = id111
            id201 = id201
            id211 = id211

            if specparm1 < 0.125:
                p1 = fs1 - 1.0
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p1 = -fs1
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk01 = 1.0 - fs1
                fk11 = fs1
                fk21 = 0.0

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk01 * fac01
            fac101 = fk11 * fac01
            fac201 = fk21 * fac01
            fac011 = fk01 * fac11
            fac111 = fk11 * fac11
            fac211 = fk21 * fac11

            for ig in range(ng13):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                co2m1 = ka_mco2[0, 0, 0][ig, jmco2, indm] + fmco2 * (
                    ka_mco2[0, 0, 0][ig, jmco2p, indm]
                    - ka_mco2[0, 0, 0][ig, jmco2, indm]
                )
                co2m2 = ka_mco2[0, 0, 0][ig, jmco2, indmp] + fmco2 * (
                    ka_mco2[0, 0, 0][ig, jmco2p, indmp]
                    - ka_mco2[0, 0, 0][ig, jmco2, indmp]
                )
                absco2 = co2m1 + minorfrac * (co2m2 - co2m1)
                com1 = ka_mco[0, 0, 0][ig, jmco, indm] + fmco * (
                    ka_mco[0, 0, 0][ig, jmcop, indm] - ka_mco[0, 0, 0][ig, jmco, indm]
                )
                com2 = ka_mco[0, 0, 0][ig, jmco, indmp] + fmco * (
                    ka_mco[0, 0, 0][ig, jmcop, indmp] - ka_mco[0, 0, 0][ig, jmco, indmp]
                )
                absco = com1 + minorfrac * (com2 - com1)

                taug[0, 0, 0][ns13 + ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                    + tauself
                    + taufor
                    + adjcolco2 * absco2
                    + colamt[0, 0, 0][6] * absco
                )

                fracs[0, 0, 0][ns13 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            indm = indminor - 1
            indmp = indm + 1

            for ig2 in range(ng13):
                abso3 = kb_mo3[0, 0, 0][ig2, indm] + minorfrac * (
                    kb_mo3[0, 0, 0][ig2, indmp] - kb_mo3[0, 0, 0][ig2, indm]
                )

                taug[0, 0, 0][ns13 + ig2] = colamt[0, 0, 0][2] * abso3

                fracs[0, 0, 0][ns13 + ig2] = fracrefb[0, 0, 0][ig2]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[13],
        "nspb": nspb[13],
        "ng14": ng14,
        "ns14": ns14,
    },
)
def taugb14(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng11, 65))],
    absb: Field[(DTYPE_FLT, (ng11, 235))],
    selfref: Field[(DTYPE_FLT, (ng11, 10))],
    forref: Field[(DTYPE_FLT, (ng11, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng11,))],
    fracrefb: Field[(DTYPE_FLT, (ng11,))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng14, ns14

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa
            ind1 = (jp * 5 + (jt1 - 1)) * nspa

            inds = indself - 1
            indf = indfor - 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indsp = inds + 1
            indfp = indf + 1

            for ig in range(ng14):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )

                taug[0, 0, 0][ns14 + ig] = (
                    colamt[0, 0, 0][1]
                    * (
                        fac00 * absa[0, 0, 0][ig, ind0]
                        + fac10 * absa[0, 0, 0][ig, ind0p]
                        + fac01 * absa[0, 0, 0][ig, ind1]
                        + fac11 * absa[0, 0, 0][ig, ind1p]
                    )
                    + tauself
                    + taufor
                )

                fracs[0, 0, 0][ns14 + ig] = fracrefa[0, 0, 0][ig]

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            ind0p = ind0 + 1
            ind1p = ind1 + 1

            for ig2 in range(ng14):
                taug[0, 0, 0][ns14 + ig2] = colamt[0, 0, 0][1] * (
                    fac00 * absb[0, 0, 0][ig2, ind0]
                    + fac10 * absb[0, 0, 0][ig2, ind0p]
                    + fac01 * absb[0, 0, 0][ig2, ind1]
                    + fac11 * absb[0, 0, 0][ig2, ind1p]
                )

                fracs[0, 0, 0][ns14 + ig2] = fracrefb[0, 0, 0][ig2]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[14],
        "nspb": nspb[14],
        "ng15": ng15,
        "ns15": ns15,
        "oneminus": oneminus,
    },
)
def taugb15(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    colbrd: FIELD_FLT,
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    indminor: FIELD_INT,
    minorfrac: FIELD_FLT,
    scaleminor: FIELD_FLT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng15, 585))],
    selfref: Field[(DTYPE_FLT, (ng15, 10))],
    forref: Field[(DTYPE_FLT, (ng15, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng15, 9))],
    ka_mn2: Field[(DTYPE_FLT, (ng15, 9, 19))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    taun2: FIELD_FLT,
    js: FIELD_INT,
    js1: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    jmn2: FIELD_INT,
    jmn2p: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    fpl: FIELD_FLT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, ng15, ns15, oneminus

    with computation(PARALLEL):
        with interval(...):

            #  --- ...  calculate reference ratio to be used in calculation of Planck
            #           fraction in lower atmosphere.

            refrat_planck_a = (
                chi_mls[0, 0, 0][3, 0] / chi_mls[0, 0, 0][1, 0]
            )  # P = 1053. mb (Level 1)
            refrat_m_a = chi_mls[0, 0, 0][3, 0] / chi_mls[0, 0, 0][1, 0]  # P = 1053. mb

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][3] + rfrate[0, 0, 0][4, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][3] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][3] + rfrate[0, 0, 0][4, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][3] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            speccomb_mn2 = colamt[0, 0, 0][3] + refrat_m_a * colamt[0, 0, 0][1]
            specparm_mn2 = colamt[0, 0, 0][3] / speccomb_mn2
            specmult_mn2 = 8.0 * min(specparm_mn2, oneminus)
            jmn2 = 1 + specmult_mn2 - 1
            fmn2 = mod(specmult_mn2, 1.0)

            speccomb_planck = colamt[0, 0, 0][3] + refrat_planck_a * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][3] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            scalen2 = colbrd * scaleminor

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1
            jplp = jpl + 1
            jmn2p = jmn2 + 1

            # Workaround for bug in gt4py, can be removed at release of next tag (>32)
            id000 = id000
            id010 = id010
            id100 = id100
            id110 = id110
            id200 = id200
            id210 = id210

            if specparm < 0.125:
                p0 = fs - 1.0
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p0 = -fs
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk00 = 1.0 - fs
                fk10 = fs
                fk20 = 0.0

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0
                id210 = ind0

            fac000 = fk00 * fac00
            fac100 = fk10 * fac00
            fac200 = fk20 * fac00
            fac010 = fk00 * fac10
            fac110 = fk10 * fac10
            fac210 = fk20 * fac10

            id001 = id001
            id011 = id011
            id101 = id101
            id111 = id111
            id201 = id201
            id211 = id211

            if specparm1 < 0.125:
                p1 = fs1 - 1.0
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p1 = -fs1
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk01 = 1.0 - fs1
                fk11 = fs1
                fk21 = 0.0

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk01 * fac01
            fac101 = fk11 * fac01
            fac201 = fk21 * fac01
            fac011 = fk01 * fac11
            fac111 = fk11 * fac11
            fac211 = fk21 * fac11

            for ig in range(ng15):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                n2m1 = ka_mn2[0, 0, 0][ig, jmn2, indm] + fmn2 * (
                    ka_mn2[0, 0, 0][ig, jmn2p, indm] - ka_mn2[0, 0, 0][ig, jmn2, indm]
                )
                n2m2 = ka_mn2[0, 0, 0][ig, jmn2, indmp] + fmn2 * (
                    ka_mn2[0, 0, 0][ig, jmn2p, indmp] - ka_mn2[0, 0, 0][ig, jmn2, indmp]
                )
                taun2 = scalen2 * (n2m1 + minorfrac * (n2m2 - n2m1))

                taug[0, 0, 0][ns15 + ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                    + tauself
                    + taufor
                    + taun2
                )

                fracs[0, 0, 0][ns15 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            for ig2 in range(ng15):
                taug[0, 0, 0][ns15 + ig2] = 0.0
                fracs[0, 0, 0][ns15 + ig2] = 0.0


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[15],
        "nspb": nspb[15],
        "ng16": ng16,
        "ns16": ns16,
        "oneminus": oneminus,
    },
)
def taugb16(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng16, 585))],
    absb: Field[(DTYPE_FLT, (ng16, 235))],
    selfref: Field[(DTYPE_FLT, (ng16, 10))],
    forref: Field[(DTYPE_FLT, (ng16, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng16, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng16,))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    js: FIELD_INT,
    js1: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    fpl: FIELD_FLT,
    # temporaries below here only necessary to work around a bug in gt4py
    # hopefully can be removed later
    speccomb: FIELD_FLT,
    speccomb1: FIELD_FLT,
    fac000: FIELD_FLT,
    fac100: FIELD_FLT,
    fac200: FIELD_FLT,
    fac010: FIELD_FLT,
    fac110: FIELD_FLT,
    fac210: FIELD_FLT,
    fac001: FIELD_FLT,
    fac101: FIELD_FLT,
    fac201: FIELD_FLT,
    fac011: FIELD_FLT,
    fac111: FIELD_FLT,
    fac211: FIELD_FLT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng16, ns16, oneminus

    with computation(PARALLEL):
        with interval(...):
            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 5] / chi_mls[0, 0, 0][5, 5]
            )  # P = 387. mb (Level 6)

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][3, 0] * colamt[0, 0, 0][4]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][3, 1] * colamt[0, 0, 0][4]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][4]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1
            jplp = jpl + 1

            # Workaround for bug in gt4py, can be removed at release of next tag (>32)
            id000 = id000
            id010 = id010
            id100 = id100
            id110 = id110
            id200 = id200
            id210 = id210

            if specparm < 0.125:
                p0 = fs - 1.0
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p0 = -fs
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk00 = 1.0 - fs
                fk10 = fs
                fk20 = 0.0

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0
                id210 = ind0

            fac000 = fk00 * fac00
            fac100 = fk10 * fac00
            fac200 = fk20 * fac00
            fac010 = fk00 * fac10
            fac110 = fk10 * fac10
            fac210 = fk20 * fac10

            id001 = id001
            id011 = id011
            id101 = id101
            id111 = id111
            id201 = id201
            id211 = id211

            if specparm1 < 0.125:
                p1 = fs1 - 1.0
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p1 = -fs1
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk01 = 1.0 - fs1
                fk11 = fs1
                fk21 = 0.0

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk01 * fac01
            fac101 = fk11 * fac01
            fac201 = fk21 * fac01
            fac011 = fk01 * fac11
            fac111 = fk11 * fac11
            fac211 = fk21 * fac11

            for ig in range(ng16):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )

                taug[0, 0, 0][ns16 + ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                    + tauself
                    + taufor
                )

                fracs[0, 0, 0][ns16 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            ind0p = ind0 + 1
            ind1p = ind1 + 1

            for ig2 in range(ng16):
                taug[0, 0, 0][ns16 + ig2] = colamt[0, 0, 0][4] * (
                    fac00 * absb[0, 0, 0][ig2, ind0]
                    + fac10 * absb[0, 0, 0][ig2, ind0p]
                    + fac01 * absb[0, 0, 0][ig2, ind1]
                    + fac11 * absb[0, 0, 0][ig2, ind1p]
                )

                fracs[0, 0, 0][ns16 + ig2] = fracrefb[0, 0, 0][ig2]


@stencil(backend=backend, rebuild=rebuild, externals={"ngptlw": ngptlw})
def combine_optical_depth(
    NGB: Field[gtscript.IJ, (DTYPE_INT, (ngptlw,))],
    ib: FIELD_2DINT,
    taug: Field[type_ngptlw],
    tauaer: Field[type_nbands],
    tautot: Field[type_ngptlw],
):
    from __externals__ import ngptlw

    with computation(FORWARD), interval(1, None):
        for ig in range(ngptlw):
            ib = NGB[0, 0][ig] - 1

            tautot[0, 0, 0][ig] = taug[0, 0, 0][ig] + tauaer[0, 0, 0][ib]


rec_6 = 0.166667
tblint = ntbl
flxfac = wtdiff * fluxfac
lhlw0 = True


@stencil(
    backend,
    rebuild=rebuild,
    externals={
        "rec_6": rec_6,
        "bpade": bpade,
        "tblint": tblint,
        "eps": eps,
        "flxfac": flxfac,
        "heatfac": heatfac,
        "lhlw0": lhlw0,
    },
)
def rtrnmc(
    semiss: Field[gtscript.IJ, type_nbands],
    secdif: Field[gtscript.IJ, type_nbands],
    delp: FIELD_FLT,
    taucld: Field[type_nbands],
    fracs: Field[type_ngptlw],
    tautot: Field[type_ngptlw],
    cldfmc: Field[type_ngptlw],
    pklay: Field[type_nbands],
    pklev: Field[type_nbands],
    exp_tbl: Field[type_ntbmx],
    tau_tbl: Field[type_ntbmx],
    tfn_tbl: Field[type_ntbmx],
    NGB: Field[gtscript.IJ, (np.int32, (140,))],
    totuflux: FIELD_FLT,
    totdflux: FIELD_FLT,
    totuclfl: FIELD_FLT,
    totdclfl: FIELD_FLT,
    upfxc_t: FIELD_2D,
    upfx0_t: FIELD_2D,
    upfxc_s: FIELD_2D,
    upfx0_s: FIELD_2D,
    dnfxc_s: FIELD_2D,
    dnfx0_s: FIELD_2D,
    hlwc: FIELD_FLT,
    hlw0: FIELD_FLT,
    clrurad: Field[type_nbands],
    clrdrad: Field[type_nbands],
    toturad: Field[type_nbands],
    totdrad: Field[type_nbands],
    gassrcu: Field[type_ngptlw],
    totsrcu: Field[type_ngptlw],
    trngas: Field[type_ngptlw],
    efclrfr: Field[type_ngptlw],
    rfdelp: FIELD_FLT,
    fnet: FIELD_FLT,
    fnetc: FIELD_FLT,
    totsrcd: Field[type_ngptlw],
    gassrcd: Field[type_ngptlw],
    tblind: Field[type_ngptlw],
    odepth: Field[type_ngptlw],
    odtot: Field[type_ngptlw],
    odcld: Field[type_ngptlw],
    atrtot: Field[type_ngptlw],
    atrgas: Field[type_ngptlw],
    reflct: Field[type_ngptlw],
    totfac: Field[type_ngptlw],
    gasfac: Field[type_ngptlw],
    plfrac: Field[type_ngptlw],
    blay: Field[type_ngptlw],
    bbdgas: Field[type_ngptlw],
    bbdtot: Field[type_ngptlw],
    bbugas: Field[type_ngptlw],
    bbutot: Field[type_ngptlw],
    dplnku: Field[type_ngptlw],
    dplnkd: Field[type_ngptlw],
    radtotu: Field[type_ngptlw],
    radclru: Field[type_ngptlw],
    radtotd: Field[type_ngptlw],
    radclrd: Field[type_ngptlw],
    rad0: Field[type_ngptlw],
    clfm: Field[type_ngptlw],
    trng: Field[type_ngptlw],
    gasu: Field[type_ngptlw],
    itgas: Field[(np.int32, (ngptlw,))],
    ittot: Field[(np.int32, (ngptlw,))],
    ib: FIELD_2DINT,
):
    from __externals__ import rec_6, bpade, tblint, eps, flxfac, heatfac, lhlw0

    # Downward radiative transfer loop.
    # - Clear sky, gases contribution
    # - Total sky, gases+clouds contribution
    # - Cloudy layer
    # - Total sky radiance
    # - Clear sky radiance
    with computation(FORWARD), interval(-2, -1):
        for ig0 in range(ngptlw):
            ib = NGB[0, 0][ig0] - 1

            # clear sky, gases contribution
            odepth[0, 0, 0][ig0] = max(0.0, secdif[0, 0][ib] * tautot[0, 0, 1][ig0])
            if odepth[0, 0, 0][ig0] <= 0.06:
                atrgas[0, 0, 0][ig0] = (
                    odepth[0, 0, 0][ig0]
                    - 0.5 * odepth[0, 0, 0][ig0] * odepth[0, 0, 0][ig0]
                )
                trng[0, 0, 0][ig0] = 1.0 - atrgas[0, 0, 0][ig0]
                gasfac[0, 0, 0][ig0] = rec_6 * odepth[0, 0, 0][ig0]
            else:
                tblind[0, 0, 0][ig0] = odepth[0, 0, 0][ig0] / (
                    bpade + odepth[0, 0, 0][ig0]
                )
                # Currently itgas needs to be a storage, and can't be a local temporary.
                itgas[0, 0, 0][ig0] = tblint * tblind[0, 0, 0][ig0] + 0.5
                trng[0, 0, 0][ig0] = exp_tbl[0, 0, 0][itgas[0, 0, 0][ig0]]
                atrgas[0, 0, 0][ig0] = 1.0 - trng[0, 0, 0][ig0]
                gasfac[0, 0, 0][ig0] = tfn_tbl[0, 0, 0][itgas[0, 0, 0][ig0]]
                odepth[0, 0, 0][ig0] = tau_tbl[0, 0, 0][itgas[0, 0, 0][ig0]]

            plfrac[0, 0, 0][ig0] = fracs[0, 0, 1][ig0]
            blay[0, 0, 0][ig0] = pklay[0, 0, 1][ib]

            dplnku[0, 0, 0][ig0] = pklev[0, 0, 1][ib] - blay[0, 0, 0][ig0]
            dplnkd[0, 0, 0][ig0] = pklev[0, 0, 0][ib] - blay[0, 0, 0][ig0]
            bbdgas[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (
                blay[0, 0, 0][ig0] + dplnkd[0, 0, 0][ig0] * gasfac[0, 0, 0][ig0]
            )
            bbugas[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (
                blay[0, 0, 0][ig0] + dplnku[0, 0, 0][ig0] * gasfac[0, 0, 0][ig0]
            )
            gassrcd[0, 0, 0][ig0] = bbdgas[0, 0, 0][ig0] * atrgas[0, 0, 0][ig0]
            gassrcu[0, 0, 0][ig0] = bbugas[0, 0, 0][ig0] * atrgas[0, 0, 0][ig0]
            trngas[0, 0, 0][ig0] = trng[0, 0, 0][ig0]

            # total sky, gases+clouds contribution
            clfm[0, 0, 0][ig0] = cldfmc[0, 0, 0][ig0]
            if clfm[0, 0, 0][ig0] >= eps:
                # cloudy layer
                odcld[0, 0, 0][ig0] = secdif[0, 0][ib] * taucld[0, 0, 1][ib]
                efclrfr[0, 0, 0][ig0] = (
                    1.0 - (1.0 - exp(-odcld[0, 0, 0][ig0])) * clfm[0, 0, 0][ig0]
                )
                odtot[0, 0, 0][ig0] = odepth[0, 0, 0][ig0] + odcld[0, 0, 0][ig0]
                if odtot[0, 0, 0][ig0] < 0.06:
                    totfac[0, 0, 0][ig0] = rec_6 * odtot[0, 0, 0][ig0]
                    atrtot[0, 0, 0][ig0] = (
                        odtot[0, 0, 0][ig0]
                        - 0.5 * odtot[0, 0, 0][ig0] * odtot[0, 0, 0][ig0]
                    )
                else:
                    tblind[0, 0, 0][ig0] = odtot[0, 0, 0][ig0] / (
                        bpade + odtot[0, 0, 0][ig0]
                    )
                    ittot[0, 0, 0][ig0] = tblint * tblind[0, 0, 0][ig0] + 0.5
                    totfac[0, 0, 0][ig0] = tfn_tbl[0, 0, 0][ittot[0, 0, 0][ig0]]
                    atrtot[0, 0, 0][ig0] = 1.0 - exp_tbl[0, 0, 0][ittot[0, 0, 0][ig0]]

                bbdtot[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (
                    blay[0, 0, 0][ig0] + dplnkd[0, 0, 0][ig0] * totfac[0, 0, 0][ig0]
                )
                bbutot[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (
                    blay[0, 0, 0][ig0] + dplnku[0, 0, 0][ig0] * totfac[0, 0, 0][ig0]
                )
                totsrcd[0, 0, 0][ig0] = bbdtot[0, 0, 0][ig0] * atrtot[0, 0, 0][ig0]
                totsrcu[0, 0, 0][ig0] = bbutot[0, 0, 0][ig0] * atrtot[0, 0, 0][ig0]

                # total sky radiance
                radtotd[0, 0, 0][ig0] = (
                    radtotd[0, 0, 0][ig0] * trng[0, 0, 0][ig0] * efclrfr[0, 0, 0][ig0]
                    + gassrcd[0, 0, 0][ig0]
                    + clfm[0, 0, 0][ig0]
                    * (totsrcd[0, 0, 0][ig0] - gassrcd[0, 0, 0][ig0])
                )
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig0]

                # clear sky radiance
                radclrd[0, 0, 0][ig0] = (
                    radclrd[0, 0, 0][ig0] * trng[0, 0, 0][ig0] + gassrcd[0, 0, 0][ig0]
                )
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig0]
            else:
                # clear layer

                # total sky radiance
                radtotd[0, 0, 0][ig0] = (
                    radtotd[0, 0, 0][ig0] * trng[0, 0, 0][ig0] + gassrcd[0, 0, 0][ig0]
                )
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig0]

                # clear sky radiance
                radclrd[0, 0, 0][ig0] = (
                    radclrd[0, 0, 0][ig0] * trng[0, 0, 0][ig0] + gassrcd[0, 0, 0][ig0]
                )
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig0]

            reflct[0, 0, 0][ig0] = 1.0 - semiss[0, 0][ib]

    with computation(BACKWARD), interval(0, -2):
        for ig in range(ngptlw):
            ib = NGB[0, 0][ig] - 1

            # clear sky, gases contribution
            odepth[0, 0, 0][ig] = max(0.0, secdif[0, 0][ib] * tautot[0, 0, 1][ig])
            if odepth[0, 0, 0][ig] <= 0.06:
                atrgas[0, 0, 0][ig] = (
                    odepth[0, 0, 0][ig]
                    - 0.5 * odepth[0, 0, 0][ig] * odepth[0, 0, 0][ig]
                )
                trng[0, 0, 0][ig] = 1.0 - atrgas[0, 0, 0][ig]
                gasfac[0, 0, 0][ig] = rec_6 * odepth[0, 0, 0][ig]
            else:
                tblind[0, 0, 0][ig] = odepth[0, 0, 0][ig] / (
                    bpade + odepth[0, 0, 0][ig]
                )
                itgas[0, 0, 0][ig] = tblint * tblind[0, 0, 0][ig] + 0.5
                trng[0, 0, 0][ig] = exp_tbl[0, 0, 0][itgas[0, 0, 0][ig]]
                atrgas[0, 0, 0][ig] = 1.0 - trng[0, 0, 0][ig]
                gasfac[0, 0, 0][ig] = tfn_tbl[0, 0, 0][itgas[0, 0, 0][ig]]
                odepth[0, 0, 0][ig] = tau_tbl[0, 0, 0][itgas[0, 0, 0][ig]]

            plfrac[0, 0, 0][ig] = fracs[0, 0, 1][ig]
            blay[0, 0, 0][ig] = pklay[0, 0, 1][ib]

            dplnku[0, 0, 0][ig] = pklev[0, 0, 1][ib] - blay[0, 0, 0][ig]
            dplnkd[0, 0, 0][ig] = pklev[0, 0, 0][ib] - blay[0, 0, 0][ig]
            bbdgas[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (
                blay[0, 0, 0][ig] + dplnkd[0, 0, 0][ig] * gasfac[0, 0, 0][ig]
            )
            bbugas[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (
                blay[0, 0, 0][ig] + dplnku[0, 0, 0][ig] * gasfac[0, 0, 0][ig]
            )
            gassrcd[0, 0, 0][ig] = bbdgas[0, 0, 0][ig] * atrgas[0, 0, 0][ig]
            gassrcu[0, 0, 0][ig] = bbugas[0, 0, 0][ig] * atrgas[0, 0, 0][ig]
            trngas[0, 0, 0][ig] = trng[0, 0, 0][ig]

            # total sky, gases+clouds contribution
            clfm[0, 0, 0][ig] = cldfmc[0, 0, 1][ig]
            if clfm[0, 0, 0][ig] >= eps:
                # cloudy layer
                odcld[0, 0, 0][ig] = secdif[0, 0][ib] * taucld[0, 0, 1][ib]
                efclrfr[0, 0, 0][ig] = (
                    1.0 - (1.0 - exp(-odcld[0, 0, 0][ig])) * clfm[0, 0, 0][ig]
                )
                odtot[0, 0, 0][ig] = odepth[0, 0, 0][ig] + odcld[0, 0, 0][ig]
                if odtot[0, 0, 0][ig] < 0.06:
                    totfac[0, 0, 0][ig] = rec_6 * odtot[0, 0, 0][ig]
                    atrtot[0, 0, 0][ig] = (
                        odtot[0, 0, 0][ig]
                        - 0.5 * odtot[0, 0, 0][ig] * odtot[0, 0, 0][ig]
                    )
                else:
                    tblind[0, 0, 0][ig] = odtot[0, 0, 0][ig] / (
                        bpade + odtot[0, 0, 0][ig]
                    )
                    ittot[0, 0, 0][ig] = tblint * tblind[0, 0, 0][ig] + 0.5
                    totfac[0, 0, 0][ig] = tfn_tbl[0, 0, 0][ittot[0, 0, 0][ig]]
                    atrtot[0, 0, 0][ig] = 1.0 - exp_tbl[0, 0, 0][ittot[0, 0, 0][ig]]

                bbdtot[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (
                    blay[0, 0, 0][ig] + dplnkd[0, 0, 0][ig] * totfac[0, 0, 0][ig]
                )
                bbutot[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (
                    blay[0, 0, 0][ig] + dplnku[0, 0, 0][ig] * totfac[0, 0, 0][ig]
                )
                totsrcd[0, 0, 0][ig] = bbdtot[0, 0, 0][ig] * atrtot[0, 0, 0][ig]
                totsrcu[0, 0, 0][ig] = bbutot[0, 0, 0][ig] * atrtot[0, 0, 0][ig]

                # total sky radiance
                radtotd[0, 0, 0][ig] = (
                    radtotd[0, 0, 1][ig] * trng[0, 0, 0][ig] * efclrfr[0, 0, 0][ig]
                    + gassrcd[0, 0, 0][ig]
                    + clfm[0, 0, 0][ig] * (totsrcd[0, 0, 0][ig] - gassrcd[0, 0, 0][ig])
                )
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig]

                # clear sky radiance
                radclrd[0, 0, 0][ig] = (
                    radclrd[0, 0, 1][ig] * trng[0, 0, 0][ig] + gassrcd[0, 0, 0][ig]
                )
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig]
            else:
                # clear layer

                # total sky radiance
                radtotd[0, 0, 0][ig] = (
                    radtotd[0, 0, 1][ig] * trng[0, 0, 0][ig] + gassrcd[0, 0, 0][ig]
                )
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig]

                # clear sky radiance
                radclrd[0, 0, 0][ig] = (
                    radclrd[0, 0, 1][ig] * trng[0, 0, 0][ig] + gassrcd[0, 0, 0][ig]
                )
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig]

            reflct[0, 0, 0][ig] = 1.0 - semiss[0, 0][ib]

    # Compute spectral emissivity & reflectance, include the
    # contribution of spectrally varying longwave emissivity and
    # reflection from the surface to the upward radiative transfer.
    # note: spectral and Lambertian reflection are identical for the
    #       diffusivity angle flux integration used here.

    with computation(FORWARD), interval(0, 1):
        for ig2 in range(ngptlw):
            ib = NGB[0, 0][ig2] - 1
            rad0[0, 0, 0][ig2] = (
                semiss[0, 0][ib] * fracs[0, 0, 1][ig2] * pklay[0, 0, 0][ib]
            )

            # Compute total sky radiance
            radtotu[0, 0, 0][ig2] = (
                rad0[0, 0, 0][ig2] + reflct[0, 0, 0][ig2] * radtotd[0, 0, 0][ig2]
            )
            toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, 0][ig2]

            # Compute clear sky radiance
            radclru[0, 0, 0][ig2] = (
                rad0[0, 0, 0][ig2] + reflct[0, 0, 0][ig2] * radclrd[0, 0, 0][ig2]
            )
            clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, 0][ig2]

    # Upward radiative transfer loop
    # - Compute total sky radiance
    # - Compute clear sky radiance

    # toturad holds summed radiance for total sky stream
    # clrurad holds summed radiance for clear sky stream

    with computation(FORWARD), interval(0, 1):
        for ig3 in range(ngptlw):
            ib = NGB[0, 0][ig3] - 1
            clfm[0, 0, 0][ig3] = cldfmc[0, 0, 1][ig3]
            trng[0, 0, 0][ig3] = trngas[0, 0, 0][ig3]
            gasu[0, 0, 0][ig3] = gassrcu[0, 0, 0][ig3]

            if clfm[0, 0, 0][ig3] > eps:
                #  --- ...  cloudy layer

                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig3] = (
                    radtotu[0, 0, 0][ig3] * trng[0, 0, 0][ig3] * efclrfr[0, 0, 0][ig3]
                    + gasu[0, 0, 0][ig3]
                    + clfm[0, 0, 0][ig3] * (totsrcu[0, 0, 0][ig3] - gasu[0, 0, 0][ig3])
                )

                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig3] = (
                    radclru[0, 0, 0][ig3] * trng[0, 0, 0][ig3] + gasu[0, 0, 0][ig3]
                )

            else:
                #  --- ...  clear layer

                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig3] = (
                    radtotu[0, 0, 0][ig3] * trng[0, 0, 0][ig3] + gasu[0, 0, 0][ig3]
                )

                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig3] = (
                    radclru[0, 0, 0][ig3] * trng[0, 0, 0][ig3] + gasu[0, 0, 0][ig3]
                )

    with computation(FORWARD), interval(1, -1):
        for ig4 in range(ngptlw):
            ib = NGB[0, 0][ig4] - 1
            clfm[0, 0, 0][ig4] = cldfmc[0, 0, 1][ig4]
            trng[0, 0, 0][ig4] = trngas[0, 0, 0][ig4]
            gasu[0, 0, 0][ig4] = gassrcu[0, 0, 0][ig4]

            if clfm[0, 0, 0][ig4] > eps:
                #  --- ...  cloudy layer
                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig4] = (
                    radtotu[0, 0, -1][ig4] * trng[0, 0, 0][ig4] * efclrfr[0, 0, 0][ig4]
                    + gasu[0, 0, 0][ig4]
                    + clfm[0, 0, 0][ig4] * (totsrcu[0, 0, 0][ig4] - gasu[0, 0, 0][ig4])
                )
                toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, -1][ig4]
                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig4] = (
                    radclru[0, 0, -1][ig4] * trng[0, 0, 0][ig4] + gasu[0, 0, 0][ig4]
                )
                clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, -1][ig4]
            else:
                #  --- ...  clear layer
                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig4] = (
                    radtotu[0, 0, -1][ig4] * trng[0, 0, 0][ig4] + gasu[0, 0, 0][ig4]
                )
                toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, -1][ig4]
                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig4] = (
                    radclru[0, 0, -1][ig4] * trng[0, 0, 0][ig4] + gasu[0, 0, 0][ig4]
                )
                clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, -1][ig4]

    with computation(FORWARD), interval(-1, None):
        for ig5 in range(ngptlw):
            ib = NGB[0, 0][ig5] - 1

            if clfm[0, 0, 0][ig5] > eps:
                #  --- ...  cloudy layer
                #  --- ... total sky radiance
                toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, -1][ig5]
                #  --- ... clear sky radiance
                clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, -1][ig5]
            else:
                #  --- ...  clear layer
                #  --- ... total sky radiance
                toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, -1][ig5]
                #  --- ... clear sky radiance
                clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, -1][ig5]

    # Process longwave output from band for total and clear streams.
    # Calculate upward, downward, and net flux.
    with computation(PARALLEL), interval(...):
        for nb in range(nbands):
            totuflux = totuflux + toturad[0, 0, 0][nb]
            totdflux = totdflux + totdrad[0, 0, 0][nb]
            totuclfl = totuclfl + clrurad[0, 0, 0][nb]
            totdclfl = totdclfl + clrdrad[0, 0, 0][nb]

        totuflux = totuflux * flxfac
        totdflux = totdflux * flxfac
        totuclfl = totuclfl * flxfac
        totdclfl = totdclfl * flxfac

    # calculate net fluxes and heating rates (fnet, htr)
    # also compute optional clear sky heating rates (fnetc, htrcl)
    with computation(FORWARD):
        with interval(0, 1):
            # Output surface fluxes
            upfxc_s = totuflux
            upfx0_s = totuclfl
            dnfxc_s = totdflux
            dnfx0_s = totdclfl

            fnet = totuflux - totdflux
            if lhlw0:
                fnetc = totuclfl - totdclfl
        with interval(-1, None):
            # Output TOA fluxes
            upfxc_t = totuflux
            upfx0_t = totuclfl

    with computation(PARALLEL), interval(1, None):
        fnet = totuflux - totdflux
        if lhlw0:
            fnetc = totuclfl - totdclfl

    with computation(PARALLEL), interval(1, None):
        rfdelp = heatfac / delp
        hlwc = (fnet[0, 0, -1] - fnet) * rfdelp
        if lhlw0:
            hlw0 = (fnetc[0, 0, -1] - fnetc) * rfdelp
