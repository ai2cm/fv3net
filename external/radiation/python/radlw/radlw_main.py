import numpy as np
import xarray as xr
import sys
import os
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
from radlw.radlw_param import (
    ntbl,
    nbands,
    nrates,
    delwave,
    ngptlw,
    ngb,
    absrain,
    abssnow0,
    ipat,
    maxgas,
    maxxsec,
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
    ns01,
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
)
from phys_const import con_g, con_avgd, con_cp, con_amd, con_amw, con_amo3
from config import *

np.set_printoptions(precision=15)


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

    def __init__(self, me, iovrlw, isubclw):
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

    def return_initdata(self):
        outdict = {
            "semiss0": self.semiss0,
            "fluxfac": self.fluxfac,
            "heatfac": self.heatfac,
            "exp_tbl": self.exp_tbl,
            "tau_tbl": self.tau_tbl,
            "tfn_tbl": self.tfn_tbl,
        }
        return outdict

    def lwrad(
        self,
        plyr,
        plvl,
        tlyr,
        tlvl,
        qlyr,
        olyr,
        gasvmr,
        clouds,
        icseed,
        aerosols,
        sfemis,
        sfgtmp,
        dzlyr,
        delpin,
        de_lgth,
        npts,
        nlay,
        nlp1,
        lprnt,
        lhlwb,
        lhlw0,
        lflxprf,
        lw_rand_file,
        verbose=False,
    ):

        self.lhlw0 = lhlw0
        self.lhlwb = lhlwb
        self.lflxprf = lflxprf
        self.rand_file = lw_rand_file

        cldfrc = np.zeros(nlp1 + 1)

        totuflux = np.zeros(nlp1)
        totdflux = np.zeros(nlp1)
        totuclfl = np.zeros(nlp1)
        totdclfl = np.zeros(nlp1)
        tz = np.zeros(nlp1)

        htr = np.zeros(nlay)
        htrcl = np.zeros(nlay)
        pavel = np.zeros(nlay)
        tavel = np.zeros(nlay)
        delp = np.zeros(nlay)
        clwp = np.zeros(nlay)
        ciwp = np.zeros(nlay)
        relw = np.zeros(nlay)
        reiw = np.zeros(nlay)
        cda1 = np.zeros(nlay)
        cda2 = np.zeros(nlay)
        cda3 = np.zeros(nlay)
        cda4 = np.zeros(nlay)
        coldry = np.zeros(nlay)
        colbrd = np.zeros(nlay)
        h2ovmr = np.zeros(nlay)
        o3vmr = np.zeros(nlay)
        fac00 = np.zeros(nlay)
        fac01 = np.zeros(nlay)
        fac10 = np.zeros(nlay)
        fac11 = np.zeros(nlay)
        selffac = np.zeros(nlay)
        selffrac = np.zeros(nlay)
        forfac = np.zeros(nlay)
        forfrac = np.zeros(nlay)
        minorfrac = np.zeros(nlay)
        scaleminor = np.zeros(nlay)
        scaleminorn2 = np.zeros(nlay)
        temcol = np.zeros(nlay)
        dz = np.zeros(nlay)

        pklev = np.zeros((nbands, nlp1))
        pklay = np.zeros((nbands, nlp1))

        htrb = np.zeros((nlay, nbands))
        taucld = np.zeros((nbands, nlay))
        tauaer = np.zeros((nbands, nlay))
        fracs = np.zeros((ngptlw, nlay))
        tautot = np.zeros((ngptlw, nlay))

        semiss = np.zeros(nbands)
        secdiff = np.zeros(nbands)

        colamt = np.zeros((nlay, maxgas))

        wx = np.zeros((nlay, maxxsec))

        rfrate = np.zeros((nlay, nrates, 2))

        ipseed = np.zeros(npts)
        jp = np.zeros(nlay, dtype=np.int32)
        jt = np.zeros(nlay, dtype=np.int32)
        jt1 = np.zeros(nlay, dtype=np.int32)
        indself = np.zeros(nlay, dtype=np.int32)
        indfor = np.zeros(nlay, dtype=np.int32)
        indminor = np.zeros(nlay, dtype=np.int32)

        # outputs
        hlwc = np.zeros((npts, nlay))
        cldtau = np.zeros((npts, nlay))

        upfxc_t = np.zeros(npts)
        upfx0_t = np.zeros(npts)
        upfxc_s = np.zeros(npts)
        upfx0_s = np.zeros(npts)
        dnfxc_s = np.zeros(npts)
        dnfx0_s = np.zeros(npts)

        hlw0 = np.zeros((npts, nlay))

        if verbose:
            print("Beginning lwrad . . .")
            print(" ")

        if self.isubclw == 1:
            for i in range(npts):
                ipseed[i] = self.ipsdlw0 + i + 1
        elif self.isubclw == 2:
            for i in range(npts):
                ipseed[i] = icseed[i]

        for iplon in range(npts):
            if sfemis[iplon] > self.eps and sfemis[iplon] <= 1.0:
                for j in range(nbands):
                    semiss[j] = sfemis[iplon]
            else:
                for j in range(nbands):
                    semiss[j] = self.semiss0[j]

            stemp = sfgtmp[iplon]
            if self.iovrlw == 3:
                delgth = de_lgth[iplon]
            else:
                delgth = 0

            tem1 = 100.0 * con_g
            tem2 = 1.0e-20 * 1.0e3 * con_avgd
            tz[0] = tlvl[iplon, 0]

            for k in range(nlay):
                pavel[k] = plyr[iplon, k]
                delp[k] = delpin[iplon, k]
                tavel[k] = tlyr[iplon, k]
                tz[k + 1] = tlvl[iplon, k + 1]
                dz[k] = dzlyr[iplon, k]

                h2ovmr[k] = max(
                    0.0, qlyr[iplon, k] * self.amdw / (1.0 - qlyr[iplon, k])
                )
                o3vmr[k] = max(0.0, olyr[iplon, k] * self.amdo3)

                tem0 = (1.0 - h2ovmr[k]) * con_amd + h2ovmr[k] * con_amw
                coldry[k] = tem2 * delp[k] / (tem1 * tem0 * (1.0 + h2ovmr[k]))
                temcol[k] = 1.0e-12 * coldry[k]

                colamt[k, 0] = max(0.0, coldry[k] * h2ovmr[k])  # h2o
                colamt[k, 1] = max(temcol[k], coldry[k] * gasvmr[iplon, k, 0])  # co2
                colamt[k, 2] = max(temcol[k], coldry[k] * o3vmr[k])  # o3

            if ilwrgas > 0:
                for k in range(nlay):
                    colamt[k, 3] = max(
                        temcol[k], coldry[k] * gasvmr[iplon, k, 1]
                    )  # n2o
                    colamt[k, 4] = max(
                        temcol[k], coldry[k] * gasvmr[iplon, k, 2]
                    )  # ch4
                    colamt[k, 5] = max(0.0, coldry[k] * gasvmr[iplon, k, 3])  # o2
                    colamt[k, 6] = max(0.0, coldry[k] * gasvmr[iplon, k, 4])  # co

                    wx[k, 0] = max(0.0, coldry[k] * gasvmr[iplon, k, 8])  # ccl4
                    wx[k, 1] = max(0.0, coldry[k] * gasvmr[iplon, k, 5])  # cf11
                    wx[k, 2] = max(0.0, coldry[k] * gasvmr[iplon, k, 6])  # cf12
                    wx[k, 3] = max(0.0, coldry[k] * gasvmr[iplon, k, 7])  # cf22
            else:
                for k in range(nlay):
                    colamt[k, 3] = 0.0  # n2o
                    colamt[k, 4] = 0.0  # ch4
                    colamt[k, 5] = 0.0  # o2
                    colamt[k, 6] = 0.0  # co

                    wx[k, 0] = 0.0
                    wx[k, 1] = 0.0
                    wx[k, 2] = 0.0
                    wx[k, 3] = 0.0

            for k in range(nlay):
                for j in range(nbands):
                    tauaer[j, k] = aerosols[iplon, k, j, 0] * (
                        1.0 - aerosols[iplon, k, j, 1]
                    )

            if ilwcliq > 0:
                for k in range(nlay):
                    cldfrc[k + 1] = clouds[iplon, k, 0]
                    clwp[k] = clouds[iplon, k, 1]
                    relw[k] = clouds[iplon, k, 2]
                    ciwp[k] = clouds[iplon, k, 3]
                    reiw[k] = clouds[iplon, k, 4]
                    cda1[k] = clouds[iplon, k, 5]
                    cda2[k] = clouds[iplon, k, 6]
                    cda3[k] = clouds[iplon, k, 7]
                    cda4[k] = clouds[iplon, k, 8]
            else:
                for k in range(nlay):
                    cldfrc[k + 1] = clouds[iplon, k, 0]
                    cda1[k] = clouds[iplon, k, 1]

            cldfrc[0] = 1.0
            cldfrc[nlp1] = 0.0

            tem1 = 0.0
            tem2 = 0.0

            for k in range(nlay):
                tem1 += coldry[k] + colamt[k, 0]
                tem2 += colamt[k, 0]

            tem0 = 10.0 * tem2 / (self.amdw * tem1 * con_g)
            pwvcm = tem0 * plvl[iplon, 0]

            for k in range(nlay):
                summol = 0.0
                for i in range(1, maxgas):
                    summol += colamt[k, i]
                colbrd[k] = coldry[k] - summol

            tem1 = 1.80
            tem2 = 1.50
            for j in range(nbands):
                if j == 0 or j == 3 or j == 9:
                    secdiff[j] = 1.66
                else:
                    secdiff[j] = min(
                        tem1,
                        max(tem2, self.a0[j] + self.a1[j] * np.exp(self.a2[j] * pwvcm)),
                    )

            lcf1 = False
            for k in range(nlay):
                if cldfrc[k + 1] > self.eps:
                    lcf1 = True
                    break

            if verbose:
                print("Running cldprop . . .")
            if lcf1:
                cldfmc, taucld = self.cldprop(
                    cldfrc,
                    clwp,
                    relw,
                    ciwp,
                    reiw,
                    cda1,
                    cda2,
                    cda3,
                    cda4,
                    nlay,
                    nlp1,
                    ipseed[iplon],
                    dz,
                    delgth,
                    iplon,
                )
                if verbose:
                    print("Done")
                    print(" ")

                for k in range(nlay):
                    cldtau[iplon, k] = taucld[6, k]
            else:
                cldfmc = np.zeros((ngptlw, nlay))
                taucld = np.zeros((nbands, nlay))

            if verbose:
                print("Running setcoef . . .")
            (
                laytrop,
                pklay,
                pklev,
                jp,
                jt,
                jt1,
                rfrate,
                fac00,
                fac01,
                fac10,
                fac11,
                selffac,
                selffrac,
                indself,
                forfac,
                forfrac,
                indfor,
                minorfrac,
                scaleminor,
                scaleminorn2,
                indminor,
            ) = self.setcoef(
                pavel, tavel, tz, stemp, h2ovmr, colamt, coldry, colbrd, nlay, nlp1
            )

            if verbose:
                print("Done")
                print(" ")
                print("Running taumol . . .")
            fracs, tautot = self.taumol(
                laytrop,
                pavel,
                coldry,
                colamt,
                colbrd,
                wx,
                tauaer,
                rfrate,
                fac00,
                fac01,
                fac10,
                fac11,
                jp + 1,
                jt + 1,
                jt1 + 1,
                selffac,
                selffrac,
                indself,
                forfac,
                forfrac,
                indfor,
                minorfrac,
                scaleminor,
                scaleminorn2,
                indminor,
                nlay,
            )
            if verbose:
                print("Done")
                print(" ")
                print("Running rtrnmc . . .")
            if self.isubclw <= 0:
                if self.iovrlw <= 0:
                    (
                        totuflux,
                        totdflux,
                        htr,
                        totuclfl,
                        totdclfl,
                        htrcl,
                        htrb,
                    ) = self.rtrn(
                        semiss,
                        delp,
                        cldfrc,
                        taucld,
                        tautot,
                        pklay,
                        pklev,
                        fracs,
                        secdiff,
                        nlay,
                        nlp1,
                    )
                else:
                    (
                        totuflux,
                        totdflux,
                        htr,
                        totuclfl,
                        totdclfl,
                        htrcl,
                        htrb,
                    ) = self.rtrnmr(
                        semiss,
                        delp,
                        cldfrc,
                        taucld,
                        tautot,
                        pklay,
                        pklev,
                        fracs,
                        secdiff,
                        nlay,
                        nlp1,
                    )
            else:
                start = time.time()
                (
                    totuflux,
                    totdflux,
                    htr,
                    totuclfl,
                    totdclfl,
                    htrcl,
                    htrb,
                ) = self.rtrnmc(
                    semiss,
                    delp,
                    cldfmc,
                    taucld,
                    tautot,
                    pklay,
                    pklev,
                    fracs,
                    secdiff,
                    nlay,
                    nlp1,
                    iplon,
                )
                end = time.time()
                if verbose:
                    print(f"rtrnmc time = {end-start}")
            if verbose:
                print("Done")
                print(" ")

            upfxc_t[iplon] = totuflux[nlay]
            upfx0_t[iplon] = totuclfl[nlay]

            upfxc_s[iplon] = totuflux[0]
            upfx0_s[iplon] = totuclfl[0]
            dnfxc_s[iplon] = totdflux[0]
            dnfx0_s[iplon] = totdclfl[0]

            for k in range(nlay):
                hlwc[iplon, k] = htr[k]

            for k in range(nlay):
                hlw0[iplon, k] = htrcl[k]

            if verbose:
                print("Finished!")

        return (
            hlwc,
            upfxc_t,
            upfx0_t,
            upfxc_s,
            upfx0_s,
            dnfxc_s,
            dnfx0_s,
            cldtau,
            hlw0,
        )

    def setcoef(
        self, pavel, tavel, tz, stemp, h2ovmr, colamt, coldry, colbrd, nlay, nlp1
    ):

        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                       -size- !
        #   pavel     - real, layer pressures (mb)                         nlay !
        #   tavel     - real, layer temperatures (k)                       nlay !
        #   tz        - real, level (interface) temperatures (k)         0:nlay !
        #   stemp     - real, surface ground temperature (k)                1   !
        #   h2ovmr    - real, layer w.v. volum mixing ratio (kg/kg)        nlay !
        #   colamt    - real, column amounts of absorbing gases      nlay*maxgas!
        #                 2nd indices range: 1-maxgas, for watervapor,          !
        #                 carbon dioxide, ozone, nitrous oxide, methane,        !
        #                 oxigen, carbon monoxide,etc. (molecules/cm**2)        !
        #   coldry    - real, dry air column amount                        nlay !
        #   colbrd    - real, column amount of broadening gases            nlay !
        #   nlay/nlp1 - integer, total number of vertical layers, levels    1   !
        #                                                                       !
        #  outputs:                                                             !
        #   laytrop   - integer, tropopause layer index (unitless)          1   !
        #   pklay     - real, integrated planck func at lay temp   nbands*0:nlay!
        #   pklev     - real, integrated planck func at lev temp   nbands*0:nlay!
        #   jp        - real, indices of lower reference pressure          nlay !
        #   jt, jt1   - real, indices of lower reference temperatures      nlay !
        #   rfrate    - real, ref ratios of binary species param   nlay*nrates*2!
        #     (:,m,:)m=1-h2o/co2,2-h2o/o3,3-h2o/n2o,4-h2o/ch4,5-n2o/co2,6-o3/co2!
        #     (:,:,n)n=1,2: the rates of ref press at the 2 sides of the layer  !
        #   facij     - real, factors multiply the reference ks,           nlay !
        #                 i,j=0/1 for lower/higher of the 2 appropriate         !
        #                 temperatures and altitudes.                           !
        #   selffac   - real, scale factor for w. v. self-continuum        nlay !
        #                 equals (w. v. density)/(atmospheric density           !
        #                 at 296k and 1013 mb)                                  !
        #   selffrac  - real, factor for temperature interpolation of      nlay !
        #                 reference w. v. self-continuum data                   !
        #   indself   - integer, index of lower ref temp for selffac       nlay !
        #   forfac    - real, scale factor for w. v. foreign-continuum     nlay !
        #   forfrac   - real, factor for temperature interpolation of      nlay !
        #                 reference w.v. foreign-continuum data                 !
        #   indfor    - integer, index of lower ref temp for forfac        nlay !
        #   minorfrac - real, factor for minor gases                       nlay !
        #   scaleminor,scaleminorn2                                             !
        #             - real, scale factors for minor gases                nlay !
        #   indminor  - integer, index of lower ref temp for minor gases   nlay !
        #                                                                       !
        #  ======================    end of definitions    ===================  !

        # ===> ... begin here
        #
        #  --- ...  calculate information needed by the radiative transfer routine
        #           that is specific to this atmosphere, especially some of the
        #           coefficients and indices needed to compute the optical depths
        #           by interpolating data from stored reference atmospheres.

        dfile = os.path.join(LOOKUP_DIR, "totplnk.nc")
        pfile = os.path.join(LOOKUP_DIR, "radlw_ref_data.nc")
        totplnk = xr.open_dataset(dfile)["totplnk"].data
        preflog = xr.open_dataset(pfile)["preflog"].data
        tref = xr.open_dataset(pfile)["tref"].data
        chi_mls = xr.open_dataset(pfile)["chi_mls"].data

        pklay = np.zeros((nbands, nlp1))
        pklev = np.zeros((nbands, nlp1))

        jp = np.zeros(nlay, dtype=np.int32)
        jt = np.zeros(nlay, dtype=np.int32)
        jt1 = np.zeros(nlay, dtype=np.int32)
        fac00 = np.zeros(nlay)
        fac01 = np.zeros(nlay)
        fac10 = np.zeros(nlay)
        fac11 = np.zeros(nlay)
        forfac = np.zeros(nlay)
        forfrac = np.zeros(nlay)
        selffac = np.zeros(nlay)
        scaleminor = np.zeros(nlay)
        scaleminorn2 = np.zeros(nlay)
        indminor = np.zeros(nlay, dtype=np.int32)
        minorfrac = np.zeros(nlay)
        indfor = np.zeros(nlay, dtype=np.int32)
        indself = np.zeros(nlay, dtype=np.int32)
        selffrac = np.zeros(nlay)
        rfrate = np.zeros((nlay, nrates, 2))

        indlay = np.minimum(180, np.maximum(1, int(stemp - 159.0)))
        indlev = np.minimum(180, np.maximum(1, int(tz[0] - 159.0)))
        tlyrfr = stemp - int(stemp)
        tlvlfr = tz[0] - int(tz[0])

        for i in range(nbands):
            tem1 = totplnk[indlay, i] - totplnk[indlay - 1, i]
            tem2 = totplnk[indlev, i] - totplnk[indlev - 1, i]
            pklay[i, 0] = delwave[i] * (totplnk[indlay - 1, i] + tlyrfr * tem1)
            pklev[i, 0] = delwave[i] * (totplnk[indlev - 1, i] + tlvlfr * tem2)

        #  --- ...  begin layer loop
        #           calculate the integrated Planck functions for each band at the
        #           surface, level, and layer temperatures.

        laytrop = 0

        for k in range(nlay):
            indlay = np.minimum(180, np.maximum(1, int(tavel[k] - 159.0)))
            tlyrfr = tavel[k] - int(tavel[k])

            indlev = np.minimum(180, np.maximum(1, int(tz[k + 1] - 159.0)))
            tlvlfr = tz[k + 1] - int(tz[k + 1])

            #  --- ...  begin spectral band loop

            for i in range(nbands):
                pklay[i, k + 1] = delwave[i] * (
                    totplnk[indlay - 1, i]
                    + tlyrfr * (totplnk[indlay, i] - totplnk[indlay - 1, i])
                )
                pklev[i, k + 1] = delwave[i] * (
                    totplnk[indlev - 1, i]
                    + tlvlfr * (totplnk[indlev, i] - totplnk[indlev - 1, i])
                )

            #  --- ...  find the two reference pressures on either side of the
            #           layer pressure. store them in jp and jp1. store in fp the
            #           fraction of the difference (in ln(pressure)) between these
            #           two values that the layer pressure lies.

            plog = np.log(pavel[k])
            jp[k] = np.maximum(1, np.minimum(58, int(36.0 - 5.0 * (plog + 0.04)))) - 1
            jp1 = jp[k] + 1
            #  --- ...  limit pressure extrapolation at the top
            fp = np.maximum(0.0, np.minimum(1.0, 5.0 * (preflog[jp[k]] - plog)))

            #  --- ...  determine, for each reference pressure (jp and jp1), which
            #           reference temperature (these are different for each
            #           reference pressure) is nearest the layer temperature but does
            #           not exceed it. store these indices in jt and jt1, resp.
            #           store in ft (resp. ft1) the fraction of the way between jt
            #           (jt1) and the next highest reference temperature that the
            #           layer temperature falls.

            tem1 = (tavel[k] - tref[jp[k]]) / 15.0
            tem2 = (tavel[k] - tref[jp1]) / 15.0
            jt[k] = np.maximum(1, np.minimum(4, int(3.0 + tem1))) - 1
            jt1[k] = np.maximum(1, np.minimum(4, int(3.0 + tem2))) - 1
            #  --- ...  restrict extrapolation ranges by limiting abs(det t) < 37.5 deg
            ft = np.maximum(-0.5, np.minimum(1.5, tem1 - float(jt[k] - 2)))
            ft1 = np.maximum(-0.5, np.minimum(1.5, tem2 - float(jt1[k] - 2)))

            #  --- ...  we have now isolated the layer ln pressure and temperature,
            #           between two reference pressures and two reference temperatures
            #           (for each reference pressure).  we multiply the pressure
            #           fraction fp with the appropriate temperature fractions to get
            #           the factors that will be needed for the interpolation that yields
            #           the optical depths (performed in routines taugbn for band n)

            tem1 = 1.0 - fp
            fac10[k] = tem1 * ft
            fac00[k] = tem1 * (1.0 - ft)
            fac11[k] = fp * ft1
            fac01[k] = fp * (1.0 - ft1)

            forfac[k] = pavel[k] * self.stpfac / (tavel[k] * (1.0 + h2ovmr[k]))
            selffac[k] = h2ovmr[k] * forfac[k]

            #  --- ...  set up factors needed to separately include the minor gases
            #           in the calculation of absorption coefficient

            scaleminor[k] = pavel[k] / tavel[k]
            scaleminorn2[k] = (pavel[k] / tavel[k]) * (
                colbrd[k] / (coldry[k] + colamt[k, 0])
            )
            tem1 = (tavel[k] - 180.8) / 7.2
            indminor[k] = np.minimum(18, np.maximum(1, int(tem1)))
            minorfrac[k] = tem1 - float(indminor[k])

            #  --- ...  if the pressure is less than ~100mb, perform a different
            #           set of species interpolations.

            if plog > 4.56:
                laytrop = laytrop + 1

                tem1 = (332.0 - tavel[k]) / 36.0
                indfor[k] = np.minimum(2, np.maximum(1, int(tem1)))
                forfrac[k] = tem1 - float(indfor[k])

                #  --- ...  set up factors needed to separately include the water vapor
                #           self-continuum in the calculation of absorption coefficient.

                tem1 = (tavel[k] - 188.0) / 7.2
                indself[k] = np.minimum(9, np.maximum(1, int(tem1) - 7))
                selffrac[k] = tem1 - float(indself[k] + 7)

                #  --- ...  setup reference ratio to be used in calculation of binary
                #           species parameter in lower atmosphere.

                rfrate[k, 0, 0] = chi_mls[0, jp[k]] / chi_mls[1, jp[k]]
                rfrate[k, 0, 1] = chi_mls[0, jp[k] + 1] / chi_mls[1, jp[k] + 1]

                rfrate[k, 1, 0] = chi_mls[0, jp[k]] / chi_mls[2, jp[k]]
                rfrate[k, 1, 1] = chi_mls[0, jp[k] + 1] / chi_mls[2, jp[k] + 1]

                rfrate[k, 2, 0] = chi_mls[0, jp[k]] / chi_mls[3, jp[k]]
                rfrate[k, 2, 1] = chi_mls[0, jp[k] + 1] / chi_mls[3, jp[k] + 1]

                rfrate[k, 3, 0] = chi_mls[0, jp[k]] / chi_mls[5, jp[k]]
                rfrate[k, 3, 1] = chi_mls[0, jp[k] + 1] / chi_mls[5, jp[k] + 1]

                rfrate[k, 4, 0] = chi_mls[3, jp[k]] / chi_mls[1, jp[k]]
                rfrate[k, 4, 1] = chi_mls[3, jp[k] + 1] / chi_mls[1, jp[k] + 1]

            else:

                tem1 = (tavel[k] - 188.0) / 36.0
                indfor[k] = 3
                forfrac[k] = tem1 - 1.0

                indself[k] = 0
                selffrac[k] = 0.0

                #  --- ...  setup reference ratio to be used in calculation of binary
                #           species parameter in upper atmosphere.

                rfrate[k, 0, 0] = chi_mls[0, jp[k]] / chi_mls[1, jp[k]]
                rfrate[k, 0, 1] = chi_mls[0, jp[k] + 1] / chi_mls[1, jp[k] + 1]

                rfrate[k, 5, 0] = chi_mls[2, jp[k]] / chi_mls[1, jp[k]]
                rfrate[k, 5, 1] = chi_mls[2, jp[k] + 1] / chi_mls[1, jp[k] + 1]

            #  --- ...  rescale selffac and forfac for use in taumol

            selffac[k] = colamt[k, 0] * selffac[k]
            forfac[k] = colamt[k, 0] * forfac[k]

        return (
            laytrop,
            pklay,
            pklev,
            jp,
            jt,
            jt1,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
        )

    def rtrn(
        self,
        semiss,
        delp,
        cldfrc,
        taucld,
        tautot,
        pklay,
        pklev,
        fracs,
        secdif,
        nlay,
        nlp1,
    ):
        #  ===================  program usage description  ===================  !
        #                                                                       !
        # purpose:  compute the upward/downward radiative fluxes, and heating   !
        # rates for both clear or cloudy atmosphere.  clouds are assumed as     !
        # randomly overlaping in a vertical colum.                              !
        #                                                                       !
        # subprograms called:  none                                             !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                     -size-   !
        #   semiss  - real, lw surface emissivity                         nbands!
        #   delp    - real, layer pressure thickness (mb)                  nlay !
        #   cldfrc  - real, layer cloud fraction                         0:nlp1 !
        #   taucld  - real, layer cloud opt depth                    nbands,nlay!
        #   tautot  - real, total optical depth (gas+aerosols)       ngptlw,nlay!
        #   pklay   - real, integrated planck func at lay temp     nbands*0:nlay!
        #   pklev   - real, integrated planck func at lev temp     nbands*0:nlay!
        #   fracs   - real, planck fractions                         ngptlw,nlay!
        #   secdif  - real, secant of diffusivity angle                   nbands!
        #   nlay    - integer, number of vertical layers                    1   !
        #   nlp1    - integer, number of vertical levels (interfaces)       1   !
        #                                                                       !
        #  outputs:                                                             !
        #   totuflux- real, total sky upward flux (w/m2)                 0:nlay !
        #   totdflux- real, total sky downward flux (w/m2)               0:nlay !
        #   htr     - real, total sky heating rate (k/sec or k/day)        nlay !
        #   totuclfl- real, clear sky upward flux (w/m2)                 0:nlay !
        #   totdclfl- real, clear sky downward flux (w/m2)               0:nlay !
        #   htrcl   - real, clear sky heating rate (k/sec or k/day)        nlay !
        #   htrb    - real, spectral band lw heating rate (k/day)    nlay*nbands!
        #                                                                       !
        #  module veriables:                                                    !
        #   ngb     - integer, band index for each g-value                ngptlw!
        #   fluxfac - real, conversion factor for fluxes (pi*2.e4)           1  !
        #   heatfac - real, conversion factor for heating rates (g/cp*1e-2)  1  !
        #   tblint  - real, conversion factor for look-up tbl (float(ntbl)   1  !
        #   bpade   - real, pade approx constant (1/0.278)                   1  !
        #   wtdiff  - real, weight for radiance to flux conversion           1  !
        #   ntbl    - integer, dimension of look-up tables                   1  !
        #   tau_tbl - real, clr-sky opt dep lookup table                 0:ntbl !
        #   exp_tbl - real, transmittance lookup table                   0:ntbl !
        #   tfn_tbl - real, tau transition function                      0:ntbl !
        #                                                                       !
        #  local variables:                                                     !
        #    itgas  - integer, index for gases contribution look-up table    1  !
        #    ittot  - integer, index for gases plus clouds  look-up table    1  !
        #    reflct - real, surface reflectance                              1  !
        #    atrgas - real, gaseous absorptivity                             1  !
        #    atrtot - real, gaseous and cloud absorptivity                   1  !
        #    odcld  - real, cloud optical depth                              1  !
        #    efclrfr- real, effective clear sky fraction (1-efcldfr)       nlay !
        #    odepth - real, optical depth of gaseous only                    1  !
        #    odtot  - real, optical depth of gas and cloud                   1  !
        #    gasfac - real, gas-only pade factor, used for planck fn         1  !
        #    totfac - real, gas+cld pade factor, used for planck fn          1  !
        #    bbdgas - real, gas-only planck function for downward rt         1  !
        #    bbugas - real, gas-only planck function for upward rt           1  !
        #    bbdtot - real, gas and cloud planck function for downward rt    1  !
        #    bbutot - real, gas and cloud planck function for upward rt      1  !
        #    gassrcu- real, upwd source radiance due to gas only            nlay!
        #    totsrcu- real, upwd source radiance due to gas+cld             nlay!
        #    gassrcd- real, dnwd source radiance due to gas only             1  !
        #    totsrcd- real, dnwd source radiance due to gas+cld              1  !
        #    radtotu- real, spectrally summed total sky upwd radiance        1  !
        #    radclru- real, spectrally summed clear sky upwd radiance        1  !
        #    radtotd- real, spectrally summed total sky dnwd radiance        1  !
        #    radclrd- real, spectrally summed clear sky dnwd radiance        1  !
        #    toturad- real, total sky upward radiance by layer     0:nlay*nbands!
        #    clrurad- real, clear sky upward radiance by layer     0:nlay*nbands!
        #    totdrad- real, total sky downward radiance by layer   0:nlay*nbands!
        #    clrdrad- real, clear sky downward radiance by layer   0:nlay*nbands!
        #    fnet   - real, net longwave flux (w/m2)                     0:nlay !
        #    fnetc  - real, clear sky net longwave flux (w/m2)           0:nlay !
        #                                                                       !
        #                                                                       !
        #  *******************************************************************  !
        #  original code description                                            !
        #                                                                       !
        #  original version:   e. j. mlawer, et al. rrtm_v3.0                   !
        #  revision for gcms:  michael j. iacono; october, 2002                 !
        #  revision for f90:   michael j. iacono; june, 2006                    !
        #                                                                       !
        #  this program calculates the upward fluxes, downward fluxes, and      !
        #  heating rates for an arbitrary clear or cloudy atmosphere. the input !
        #  to this program is the atmospheric profile, all Planck function      !
        #  information, and the cloud fraction by layer.  a variable diffusivity!
        #  angle (secdif) is used for the angle integration. bands 2-3 and 5-9  !
        #  use a value for secdif that varies from 1.50 to 1.80 as a function   !
        #  of the column water vapor, and other bands use a value of 1.66.  the !
        #  gaussian weight appropriate to this angle (wtdiff=0.5) is applied    !
        #  here.  note that use of the emissivity angle for the flux integration!
        #  can cause errors of 1 to 4 W/m2 within cloudy layers.                !
        #  clouds are treated with a random cloud overlap method.               !
        #                                                                       !
        #  *******************************************************************  !
        #  ======================  end of description block  =================  !

        #  ---  outputs:
        htr = np.zeros(nlay)
        htrcl = np.zeros(nlay)

        htrb = np.zeros((nlay, nbands))

        totuflux = np.zeros(nlp1)
        totdflux = np.zeros(nlp1)
        totuclfl = np.zeros(nlp1)
        totdclfl = np.zeros(nlp1)

        #  ---  locals:
        rec_6 = 0.166667

        clrurad = np.zeros((nlp1, nbands))
        clrdrad = np.zeros((nlp1, nbands))
        toturad = np.zeros((nlp1, nbands))
        totdrad = np.zeros((nlp1, nbands))

        gassrcu = np.zeros(nlay)
        totsrcu = np.zeros(nlay)
        trngas = np.zeros(nlay)
        efclrfr = np.zeros(nlay)
        rfdelp = np.zeros(nlay)

        fnet = np.zeros(nlp1)
        fnetc = np.zeros(nlp1)

        #
        # ===> ...  begin here
        #

        #  --- ...  loop over all g-points

        for ig in range(ngptlw):
            ib = ngb(ig)

            radtotd = 0.0
            radclrd = 0.0

            # > -# Downward radiative transfer loop.

            for k in range(nlay - 1, -1, -1):

                #  - clear sky, gases contribution

                odepth = max(0.0, secdif[ib] * tautot[ig, k])
                if odepth <= 0.06:
                    atrgas = odepth - 0.5 * odepth * odepth
                    trng = 1.0 - atrgas
                    gasfac = rec_6 * odepth
                else:
                    tblind = odepth / (self.bpade + odepth)
                    itgas = self.tblint * tblind + 0.5
                    trng = self.exp_tbl[itgas]
                    atrgas = 1.0 - trng
                    gasfac = self.tfn_tbl[itgas]
                    odepth = self.tau_tbl[itgas]

                plfrac = fracs[ig, k]
                blay = pklay[ib, k]

                dplnku = pklev[ib, k] - blay
                dplnkd = pklev[ib, k - 1] - blay
                bbdgas = plfrac * (blay + dplnkd * gasfac)
                bbugas = plfrac * (blay + dplnku * gasfac)
                gassrcd = bbdgas * atrgas
                gassrcu[k] = bbugas * atrgas
                trngas[k] = trng

                # - total sky, gases+clouds contribution

                clfr = cldfrc[k]
                if clfr >= self.eps:
                    # \n  - cloudy layer

                    odcld = secdif[ib] * taucld[ib, k]
                    efclrfr[k] = 1.0 - (1.0 - np.exp(-odcld)) * clfr
                    odtot = odepth + odcld
                    if odtot < 0.06:
                        totfac = rec_6 * odtot
                        atrtot = odtot - 0.5 * odtot * odtot
                    else:
                        tblind = odtot / (self.bpade + odtot)
                        ittot = self.tblint * tblind + 0.5
                        totfac = self.tfn_tbl[ittot]
                        atrtot = 1.0 - self.exp_tbl[ittot]

                    bbdtot = plfrac * (blay + dplnkd * totfac)
                    bbutot = plfrac * (blay + dplnku * totfac)
                    totsrcd = bbdtot * atrtot
                    totsrcu[k] = bbutot * atrtot

                    #  --- ...  total sky radiance
                    radtotd = (
                        radtotd * trng * efclrfr[k]
                        + gassrcd
                        + clfr * (totsrcd - gassrcd)
                    )
                    totdrad[k - 1, ib] = totdrad[k - 1, ib] + radtotd

                    #  --- ...  clear sky radiance
                    radclrd = radclrd * trng + gassrcd
                    clrdrad[k - 1, ib] = clrdrad[k - 1, ib] + radclrd

                else:
                    #  --- ...  clear layer

                    #  --- ...  total sky radiance
                    radtotd = radtotd * trng + gassrcd
                    totdrad[k - 1, ib] = totdrad[k - 1, ib] + radtotd

                    #  --- ...  clear sky radiance
                    radclrd = radclrd * trng + gassrcd
                    clrdrad[k - 1, ib] = clrdrad[k - 1, ib] + radclrd

            # > -# Compute spectral emissivity & reflectance, include the
            #!    contribution of spectrally varying longwave emissivity and
            #!     reflection from the surface to the upward radiative transfer.

            #     note: spectral and Lambertian reflection are identical for the
            #           diffusivity angle flux integration used here.

            reflct = 1.0 - semiss[ib]
            rad0 = semiss[ib] * fracs[ig, 0] * pklay[ib, 0]

            # -# Compute total sky radiance.
            radtotu = rad0 + reflct * radtotd
            toturad[0, ib] = toturad[0, ib] + radtotu

            # -# Compute clear sky radiance
            radclru = rad0 + reflct * radclrd
            clrurad[0, ib] = clrurad[0, ib] + radclru

            # -# Upward radiative transfer loop.

            for k in range(nlay):
                clfr = cldfrc[k + 1]
                trng = trngas[k]
                gasu = gassrcu(k)

                if clfr >= self.eps:
                    #  --- ...  cloudy layer

                    #  --- ... total sky radiance
                    radtotu = (
                        radtotu * trng * efclrfr(k) + gasu + +clfr * (totsrcu(k) - gasu)
                    )
                    toturad[k, ib] = toturad[k, ib] + radtotu

                    #  --- ... clear sky radiance
                    radclru = radclru * trng + gasu
                    clrurad[k, ib] = clrurad[k, ib] + radclru

                else:
                    #  --- ...  clear layer

                    #  --- ... total sky radiance
                    radtotu = radtotu * trng + gasu
                    toturad[k, ib] = toturad[k, ib] + radtotu

                    #  --- ... clear sky radiance
                    radclru = radclru * trng + gasu
                    clrurad[k, ib] = clrurad[k, ib] + radclru

        # -    # Process longwave output from band for total and clear streams.
        #      Calculate upward, downward, and net flux.

        flxfac = self.wtdiff * self.fluxfac

        for k in range(nlp1):
            for ib in range(nbands):
                totuflux[k] = totuflux[k] + toturad[k, ib]
                totdflux[k] = totdflux[k] + totdrad[k, ib]
                totuclfl[k] = totuclfl[k] + clrurad[k, ib]
                totdclfl[k] = totdclfl[k] + clrdrad[k, ib]

            totuflux[k] = totuflux[k] * flxfac
            totdflux[k] = totdflux[k] * flxfac
            totuclfl[k] = totuclfl[k] * flxfac
            totdclfl[k] = totdclfl[k] * flxfac

        #  --- ...  calculate net fluxes and heating rates
        fnet[0] = totuflux[0] - totdflux[0]

        for k in range(nlay):
            rfdelp[k] = self.heatfac / delp[k]
            fnet[k] = totuflux[k] - totdflux[k]
            htr[k] = (fnet[k - 1] - fnet[k]) * rfdelp[k]

        # --- ...  optional clear sky heating rates
        if self.lhlw0:
            fnetc[0] = totuclfl[0] - totdclfl[0]

            for k in range(nlay):
                fnetc[k] = totuclfl[k] - totdclfl[k]
                htrcl[k] = (fnetc[k - 1] - fnetc[k]) * rfdelp[k]

        # --- ...  optional spectral band heating rates
        if self.lhlwb:
            for ib in range(nbands):
                fnet[0] = (toturad[0, ib] - totdrad[0, ib]) * flxfac

                for k in range(nlay):
                    fnet[k] = (toturad[k, ib] - totdrad[k, ib]) * flxfac
                    htrb[k, ib] = (fnet[k - 1] - fnet[k]) * rfdelp[k]

        return totuflux, totdflux, htr, totuclfl, totdclfl, htrcl, htrb

    def rtrnmr(
        self,
        semiss,
        delp,
        cldfrc,
        taucld,
        tautot,
        pklay,
        pklev,
        fracs,
        secdif,
        nlay,
        nlp1,
    ):
        #  ===================  program usage description  ===================  !
        #                                                                       !
        # purpose:  compute the upward/downward radiative fluxes, and heating   !
        # rates for both clear or cloudy atmosphere.  clouds are assumed as in  !
        # maximum-randomly overlaping in a vertical colum.                      !
        #                                                                       !
        # subprograms called:  none                                             !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                     -size-   !
        #   semiss  - real, lw surface emissivity                         nbands!
        #   delp    - real, layer pressure thickness (mb)                  nlay !
        #   cldfrc  - real, layer cloud fraction                         0:nlp1 !
        #   taucld  - real, layer cloud opt depth                    nbands,nlay!
        #   tautot  - real, total optical depth (gas+aerosols)       ngptlw,nlay!
        #   pklay   - real, integrated planck func at lay temp     nbands*0:nlay!
        #   pklev   - real, integrated planck func at lev temp     nbands*0:nlay!
        #   fracs   - real, planck fractions                         ngptlw,nlay!
        #   secdif  - real, secant of diffusivity angle                   nbands!
        #   nlay    - integer, number of vertical layers                    1   !
        #   nlp1    - integer, number of vertical levels (interfaces)       1   !
        #                                                                       !
        #  outputs:                                                             !
        #   totuflux- real, total sky upward flux (w/m2)                 0:nlay !
        #   totdflux- real, total sky downward flux (w/m2)               0:nlay !
        #   htr     - real, total sky heating rate (k/sec or k/day)        nlay !
        #   totuclfl- real, clear sky upward flux (w/m2)                 0:nlay !
        #   totdclfl- real, clear sky downward flux (w/m2)               0:nlay !
        #   htrcl   - real, clear sky heating rate (k/sec or k/day)        nlay !
        #   htrb    - real, spectral band lw heating rate (k/day)    nlay*nbands!
        #                                                                       !
        #  module veriables:                                                    !
        #   ngb     - integer, band index for each g-value                ngptlw!
        #   fluxfac - real, conversion factor for fluxes (pi*2.e4)           1  !
        #   heatfac - real, conversion factor for heating rates (g/cp*1e-2)  1  !
        #   tblint  - real, conversion factor for look-up tbl (float(ntbl)   1  !
        #   bpade   - real, pade approx constant (1/0.278)                   1  !
        #   wtdiff  - real, weight for radiance to flux conversion           1  !
        #   ntbl    - integer, dimension of look-up tables                   1  !
        #   tau_tbl - real, clr-sky opt dep lookup table                 0:ntbl !
        #   exp_tbl - real, transmittance lookup table                   0:ntbl !
        #   tfn_tbl - real, tau transition function                      0:ntbl !
        #                                                                       !
        #  local variables:                                                     !
        #    itgas  - integer, index for gases contribution look-up table    1  !
        #    ittot  - integer, index for gases plus clouds  look-up table    1  !
        #    reflct - real, surface reflectance                              1  !
        #    atrgas - real, gaseous absorptivity                             1  !
        #    atrtot - real, gaseous and cloud absorptivity                   1  !
        #    odcld  - real, cloud optical depth                              1  !
        #    odepth - real, optical depth of gaseous only                    1  !
        #    odtot  - real, optical depth of gas and cloud                   1  !
        #    gasfac - real, gas-only pade factor, used for planck fn         1  !
        #    totfac - real, gas+cld pade factor, used for planck fn          1  !
        #    bbdgas - real, gas-only planck function for downward rt         1  !
        #    bbugas - real, gas-only planck function for upward rt           1  !
        #    bbdtot - real, gas and cloud planck function for downward rt    1  !
        #    bbutot - real, gas and cloud planck function for upward rt      1  !
        #    gassrcu- real, upwd source radiance due to gas only            nlay!
        #    totsrcu- real, upwd source radiance due to gas + cld           nlay!
        #    gassrcd- real, dnwd source radiance due to gas only             1  !
        #    totsrcd- real, dnwd source radiance due to gas + cld            1  !
        #    radtotu- real, spectrally summed total sky upwd radiance        1  !
        #    radclru- real, spectrally summed clear sky upwd radiance        1  !
        #    radtotd- real, spectrally summed total sky dnwd radiance        1  !
        #    radclrd- real, spectrally summed clear sky dnwd radiance        1  !
        #    toturad- real, total sky upward radiance by layer     0:nlay*nbands!
        #    clrurad- real, clear sky upward radiance by layer     0:nlay*nbands!
        #    totdrad- real, total sky downward radiance by layer   0:nlay*nbands!
        #    clrdrad- real, clear sky downward radiance by layer   0:nlay*nbands!
        #    fnet   - real, net longwave flux (w/m2)                     0:nlay !
        #    fnetc  - real, clear sky net longwave flux (w/m2)           0:nlay !
        #                                                                       !
        #                                                                       !
        #  *******************************************************************  !
        #  original code description                                            !
        #                                                                       !
        #  original version:   e. j. mlawer, et al. rrtm_v3.0                   !
        #  revision for gcms:  michael j. iacono; october, 2002                 !
        #  revision for f90:   michael j. iacono; june, 2006                    !
        #                                                                       !
        #  this program calculates the upward fluxes, downward fluxes, and      !
        #  heating rates for an arbitrary clear or cloudy atmosphere. the input !
        #  to this program is the atmospheric profile, all Planck function      !
        #  information, and the cloud fraction by layer.  a variable diffusivity!
        #  angle (secdif) is used for the angle integration. bands 2-3 and 5-9  !
        #  use a value for secdif that varies from 1.50 to 1.80 as a function   !
        #  of the column water vapor, and other bands use a value of 1.66.  the !
        #  gaussian weight appropriate to this angle (wtdiff=0.5) is applied    !
        #  here.  note that use of the emissivity angle for the flux integration!
        #  can cause errors of 1 to 4 W/m2 within cloudy layers.                !
        #  clouds are treated with a maximum-random cloud overlap method.       !
        #                                                                       !
        #  *******************************************************************  !
        #  ======================  end of description block  =================  !   #

        #  ---  outputs:
        htr = np.zeros(nlay)
        htrcl = np.zeros(nlay)
        htrb = np.zeros((nlay, nbands))

        totuflux = np.zeros(nlp1)
        totdflux = np.zeros(nlp1)
        totuclfl = np.zeros(nlp1)
        totdclfl = np.zeros(nlp1)

        #  ---  locals:
        rec_6 = 0.166667

        clrurad = np.zeros((nlp1, nbands))
        clrdrad = np.zeros((nlp1, nbands))
        toturad = np.zeros((nlp1, nbands))
        totdrad = np.zeros((nlp1, nbands))

        gassrcu = np.zeros(nlay)
        totsrcu = np.zeros(nlay)
        trngas = np.zeros(nlay)
        trntot = np.zeros(nlay)
        rfdelp = np.zeros(nlay)

        fnet = np.zeros(nlp1)
        fnetc = np.zeros(nlp1)

        faccld1u = np.zeros(nlp1)
        faccld2u = np.zeros(nlp1)
        facclr1u = np.zeros(nlp1)
        facclr2u = np.zeros(nlp1)
        faccmb1u = np.zeros(nlp1)
        faccmb2u = np.zeros(nlp1)

        faccld1d = np.zeros(nlp1)
        faccld2d = np.zeros(nlp1)
        facclr1d = np.zeros(nlp1)
        facclr2d = np.zeros(nlp1)
        faccmb1d = np.zeros(nlp1)
        faccmb2d = np.zeros(nlp1)

        lstcldu = np.zeros(nlay)
        lstcldd = np.zeros(nlay)

        tblint = ntbl
        #
        # ===> ...  begin here
        #

        lstcldu[0] = cldfrc[0] > self.eps
        rat1 = 0.0
        rat2 = 0.0

        for k in range(nlay - 1):

            lstcldu[k + 1] = cldfrc[k + 1] > self.eps and cldfrc[k] <= self.eps

            if cldfrc[k] > self.eps:
                # Setup maximum/random cloud overlap.

                if cldfrc[k + 1] >= cldfrc[k]:
                    if lstcldu[k]:
                        if cldfrc[k] < 1.0:
                            facclr2u[k + 1] = (cldfrc[k + 1] - cldfrc[k]) / (
                                1.0 - cldfrc[k]
                            )
                        facclr2u[k] = 0.0
                        faccld2u[k] = 0.0
                    else:
                        fmax = max(cldfrc[k], cldfrc[k - 1])
                        if cldfrc[k + 1] > fmax:
                            facclr1u[k + 1] = rat2
                            facclr2u[k + 1] = (cldfrc[k + 1] - fmax) / (1.0 - fmax)
                        elif cldfrc[k + 1] < fmax:
                            facclr1u[k + 1] = (cldfrc[k + 1] - cldfrc[k]) / (
                                cldfrc[k - 1] - cldfrc[k]
                            )
                        else:
                            facclr1u[k + 1] = rat2

                    if facclr1u[k + 1] > 0.0 or facclr2u[k + 1] > 0.0:
                        rat1 = 1.0
                        rat2 = 0.0
                    else:
                        rat1 = 0.0
                        rat2 = 0.0
                else:
                    if lstcldu[k]:
                        faccld2u[k + 1] = (cldfrc[k] - cldfrc[k + 1]) / cldfrc[k]
                        facclr2u[k] = 0.0
                        faccld2u[k] = 0.0
                    else:
                        fmin = min(cldfrc[k], cldfrc[k - 1])
                        if cldfrc[k + 1] <= fmin:
                            faccld1u[k + 1] = rat1
                            faccld2u[k + 1] = (fmin - cldfrc[k + 1]) / fmin
                        else:
                            faccld1u[k + 1] = (cldfrc[k] - cldfrc[k + 1]) / (
                                cldfrc[k] - fmin
                            )

                    if faccld1u[k + 1] > 0.0 or faccld2u[k + 1] > 0.0:
                        rat1 = 0.0
                        rat2 = 1.0
                    else:
                        rat1 = 0.0
                        rat2 = 0.0

                faccmb1u[k + 1] = facclr1u[k + 1] * faccld2u[k] * cldfrc[k - 1]
                faccmb2u[k + 1] = faccld1u[k + 1] * facclr2u[k] * (1.0 - cldfrc[k - 1])

        for k in range(nlp1):
            faccld1d[k] = 0.0
            faccld2d[k] = 0.0
            facclr1d[k] = 0.0
            facclr2d[k] = 0.0
            faccmb1d[k] = 0.0
            faccmb2d[k] = 0.0

        lstcldd[nlay] = cldfrc[nlay] > self.eps
        rat1 = 0.0
        rat2 = 0.0

        for k in range(nlay - 1, 0, -1):
            lstcldd[k - 1] = cldfrc[k - 1] > self.eps and cldfrc[k] <= self.eps

            if cldfrc[k] > self.eps:
                if cldfrc[k - 1] >= cldfrc[k]:
                    if lstcldd[k]:
                        if cldfrc[k] < 1.0:
                            facclr2d[k - 1] = (cldfrc[k - 1] - cldfrc[k]) / (
                                1.0 - cldfrc[k]
                            )

                        facclr2d[k] = 0.0
                        faccld2d[k] = 0.0
                    else:
                        fmax = max(cldfrc[k], cldfrc[k + 1])

                        if cldfrc[k - 1] > fmax:
                            facclr1d[k - 1] = rat2
                            facclr2d[k - 1] = (cldfrc[k - 1] - fmax) / (1.0 - fmax)
                        elif cldfrc[k - 1] < fmax:
                            facclr1d[k - 1] = (cldfrc[k - 1] - cldfrc[k]) / (
                                cldfrc[k + 1] - cldfrc[k]
                            )
                        else:
                            facclr1d[k - 1] = rat2

                    if facclr1d[k - 1] > 0.0 or facclr2d[k - 1] > 0.0:
                        rat1 = 1.0
                        rat2 = 0.0
                    else:
                        rat1 = 0.0
                        rat2 = 0.0
                else:
                    if lstcldd[k]:
                        faccld2d[k - 1] = (cldfrc[k] - cldfrc[k - 1]) / cldfrc[k]
                        facclr2d[k] = 0.0
                        faccld2d[k] = 0.0
                    else:
                        fmin = min(cldfrc[k], cldfrc[k + 1])

                        if cldfrc[k - 1] <= fmin:
                            faccld1d[k - 1] = rat1
                            faccld2d[k - 1] = (fmin - cldfrc[k - 1]) / fmin
                        else:
                            faccld1d[k - 1] = (cldfrc[k] - cldfrc[k - 1]) / (
                                cldfrc[k] - fmin
                            )

                    if faccld1d[k - 1] > 0.0 or faccld2d[k - 1] > 0.0:
                        rat1 = 0.0
                        rat2 = 1.0
                    else:
                        rat1 = 0.0
                        rat2 = 0.0

                faccmb1d[k - 1] = facclr1d[k - 1] * faccld2d[k] * cldfrc[k + 1]
                faccmb2d[k - 1] = faccld1d[k - 1] * facclr2d[k] * (1.0 - cldfrc[k + 1])

        # Initialize for radiative transfer

        for ib in range(nbands):
            for k in range(nlp1):
                toturad[k, ib] = 0.0
                totdrad[k, ib] = 0.0
                clrurad[k, ib] = 0.0
                clrdrad[k, ib] = 0.0

            for k in range(nlp1):
                totuflux[k] = 0.0
                totdflux[k] = 0.0
                totuclfl[k] = 0.0
                totdclfl[k] = 0.0

            #  --- ...  loop over all g-points

            for ig in range(ngptlw):
                ib = ngb[ig] - 1

                radtotd = 0.0
                radclrd = 0.0

                # Downward radiative transfer loop:

                for k in range(nlay - 1, -1, -1):

                    #  --- ...  clear sky, gases contribution

                    odepth = max(0.0, secdif[ib] * tautot[ig, k])
                    if odepth <= 0.06:
                        atrgas = odepth - 0.5 * odepth * odepth
                        trng = 1.0 - atrgas
                        gasfac = rec_6 * odepth
                    else:
                        tblind = odepth / (self.bpade + odepth)
                        itgas = tblint * tblind + 0.5
                        trng = self.exp_tbl[itgas]
                        atrgas = 1.0 - trng
                        gasfac = self.tfn_tbl[itgas]
                        odepth = self.tau_tbl[itgas]

                    plfrac = fracs[ig, k]
                    blay = pklay[ib, k]

                    dplnku = pklev[ib, k] - blay
                    dplnkd = pklev[ib, k - 1] - blay
                    bbdgas = plfrac * (blay + dplnkd * gasfac)
                    bbugas = plfrac * (blay + dplnku * gasfac)
                    gassrcd = bbdgas * atrgas
                    gassrcu[k] = bbugas * atrgas
                    trngas[k] = trng

                    #  --- ...  total sky, gases+clouds contribution

                    clfr = cldfrc[k]
                    if lstcldd[k]:
                        totradd = clfr * radtotd
                        clrradd = radtotd - totradd
                        rad = 0.0

                    if clfr >= self.eps:
                        #  - cloudy layer

                        odcld = secdif[ib] * taucld[ib, k]
                        odtot = odepth + odcld
                        if odtot < 0.06:
                            totfac = rec_6 * odtot
                            atrtot = odtot - 0.5 * odtot * odtot
                            trnt = 1.0 - atrtot
                        else:
                            tblind = odtot / (self.bpade + odtot)
                            ittot = tblint * tblind + 0.5
                            totfac = self.tfn_tbl[ittot]
                            trnt = self.exp_tbl[ittot]
                            atrtot = 1.0 - trnt

                        bbdtot = plfrac * (blay + dplnkd * totfac)
                        bbutot = plfrac * (blay + dplnku * totfac)
                        totsrcd = bbdtot * atrtot
                        totsrcu[k] = bbutot * atrtot
                        trntot[k] = trnt

                        totradd = totradd * trnt + clfr * totsrcd
                        clrradd = clrradd * trng + (1.0 - clfr) * gassrcd

                        #  - total sky radiance
                        radtotd = totradd + clrradd
                        totdrad[k - 1, ib] = totdrad[k - 1, ib] + radtotd

                        #  - clear sky radiance
                        radclrd = radclrd * trng + gassrcd
                        clrdrad[k - 1, ib] = clrdrad[k - 1, ib] + radclrd

                        radmod = (
                            rad * (facclr1d[k - 1] * trng + faccld1d[k - 1] * trnt)
                            - faccmb1d[k - 1] * gassrcd
                            + faccmb2d[k - 1] * totsrcd
                        )

                        rad = (
                            -radmod
                            + facclr2d[k - 1] * (clrradd + radmod)
                            - faccld2d[k - 1] * (totradd - radmod)
                        )
                        totradd = totradd + rad
                        clrradd = clrradd - rad

                    else:
                        #  --- ...  clear layer

                        #  --- ...  total sky radiance
                        radtotd = radtotd * trng + gassrcd
                        totdrad[k - 1, ib] = totdrad[k - 1, ib] + radtotd

                        #  --- ...  clear sky radiance
                        radclrd = radclrd * trng + gassrcd
                        clrdrad[k - 1, ib] = clrdrad[k - 1, ib] + radclrd

                # Compute spectral emissivity & reflectance, include the
                #    contribution of spectrally varying longwave emissivity and
                #    reflection from the surface to the upward radiative transfer.

                #    note: spectral and Lambertian reflection are identical for the
                #          diffusivity angle flux integration used here.

                reflct = 1.0 - semiss[ib]
                rad0 = semiss[ib] * fracs[ig, 0] * pklay[ib, 0]

                # -# Compute total sky radiance.
                radtotu = rad0 + reflct * radtotd
                toturad[0, ib] = toturad[0, ib] + radtotu

                # Compute clear sky radiance.
                radclru = rad0 + reflct * radclrd
                clrurad[0, ib] = clrurad[0, ib] + radclru

                # Upward radiative transfer loop:
                for k in range(nlay):
                    clfr = cldfrc[k]
                    trng = trngas[k]
                    gasu = gassrcu[k]

                    if lstcldu[k]:
                        totradu = clfr * radtotu
                        clrradu = radtotu - totradu
                        rad = 0.0

                    if clfr >= self.eps:
                        #  - cloudy layer radiance
                        trnt = trntot[k]
                        totu = totsrcu[k]
                        totradu = totradu * trnt + clfr * totu
                        clrradu = clrradu * trng + (1.0 - clfr) * gasu

                        #  - total sky radiance
                        radtotu = totradu + clrradu
                        toturad[k, ib] = toturad[k, ib] + radtotu

                        #  - clear sky radiance
                        radclru = radclru * trng + gasu
                        clrurad[k, ib] = clrurad[k, ib] + radclru

                        radmod = (
                            rad * (facclr1u[k + 1] * trng + faccld1u[k + 1] * trnt)
                            - faccmb1u[k + 1] * gasu
                            + faccmb2u[k + 1] * totu
                        )
                        rad = (
                            -radmod
                            + facclr2u[k + 1] * (clrradu + radmod)
                            - faccld2u[k + 1] * (totradu - radmod)
                        )
                        totradu += rad
                        clrradu -= rad
                    else:
                        #  --- ...  clear layer

                        #  --- ...  total sky radiance
                        radtotu = radtotu * trng + gasu
                        toturad[k, ib] = toturad[k, ib] + radtotu

                        #  --- ...  clear sky radiance
                        radclru = radclru * trng + gasu
                        clrurad[k, ib] = clrurad[k, ib] + radclru

            # -# Process longwave output from band for total and clear streams.
            # calculate upward, downward, and net flux.

            flxfac = self.wtdiff * self.fluxfac

            for k in range(nlp1):
                for ib in range(nbands):
                    totuflux[k] = totuflux[k] + toturad[k, ib]
                    totdflux[k] = totdflux[k] + totdrad[k, ib]
                    totuclfl[k] = totuclfl[k] + clrurad[k, ib]
                    totdclfl[k] = totdclfl[k] + clrdrad[k, ib]

                totuflux[k] = totuflux[k] * flxfac
                totdflux[k] = totdflux[k] * flxfac
                totuclfl[k] = totuclfl[k] * flxfac
                totdclfl[k] = totdclfl[k] * flxfac

            #  --- ...  calculate net fluxes and heating rates
            fnet[0] = totuflux[0] - totdflux[0]

            for k in range(nlay):
                rfdelp[k] = self.heatfac / delp[k]
                fnet[k] = totuflux[k] - totdflux[k]
                htr[k] = (fnet[k - 1] - fnet[k]) * rfdelp[k]

            # --- ...  optional clear sky heating rates
            if self.lhlw0:
                fnetc[0] = totuclfl[0] - totdclfl[0]

                for k in range(nlay):
                    fnetc[k] = totuclfl[k] - totdclfl[k]
                    htrcl[k] = (fnetc[k - 1] - fnetc[k]) * rfdelp[k]

            # --- ...  optional spectral band heating rates
            if self.lhlwb:
                for ib in range(nbands):
                    fnet[0] = (toturad[0, ib] - totdrad[0, ib]) * flxfac

                    for k in range(nlay):
                        fnet[k] = (toturad[k, ib] - totdrad[k, ib]) * flxfac
                        htrb[k, ib] = (fnet[k - 1] - fnet[k]) * rfdelp[k]

        return totuflux, totdflux, htr, totuclfl, totdclfl, htrcl, htrb

    def rtrnmc(
        self,
        semiss,
        delp,
        cldfmc,
        taucld,
        tautot,
        pklay,
        pklev,
        fracs,
        secdif,
        nlay,
        nlp1,
        iplon,
    ):
        #  ===================  program usage description  ===================  !
        # purpose:  compute the upward/downward radiative fluxes, and heating   !
        # rates for both clear or cloudy atmosphere.  clouds are treated with   !
        # the mcica stochastic approach.                                        !
        #                                                                       !
        # subprograms called:  none                                             !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                     -size-   !
        #   semiss  - real, lw surface emissivity                         nbands!
        #   delp    - real, layer pressure thickness (mb)                  nlay !
        #   cldfmc  - real, layer cloud fraction (sub-column)        ngptlw*nlay!
        #   taucld  - real, layer cloud opt depth                    nbands*nlay!
        #   tautot  - real, total optical depth (gas+aerosols)       ngptlw*nlay!
        #   pklay   - real, integrated planck func at lay temp     nbands*0:nlay!
        #   pklev   - real, integrated planck func at lev temp     nbands*0:nlay!
        #   fracs   - real, planck fractions                         ngptlw*nlay!
        #   secdif  - real, secant of diffusivity angle                   nbands!
        #   nlay    - integer, number of vertical layers                    1   !
        #   nlp1    - integer, number of vertical levels (interfaces)       1   !
        #                                                                       !
        #  outputs:                                                             !
        #   totuflux- real, total sky upward flux (w/m2)                 0:nlay !
        #   totdflux- real, total sky downward flux (w/m2)               0:nlay !
        #   htr     - real, total sky heating rate (k/sec or k/day)        nlay !
        #   totuclfl- real, clear sky upward flux (w/m2)                 0:nlay !
        #   totdclfl- real, clear sky downward flux (w/m2)               0:nlay !
        #   htrcl   - real, clear sky heating rate (k/sec or k/day)        nlay !
        #   htrb    - real, spectral band lw heating rate (k/day)    nlay*nbands!
        #                                                                       !
        #  module veriables:                                                    !
        #   ngb     - integer, band index for each g-value                ngptlw!
        #   fluxfac - real, conversion factor for fluxes (pi*2.e4)           1  !
        #   heatfac - real, conversion factor for heating rates (g/cp*1e-2)  1  !
        #   tblint  - real, conversion factor for look-up tbl (float(ntbl)   1  !
        #   bpade   - real, pade approx constant (1/0.278)                   1  !
        #   wtdiff  - real, weight for radiance to flux conversion           1  !
        #   ntbl    - integer, dimension of look-up tables                   1  !
        #   tau_tbl - real, clr-sky opt dep lookup table                 0:ntbl !
        #   exp_tbl - real, transmittance lookup table                   0:ntbl !
        #   tfn_tbl - real, tau transition function                      0:ntbl !
        #                                                                       !
        #  local variables:                                                     !
        #    itgas  - integer, index for gases contribution look-up table    1  !
        #    ittot  - integer, index for gases plus clouds  look-up table    1  !
        #    reflct - real, surface reflectance                              1  !
        #    atrgas - real, gaseous absorptivity                             1  !
        #    atrtot - real, gaseous and cloud absorptivity                   1  !
        #    odcld  - real, cloud optical depth                              1  !
        #    efclrfr- real, effective clear sky fraction (1-efcldfr)        nlay!
        #    odepth - real, optical depth of gaseous only                    1  !
        #    odtot  - real, optical depth of gas and cloud                   1  !
        #    gasfac - real, gas-only pade factor, used for planck function   1  !
        #    totfac - real, gas and cloud pade factor, used for planck fn    1  !
        #    bbdgas - real, gas-only planck function for downward rt         1  !
        #    bbugas - real, gas-only planck function for upward rt           1  !
        #    bbdtot - real, gas and cloud planck function for downward rt    1  !
        #    bbutot - real, gas and cloud planck function for upward rt      1  !
        #    gassrcu- real, upwd source radiance due to gas                 nlay!
        #    totsrcu- real, upwd source radiance due to gas+cld             nlay!
        #    gassrcd- real, dnwd source radiance due to gas                  1  !
        #    totsrcd- real, dnwd source radiance due to gas+cld              1  !
        #    radtotu- real, spectrally summed total sky upwd radiance        1  !
        #    radclru- real, spectrally summed clear sky upwd radiance        1  !
        #    radtotd- real, spectrally summed total sky dnwd radiance        1  !
        #    radclrd- real, spectrally summed clear sky dnwd radiance        1  !
        #    toturad- real, total sky upward radiance by layer     0:nlay*nbands!
        #    clrurad- real, clear sky upward radiance by layer     0:nlay*nbands!
        #    totdrad- real, total sky downward radiance by layer   0:nlay*nbands!
        #    clrdrad- real, clear sky downward radiance by layer   0:nlay*nbands!
        #    fnet   - real, net longwave flux (w/m2)                     0:nlay !
        #    fnetc  - real, clear sky net longwave flux (w/m2)           0:nlay !
        #                                                                       !
        #                                                                       !
        #  *******************************************************************  !
        #  original code description                                            !
        #                                                                       !
        #  original version:   e. j. mlawer, et al. rrtm_v3.0                   !
        #  revision for gcms:  michael j. iacono; october, 2002                 !
        #  revision for f90:   michael j. iacono; june, 2006                    !
        #                                                                       !
        #  this program calculates the upward fluxes, downward fluxes, and      !
        #  heating rates for an arbitrary clear or cloudy atmosphere. the input !
        #  to this program is the atmospheric profile, all Planck function      !
        #  information, and the cloud fraction by layer.  a variable diffusivity!
        #  angle (secdif) is used for the angle integration. bands 2-3 and 5-9  !
        #  use a value for secdif that varies from 1.50 to 1.80 as a function   !
        #  of the column water vapor, and other bands use a value of 1.66.  the !
        #  gaussian weight appropriate to this angle (wtdiff=0.5) is applied    !
        #  here.  note that use of the emissivity angle for the flux integration!
        #  can cause errors of 1 to 4 W/m2 within cloudy layers.                !
        #  clouds are treated with the mcica stochastic approach and            !
        #  maximum-random cloud overlap.                                        !
        #                                                                       !
        #  *******************************************************************  !
        #  ======================  end of description block  =================  !

        #  ---  outputs:
        htr = np.zeros(nlay)
        htrcl = np.zeros(nlay)
        htrb = np.zeros((nlay, nbands))

        totuflux = np.zeros(nlp1)
        totdflux = np.zeros(nlp1)
        totuclfl = np.zeros(nlp1)
        totdclfl = np.zeros(nlp1)

        #  ---  locals:
        rec_6 = 0.166667

        clrurad = np.zeros((nlp1, nbands))
        clrdrad = np.zeros((nlp1, nbands))
        toturad = np.zeros((nlp1, nbands))
        totdrad = np.zeros((nlp1, nbands))

        gassrcu = np.zeros(nlay)
        totsrcu = np.zeros(nlay)
        trngas = np.zeros(nlay)
        efclrfr = np.zeros(nlay)
        rfdelp = np.zeros(nlay)

        fnet = np.zeros(nlp1)
        fnetc = np.zeros(nlp1)

        tblint = ntbl

        #
        # ===> ...  begin here
        #

        #  --- ...  loop over all g-points

        for ig in range(ngptlw):
            ib = ngb[ig] - 1

            radtotd = 0.0
            radclrd = 0.0

            # Downward radiative transfer loop.
            # - Clear sky, gases contribution
            # - Total sky, gases+clouds contribution
            # - Cloudy layer
            # - Total sky radiance
            # - Clear sky radiance

            for k in range(nlay - 1, -1, -1):

                #  --- ...  clear sky, gases contribution

                odepth = max(0.0, secdif[ib] * tautot[ig, k])
                if odepth <= 0.06:
                    atrgas = odepth - 0.5 * odepth * odepth
                    trng = 1.0 - atrgas
                    gasfac = rec_6 * odepth
                else:
                    tblind = odepth / (self.bpade + odepth)
                    itgas = int(tblint * tblind + 0.5)
                    trng = self.exp_tbl[itgas]
                    atrgas = 1.0 - trng
                    gasfac = self.tfn_tbl[itgas]
                    odepth = self.tau_tbl[itgas]

                plfrac = fracs[ig, k]
                blay = pklay[ib, k + 1]

                dplnku = pklev[ib, k + 1] - blay
                dplnkd = pklev[ib, k] - blay
                bbdgas = plfrac * (blay + dplnkd * gasfac)
                bbugas = plfrac * (blay + dplnku * gasfac)
                gassrcd = bbdgas * atrgas
                gassrcu[k] = bbugas * atrgas
                trngas[k] = trng

                #  --- ...  total sky, gases+clouds contribution
                clfm = cldfmc[ig, k]
                if clfm >= self.eps:
                    #  --- ...  cloudy layer
                    odcld = secdif[ib] * taucld[ib, k]
                    efclrfr[k] = 1.0 - (1.0 - np.exp(-odcld)) * clfm
                    odtot = odepth + odcld
                    if odtot < 0.06:
                        totfac = rec_6 * odtot
                        atrtot = odtot - 0.5 * odtot * odtot
                    else:
                        tblind = odtot / (self.bpade + odtot)
                        ittot = int(tblint * tblind + 0.5)
                        totfac = self.tfn_tbl[ittot]
                        atrtot = 1.0 - self.exp_tbl[ittot]

                    bbdtot = plfrac * (blay + dplnkd * totfac)
                    bbutot = plfrac * (blay + dplnku * totfac)
                    totsrcd = bbdtot * atrtot
                    totsrcu[k] = bbutot * atrtot

                    #  --- ...  total sky radiance
                    radtotd = (
                        radtotd * trng * efclrfr[k]
                        + gassrcd
                        + clfm * (totsrcd - gassrcd)
                    )
                    totdrad[k, ib] = totdrad[k, ib] + radtotd

                    #  --- ...  clear sky radiance
                    radclrd = radclrd * trng + gassrcd
                    clrdrad[k, ib] = clrdrad[k, ib] + radclrd

                else:
                    #  --- ...  clear layer

                    #  --- ...  total sky radiance
                    radtotd = radtotd * trng + gassrcd
                    totdrad[k, ib] = totdrad[k, ib] + radtotd

                    #  --- ...  clear sky radiance
                    radclrd = radclrd * trng + gassrcd
                    clrdrad[k, ib] = clrdrad[k, ib] + radclrd

            #    Compute spectral emissivity & reflectance, include the
            #    contribution of spectrally varying longwave emissivity and
            #    reflection from the surface to the upward radiative transfer.

            #     note: spectral and Lambertian reflection are identical for the
            #           diffusivity angle flux integration used here.

            reflct = 1.0 - semiss[ib]
            rad0 = semiss[ib] * fracs[ig, 0] * pklay[ib, 0]

            # Compute total sky radiance
            radtotu = rad0 + reflct * radtotd
            toturad[0, ib] = toturad[0, ib] + radtotu

            # Compute clear sky radiance
            radclru = rad0 + reflct * radclrd
            clrurad[0, ib] = clrurad[0, ib] + radclru

            # Upward radiative transfer loop
            # - Compute total sky radiance
            # - Compute clear sky radiance

            # toturad holds summed radiance for total sky stream
            # clrurad holds summed radiance for clear sky stream

            for k in range(nlay):
                clfm = cldfmc[ig, k]
                trng = trngas[k]
                gasu = gassrcu[k]

                if clfm > self.eps:
                    #  --- ...  cloudy layer

                    #  --- ... total sky radiance
                    radtotu = (
                        radtotu * trng * efclrfr[k] + gasu + clfm * (totsrcu[k] - gasu)
                    )
                    toturad[k + 1, ib] = toturad[k + 1, ib] + radtotu

                    #  --- ... clear sky radiance
                    radclru = radclru * trng + gasu
                    clrurad[k + 1, ib] = clrurad[k + 1, ib] + radclru

                else:
                    #  --- ...  clear layer

                    #  --- ... total sky radiance
                    radtotu = radtotu * trng + gasu
                    toturad[k + 1, ib] = toturad[k + 1, ib] + radtotu

                    #  --- ... clear sky radiance
                    radclru = radclru * trng + gasu
                    clrurad[k + 1, ib] = clrurad[k + 1, ib] + radclru

        # Process longwave output from band for total and clear streams.
        # Calculate upward, downward, and net flux.

        flxfac = self.wtdiff * self.fluxfac

        for k in range(nlp1):
            for ib in range(nbands):
                totuflux[k] = totuflux[k] + toturad[k, ib]
                totdflux[k] = totdflux[k] + totdrad[k, ib]
                totuclfl[k] = totuclfl[k] + clrurad[k, ib]
                totdclfl[k] = totdclfl[k] + clrdrad[k, ib]

            totuflux[k] = totuflux[k] * flxfac
            totdflux[k] = totdflux[k] * flxfac
            totuclfl[k] = totuclfl[k] * flxfac
            totdclfl[k] = totdclfl[k] * flxfac

        #  --- ...  calculate net fluxes and heating rates
        fnet[0] = totuflux[0] - totdflux[0]

        for k in range(nlay):
            rfdelp[k] = self.heatfac / delp[k]
            fnet[k + 1] = totuflux[k + 1] - totdflux[k + 1]
            htr[k] = (fnet[k] - fnet[k + 1]) * rfdelp[k]

        # --- ...  optional clear sky heating rates
        if self.lhlw0:
            fnetc[0] = totuclfl[0] - totdclfl[0]

            for k in range(nlay):
                fnetc[k + 1] = totuclfl[k + 1] - totdclfl[k + 1]
                htrcl[k] = (fnetc[k] - fnetc[k + 1]) * rfdelp[k]

        # --- ...  optional spectral band heating rates
        if self.lhlwb:
            for ib in range(nbands):
                fnet[0] = (toturad[0, ib] - totdrad[0, ib]) * flxfac

                for k in range(nlay):
                    fnet[k + 1] = (toturad[k + 1, ib] - totdrad[k + 1, ib]) * flxfac
                    htrb[k, ib] = (fnet[k] - fnet[k + 1]) * rfdelp[k]

        return totuflux, totdflux, htr, totuclfl, totdclfl, htrcl, htrb

    def cldprop(
        self,
        cfrac,
        cliqp,
        reliq,
        cicep,
        reice,
        cdat1,
        cdat2,
        cdat3,
        cdat4,
        nlay,
        nlp1,
        ipseed,
        dz,
        de_lgth,
        iplon,
    ):
        #  ===================  program usage description  ===================  !
        #                                                                       !
        # purpose:  compute the cloud optical depth(s) for each cloudy layer    !
        # and g-point interval.                                                 !
        #                                                                       !
        # subprograms called:  none                                             !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                       -size- !
        #    cfrac - real, layer cloud fraction                          0:nlp1 !
        #        .....  for ilwcliq > 0  (prognostic cloud sckeme)  - - -       !
        #    cliqp - real, layer in-cloud liq water path (g/m**2)          nlay !
        #    reliq - real, mean eff radius for liq cloud (micron)          nlay !
        #    cicep - real, layer in-cloud ice water path (g/m**2)          nlay !
        #    reice - real, mean eff radius for ice cloud (micron)          nlay !
        #    cdat1 - real, layer rain drop water path  (g/m**2)            nlay !
        #    cdat2 - real, effective radius for rain drop (microm)         nlay !
        #    cdat3 - real, layer snow flake water path (g/m**2)            nlay !
        #    cdat4 - real, effective radius for snow flakes (micron)       nlay !
        #        .....  for ilwcliq = 0  (diagnostic cloud sckeme)  - - -       !
        #    cdat1 - real, input cloud optical depth                       nlay !
        #    cdat2 - real, layer cloud single scattering albedo            nlay !
        #    cdat3 - real, layer cloud asymmetry factor                    nlay !
        #    cdat4 - real, optional use                                    nlay !
        #    cliqp - not used                                              nlay !
        #    reliq - not used                                              nlay !
        #    cicep - not used                                              nlay !
        #    reice - not used                                              nlay !
        #                                                                       !
        #    dz     - real, layer thickness (km)                           nlay !
        #    de_lgth- real, layer cloud decorrelation length (km)             1 !
        #    nlay  - integer, number of vertical layers                      1  !
        #    nlp1  - integer, number of vertical levels                      1  !
        #    ipseed- permutation seed for generating random numbers (isubclw>0) !
        #                                                                       !
        #  outputs:                                                             !
        #    cldfmc - real, cloud fraction for each sub-column       ngptlw*nlay!
        #    taucld - real, cld opt depth for bands (non-mcica)      nbands*nlay!
        #                                                                       !
        #  explanation of the method for each value of ilwcliq, and ilwcice.    !
        #    set up in module "module_radlw_cntr_para"                          !
        #                                                                       !
        #     ilwcliq=0  : input cloud optical property (tau, ssa, asy).        !
        #                  (used for diagnostic cloud method)                   !
        #     ilwcliq>0  : input cloud liq/ice path and effective radius, also  !
        #                  require the user of 'ilwcice' to specify the method  !
        #                  used to compute aborption due to water/ice parts.    !
        #  ...................................................................  !
        #                                                                       !
        #     ilwcliq=1:   the water droplet effective radius (microns) is input!
        #                  and the opt depths due to water clouds are computed  !
        #                  as in hu and stamnes, j., clim., 6, 728-742, (1993). !
        #                  the values for absorption coefficients appropriate for
        #                  the spectral bands in rrtm have been obtained for a  !
        #                  range of effective radii by an averaging procedure   !
        #                  based on the work of j. pinto (private communication).
        #                  linear interpolation is used to get the absorption   !
        #                  coefficients for the input effective radius.         !
        #                                                                       !
        #     ilwcice=1:   the cloud ice path (g/m2) and ice effective radius   !
        #                  (microns) are input and the optical depths due to ice!
        #                  clouds are computed as in ebert and curry, jgr, 97,  !
        #                  3831-3836 (1992).  the spectral regions in this work !
        #                  have been matched with the spectral bands in rrtm to !
        #                  as great an extent as possible:                      !
        #                     e&c 1      ib = 5      rrtm bands 9-16            !
        #                     e&c 2      ib = 4      rrtm bands 6-8             !
        #                     e&c 3      ib = 3      rrtm bands 3-5             !
        #                     e&c 4      ib = 2      rrtm band 2                !
        #                     e&c 5      ib = 1      rrtm band 1                !
        #     ilwcice=2:   the cloud ice path (g/m2) and ice effective radius   !
        #                  (microns) are input and the optical depths due to ice!
        #                  clouds are computed as in rt code, streamer v3.0     !
        #                  (ref: key j., streamer user's guide, cooperative     !
        #                  institute for meteorological satellite studies, 2001,!
        #                  96 pp.) valid range of values for re are between 5.0 !
        #                  and 131.0 micron.                                    !
        #     ilwcice=3:   the ice generalized effective size (dge) is input and!
        #                  the optical properties, are calculated as in q. fu,  !
        #                  j. climate, (1998). q. fu provided high resolution   !
        #                  tales which were appropriately averaged for the bands!
        #                  in rrtm_lw. linear interpolation is used to get the  !
        #                  coeff from the stored tables. valid range of values  !
        #                  for deg are between 5.0 and 140.0 micron.            !
        #                                                                       !
        #  other cloud control module variables:                                !
        #     isubclw =0: standard cloud scheme, no sub-col cloud approximation !
        #             >0: mcica sub-col cloud scheme using ipseed as permutation!
        #                 seed for generating rundom numbers                    !
        #                                                                       !
        #  ======================  end of description block  =================  !
        #

        #
        # ===> ...  begin here
        #
        cldmin = 1.0e-80

        taucld = np.zeros((nbands, nlay))
        tauice = np.zeros(nbands)
        tauliq = np.zeros(nbands)
        cldfmc = np.zeros((ngptlw, nlay))
        cldf = np.zeros(nlay)

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_cldprlw_data.nc"))
        absliq1 = ds["absliq1"]
        absice0 = ds["absice0"]
        absice1 = ds["absice1"]
        absice2 = ds["absice2"]
        absice3 = ds["absice3"]

        # Compute cloud radiative properties for a cloudy column:
        # - Compute cloud radiative properties for rain and snow (tauran,tausnw)
        # - Calculation of absorption coefficients due to water clouds(tauliq)
        # - Calculation of absorption coefficients due to ice clouds (tauice).
        # - For prognostic cloud scheme: sum up the cloud optical property:
        #   \f$ taucld=tauice+tauliq+tauran+tausnw \f$

        #  --- ...  compute cloud radiative properties for a cloudy column

        if ilwcliq > 0:
            for k in range(nlay):
                if cfrac[k + 1] > cldmin:
                    tauran = absrain * cdat1[k]  # ncar formula

                    #  ---  if use fu's formula it needs to be normalized by snow density
                    #       !not use snow density = 0.1 g/cm**3 = 0.1 g/(mu * m**2)
                    #       use ice density = 0.9167 g/cm**3 = 0.9167 g/(mu * m**2)
                    #       factor 1.5396=8/(3*sqrt(3)) converts reff to generalized ice particle size
                    #       use newer factor value 1.0315
                    #       1/(0.9167*1.0315) = 1.05756
                    if cdat3[k] > 0.0 and cdat4[k] > 10.0:
                        tausnw = (
                            abssnow0 * 1.05756 * cdat3[k] / cdat4[k]
                        )  # fu's formula
                    else:
                        tausnw = 0.0

                    cldliq = cliqp[k]
                    cldice = cicep[k]
                    refliq = reliq[k]
                    refice = reice[k]

                    #  --- ...  calculation of absorption coefficients due to water clouds.

                    if cldliq <= 0.0:
                        for ib in range(nbands):
                            tauliq[ib] = 0.0
                    else:
                        if ilwcliq == 1:
                            factor = refliq - 1.5
                            index = max(1, min(57, int(factor))) - 1
                            fint = factor - float(index + 1)

                            for ib in range(nbands):
                                tauliq[ib] = max(
                                    0.0,
                                    cldliq
                                    * (
                                        absliq1[index, ib]
                                        + fint
                                        * (absliq1[index + 1, ib] - absliq1[index, ib])
                                    ),
                                )

                    #  --- ...  calculation of absorption coefficients due to ice clouds.
                    if cldice <= 0.0:
                        for ib in range(nbands):
                            tauice[ib] = 0.0
                    else:
                        #  --- ...  ebert and curry approach for all particle sizes though somewhat
                        #           unjustified for large ice particles
                        if ilwcice == 1:
                            refice = min(130.0, max(13.0, np.real(refice)))

                            for ib in range(nbands):
                                ia = (
                                    ipat[ib] - 1
                                )  # eb_&_c band index for ice cloud coeff
                                tauice[ib] = max(
                                    0.0,
                                    cldice * (absice1[0, ia] + absice1[1, ia] / refice),
                                )

                            #  --- ...  streamer approach for ice effective radius between 5.0 and 131.0 microns
                            #           and ebert and curry approach for ice eff radius greater than 131.0 microns.
                            #           no smoothing between the transition of the two methods.

                        elif ilwcice == 2:
                            factor = (refice - 2.0) / 3.0
                            index = max(1, min(42, int(factor))) - 1
                            fint = factor - float(index + 1)

                            for ib in range(nbands):
                                tauice[ib] = max(
                                    0.0,
                                    cldice
                                    * (
                                        absice2[index, ib]
                                        + fint
                                        * (absice2[index + 1, ib] - absice2[index, ib])
                                    ),
                                )

                        #  --- ...  fu's approach for ice effective radius between 4.8 and 135 microns
                        #           (generalized effective size from 5 to 140 microns)

                        elif ilwcice == 3:
                            dgeice = max(5.0, 1.0315 * refice)  # v4.71 value
                            factor = (dgeice - 2.0) / 3.0
                            index = max(1, min(45, int(factor))) - 1
                            fint = factor - float(index + 1)

                            for ib in range(nbands):
                                tauice[ib] = max(
                                    0.0,
                                    cldice
                                    * (
                                        absice3[index, ib]
                                        + fint
                                        * (absice3[index + 1, ib] - absice3[index, ib])
                                    ),
                                )

                    for ib in range(nbands):
                        taucld[ib, k] = tauice[ib] + tauliq[ib] + tauran + tausnw

        else:
            for k in range(nlay):
                if cfrac[k + 1] > cldmin:
                    for ib in range(nbands):
                        taucld[ib, k] = cdat1[k]

        # -# if physparam::isubclw > 0, call mcica_subcol() to distribute
        #    cloud properties to each g-point.

        if self.isubclw > 0:  # mcica sub-col clouds approx
            for k in range(nlay):
                if cfrac[k + 1] < cldmin:
                    cldf[k] = 0.0
                else:
                    cldf[k] = cfrac[k + 1]

            #  --- ...  call sub-column cloud generator
            lcloudy = self.mcica_subcol(cldf, nlay, ipseed, dz, de_lgth, iplon)

            for k in range(nlay):
                for ig in range(ngptlw):
                    if lcloudy[ig, k]:
                        cldfmc[ig, k] = 1.0
                    else:
                        cldfmc[ig, k] = 0.0

        return cldfmc, taucld

    def taumol(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
    ):

        #  ************    original subprogram description    ***************   !
        #                                                                       !
        #                  optical depths developed for the                     !
        #                                                                       !
        #                rapid radiative transfer model (rrtm)                  !
        #                                                                       !
        #            atmospheric and environmental research, inc.               !
        #                        131 hartwell avenue                            !
        #                        lexington, ma 02421                            !
        #                                                                       !
        #                           eli j. mlawer                               !
        #                         jennifer delamere                             !
        #                         steven j. taubman                             !
        #                         shepard a. clough                             !
        #                                                                       !
        #                       email:  mlawer@aer.com                          !
        #                       email:  jdelamer@aer.com                        !
        #                                                                       !
        #        the authors wish to acknowledge the contributions of the       !
        #        following people:  karen cady-pereira, patrick d. brown,       !
        #        michael j. iacono, ronald e. farren, luke chen,                !
        #        robert bergstrom.                                              !
        #                                                                       !
        #  revision for g-point reduction: michael j. iacono; aer, inc.         !
        #                                                                       !
        #     taumol                                                            !
        #                                                                       !
        #     this file contains the subroutines taugbn (where n goes from      !
        #     1 to 16).  taugbn calculates the optical depths and planck        !
        #     fractions per g-value and layer for band n.                       !
        #                                                                       !
        #  *******************************************************************  !
        #  ==================   program usage description   ==================  !
        #                                                                       !
        #    call  taumol                                                       !
        #       inputs:                                                         !
        #          ( laytrop,pavel,coldry,colamt,colbrd,wx,tauaer,              !
        #            rfrate,fac00,fac01,fac10,fac11,jp,jt,jt1,                  !
        #            selffac,selffrac,indself,forfac,forfrac,indfor,            !
        #            minorfrac,scaleminor,scaleminorn2,indminor,                !
        #            nlay,                                                      !
        #       outputs:                                                        !
        #            fracs, tautot )                                            !
        #                                                                       !
        #  subprograms called:  taugb## (## = 01 -16)                           !
        #                                                                       !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                        size  !
        #     laytrop   - integer, tropopause layer index (unitless)        1   !
        #                   layer at which switch is made for key species       !
        #     pavel     - real, layer pressures (mb)                       nlay !
        #     coldry    - real, column amount for dry air (mol/cm2)        nlay !
        #     colamt    - real, column amounts of h2o, co2, o3, n2o, ch4,       !
        #                   o2, co (mol/cm**2)                       nlay*maxgas!
        #     colbrd    - real, column amount of broadening gases          nlay !
        #     wx        - real, cross-section amounts(mol/cm2)      nlay*maxxsec!
        #     tauaer    - real, aerosol optical depth               nbands*nlay !
        #     rfrate    - real, reference ratios of binary species parameter    !
        #     (:,m,:)m=1-h2o/co2,2-h2o/o3,3-h2o/n2o,4-h2o/ch4,5-n2o/co2,6-o3/co2!
        #     (:,:,n)n=1,2: the rates of ref press at the 2 sides of the layer  !
        #                                                          nlay*nrates*2!
        #     facij     - real, factors multiply the reference ks, i,j of 0/1   !
        #                   for lower/higher of the 2 appropriate temperatures  !
        #                   and altitudes                                  nlay !
        #     jp        - real, index of lower reference pressure          nlay !
        #     jt, jt1   - real, indices of lower reference temperatures    nlay !
        #                   for pressure levels jp and jp+1, respectively       !
        #     selffac   - real, scale factor for water vapor self-continuum     !
        #                   equals (water vapor density)/(atmospheric density   !
        #                   at 296k and 1013 mb)                           nlay !
        #     selffrac  - real, factor for temperature interpolation of         !
        #                   reference water vapor self-continuum data      nlay !
        #     indself   - integer, index of lower reference temperature for     !
        #                   the self-continuum interpolation               nlay !
        #     forfac    - real, scale factor for w. v. foreign-continuum   nlay !
        #     forfrac   - real, factor for temperature interpolation of         !
        #                   reference w.v. foreign-continuum data          nlay !
        #     indfor    - integer, index of lower reference temperature for     !
        #                   the foreign-continuum interpolation            nlay !
        #     minorfrac - real, factor for minor gases                     nlay !
        #     scaleminor,scaleminorn2                                           !
        #               - real, scale factors for minor gases              nlay !
        #     indminor  - integer, index of lower reference temperature for     !
        #                   minor gases                                    nlay !
        #     nlay      - integer, total number of layers                   1   !
        #                                                                       !
        #  outputs:                                                             !
        #     fracs     - real, planck fractions                     ngptlw,nlay!
        #     tautot    - real, total optical depth (gas+aerosols)   ngptlw,nlay!
        #                                                                       !
        #  internal variables:                                                  !
        #     ng##      - integer, number of g-values in band ## (##=01-16) 1   !
        #     nspa      - integer, for lower atmosphere, the number of ref      !
        #                   atmos, each has different relative amounts of the   !
        #                   key species for the band                      nbands!
        #     nspb      - integer, same but for upper atmosphere          nbands!
        #     absa      - real, k-values for lower ref atmospheres (no w.v.     !
        #                   self-continuum) (cm**2/molecule)  nspa(##)*5*13*ng##!
        #     absb      - real, k-values for high ref atmospheres (all sources) !
        #                   (cm**2/molecule)               nspb(##)*5*13:59*ng##!
        #     ka_m'mgas'- real, k-values for low ref atmospheres minor species  !
        #                   (cm**2/molecule)                          mmn##*ng##!
        #     kb_m'mgas'- real, k-values for high ref atmospheres minor species !
        #                   (cm**2/molecule)                          mmn##*ng##!
        #     selfref   - real, k-values for w.v. self-continuum for ref atmos  !
        #                   used below laytrop (cm**2/mol)               10*ng##!
        #     forref    - real, k-values for w.v. foreign-continuum for ref atmos
        #                   used below/above laytrop (cm**2/mol)          4*ng##!
        #                                                                       !
        #  ******************************************************************   !

        #
        # ===> ...  begin here
        #
        taug, fracs = self.taugb01(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
        )
        taug, fracs, tauself = self.taugb02(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs = self.taugb03(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
            tauself,
        )
        taug, fracs = self.taugb04(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs = self.taugb05(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs = self.taugb06(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs = self.taugb07(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs = self.taugb08(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs = self.taugb09(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs = self.taugb10(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs = self.taugb11(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs = self.taugb12(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs, taufor = self.taugb13(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs = self.taugb14(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
            taufor,
        )
        taug, fracs = self.taugb15(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )
        taug, fracs = self.taugb16(
            laytrop,
            pavel,
            coldry,
            colamt,
            colbrd,
            wx,
            tauaer,
            rfrate,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
            minorfrac,
            scaleminor,
            scaleminorn2,
            indminor,
            nlay,
            taug,
            fracs,
        )

        tautot = np.zeros((ngptlw, nlay))

        #  ---  combine gaseous and aerosol optical depths

        for ig in range(ngptlw):
            ib = ngb[ig] - 1

            for k in range(nlay):
                tautot[ig, k] = taug[ig, k] + tauaer[ib, k]

        return fracs, tautot

        # band 1:  10-350 cm-1 (low key - h2o; low minor - n2);
        #  (high key - h2o; high minor - n2)

    def taugb01(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
    ):
        #  ------------------------------------------------------------------  !
        #  written by eli j. mlawer, atmospheric & environmental research.     !
        #  revised by michael j. iacono, atmospheric & environmental research. !
        #                                                                      !
        #     band 1:  10-350 cm-1 (low key - h2o; low minor - n2)             !
        #                          (high key - h2o; high minor - n2)           !
        #                                                                      !
        #  compute the optical depth by interpolating in ln(pressure) and      !
        #  temperature.  below laytrop, the water vapor self-continuum and     !
        #  foreign continuum is interpolated (in temperature) separately.      !
        #  ------------------------------------------------------------------  !

        #  ---  minor gas mapping levels:
        #     lower - n2, p = 142.5490 mbar, t = 215.70 k
        #     upper - n2, p = 142.5490 mbar, t = 215.70 k

        #  --- ...  lower atmosphere loop

        taug = np.zeros((ngptlw, nlay))
        fracs = np.zeros((ngptlw, nlay))

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb01_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        ka_mn2 = ds["ka_mn2"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data

        ind0 = ((jp - 1) * 5 + (jt - 1)) * self.nspa[0]
        ind1 = (jp * 5 + (jt1 - 1)) * self.nspa[0]
        inds = indself - 1
        indf = indfor - 1
        indm = indminor - 1

        ind0 = ind0[:laytrop]
        ind1 = ind1[:laytrop]
        inds = inds[:laytrop]
        indf = indf[:laytrop]
        indm = indm[:laytrop]

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1

        pp = pavel[:laytrop]
        scalen2 = colbrd[:laytrop] * scaleminorn2[:laytrop]
        corradj = np.where(pp < 250.0, 1.0 - 0.15 * (250.0 - pp) / 154.4, 1.0)
        # if pp < 250.0:
        #     corradj = 1.0 - 0.15 * (250.0-pp) / 154.4
        # else:
        #     corradj = 1.0

        for ig in range(ng01):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )
            taun2 = scalen2 * (
                ka_mn2[ig, indm]
                + minorfrac[:laytrop] * (ka_mn2[ig, indmp] - ka_mn2[ig, indm])
            )

            taug[ig, :laytrop] = corradj * (
                colamt[:laytrop, 0]
                * (
                    fac00[:laytrop] * absa[ig, ind0]
                    + fac10[:laytrop] * absa[ig, ind0p]
                    + fac01[:laytrop] * absa[ig, ind1]
                    + fac11[:laytrop] * absa[ig, ind1p]
                )
                + tauself
                + taufor
                + taun2
            )

            fracs[ig, :laytrop] = fracrefa[ig]

        #  --- ...  upper atmosphere loop

        ind0 = ((jp - 13) * 5 + (jt - 1)) * self.nspb[0]
        ind1 = ((jp - 12) * 5 + (jt1 - 1)) * self.nspb[0]
        indf = indfor - 1
        indm = indminor - 1

        ind0 = ind0[laytrop:nlay]
        ind1 = ind1[laytrop:nlay]
        indf = indf[laytrop:nlay]
        indm = indm[laytrop:nlay]

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indfp = indf + 1
        indmp = indm + 1

        scalen2 = colbrd[laytrop:nlay] * scaleminorn2[laytrop:nlay]
        corradj = 1.0 - 0.15 * (pavel[laytrop:nlay] / 95.6)

        for ig in range(ng01):
            taufor = forfac[laytrop:nlay] * (
                forref[ig, indf]
                + forfrac[laytrop:nlay] * (forref[ig, indfp] - forref[ig, indf])
            )
            taun2 = scalen2 * (
                ka_mn2[ig, indm]
                + minorfrac[laytrop:nlay] * (ka_mn2[ig, indmp] - ka_mn2[ig, indm])
            )

            taug[ig, laytrop:nlay] = corradj * (
                colamt[laytrop:nlay, 0]
                * (
                    fac00[laytrop:nlay] * absb[ig, ind0]
                    + fac10[laytrop:nlay] * absb[ig, ind0p]
                    + fac01[laytrop:nlay] * absb[ig, ind1]
                    + fac11[laytrop:nlay] * absb[ig, ind1p]
                )
                + taufor
                + taun2
            )

            fracs[ig, laytrop:nlay] = fracrefb[ig]

        return taug, fracs

    # Band 2:  350-500 cm-1 (low key - h2o; high key - h2o)
    def taugb02(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 2:  350-500 cm-1 (low key - h2o; high key - h2o)            !
        #  ------------------------------------------------------------------  !
        #
        # ===> ...  begin here
        #
        #  --- ...  lower atmosphere loop

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb02_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data

        ind0 = ((jp - 1) * 5 + (jt - 1)) * self.nspa[1]
        ind1 = (jp * 5 + (jt1 - 1)) * self.nspa[1]
        inds = indself - 1
        indf = indfor - 1

        ind0 = ind0[:laytrop]
        ind1 = ind1[:laytrop]
        inds = inds[:laytrop]
        indf = indf[:laytrop]

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1

        corradj = 1.0 - 0.05 * (pavel[:laytrop] - 100.0) / 900.0

        for ig in range(ng02):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )

            taug[ns02 + ig, :laytrop] = corradj * (
                colamt[:laytrop, 0]
                * (
                    fac00[:laytrop] * absa[ig, ind0]
                    + fac10[:laytrop] * absa[ig, ind0p]
                    + fac01[:laytrop] * absa[ig, ind1]
                    + fac11[:laytrop] * absa[ig, ind1p]
                )
                + +tauself
                + taufor
            )

            fracs[ns02 + ig, :laytrop] = fracrefa[ig]

        #  --- ...  upper atmosphere loop

        ind0 = ((jp - 13) * 5 + (jt - 1)) * self.nspb[1]
        ind1 = ((jp - 12) * 5 + (jt1 - 1)) * self.nspb[1]
        indf = indfor - 1

        ind0 = ind0[laytrop:nlay]
        ind1 = ind1[laytrop:nlay]
        indf = indf[laytrop:nlay]

        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indfp = indf + 1

        for ig in range(ng02):
            taufor = forfac[laytrop:nlay] * (
                forref[ig, indf]
                + forfrac[laytrop:nlay] * (forref[ig, indfp] - forref[ig, indf])
            )

            taug[ns02 + ig, laytrop:nlay] = (
                colamt[laytrop:nlay, 0]
                * (
                    fac00[laytrop:nlay] * absb[ig, ind0]
                    + fac10[laytrop:nlay] * absb[ig, ind0p]
                    + fac01[laytrop:nlay] * absb[ig, ind1]
                    + fac11[laytrop:nlay] * absb[ig, ind1p]
                )
                + taufor
            )

            fracs[ns02 + ig, laytrop:nlay] = fracrefb[ig]

        return taug, fracs, tauself

    # Band 3:  500-630 cm-1 (low key - h2o,co2; low minor - n2o);
    #                        (high key - h2o,co2; high minor - n2o)
    def taugb03(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
        tauself,
    ):
        #  ------------------------------------------------------------------  !
        #     band 3:  500-630 cm-1 (low key - h2o,co2; low minor - n2o)       !
        #                           (high key - h2o,co2; high minor - n2o)     !
        #  ------------------------------------------------------------------  !

        #
        # ===> ...  begin here
        #
        #  --- ...  minor gas mapping levels:
        #     lower - n2o, p = 706.272 mbar, t = 278.94 k
        #     upper - n2o, p = 95.58 mbar, t = 215.7 k

        dsc = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))
        chi_mls = dsc["chi_mls"].data

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb03_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        ka_mn2o = ds["ka_mn2o"].data
        kb_mn2o = ds["kb_mn2o"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data

        refrat_planck_a = chi_mls[0, 8] / chi_mls[1, 8]  # P = 212.725 mb
        refrat_planck_b = chi_mls[0, 12] / chi_mls[1, 12]  # P = 95.58   mb
        refrat_m_a = chi_mls[0, 2] / chi_mls[1, 2]  # P = 706.270 mb
        refrat_m_b = chi_mls[0, 12] / chi_mls[1, 12]  # P = 95.58   mb

        #  --- ...  lower atmosphere loop

        speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 0] * colamt[:laytrop, 1]
        specparm = colamt[:laytrop, 0] / speccomb
        specmult = 8.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[2] + js - 1

        speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 1] * colamt[:laytrop, 1]
        specparm1 = colamt[:laytrop, 0] / speccomb1
        specmult1 = 8.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[2] + js1 - 1

        speccomb_mn2o = colamt[:laytrop, 0] + refrat_m_a * colamt[:laytrop, 1]
        specparm_mn2o = colamt[:laytrop, 0] / speccomb_mn2o
        specmult_mn2o = 8.0 * np.minimum(specparm_mn2o, self.oneminus)
        jmn2o = 1 + specmult_mn2o.astype(np.int32) - 1
        fmn2o = specmult_mn2o % 1.0

        speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 1]
        specparm_planck = colamt[:laytrop, 0] / speccomb_planck
        specmult_planck = 8.0 * np.minimum(specparm_planck, self.oneminus)
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indm = indminor[:laytrop] - 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jmn2op = jmn2o + 1
        jplp = jpl + 1

        #  --- ...  in atmospheres where the amount of n2O is too great to be considered
        #           a minor species, adjust the column amount of n2O by an empirical factor
        #           to obtain the proper contribution.

        p = coldry[:laytrop] * chi_mls[3, jp[:laytrop]]
        ratn2o = colamt[:laytrop, 3] / p

        adjcoln2o = np.where(
            ratn2o > 1.5, (0.5 + (ratn2o - 0.5) ** 0.65) * p, colamt[:laytrop, 3]
        )

        p = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(specparm > 0.875, -fs, 0)
        p = np.where(p == 0, 0, p)

        p4 = np.where(specparm < 0.125, p ** 4, 0) + np.where(
            specparm > 0.875, p ** 4, 0
        )
        p4 = np.where(p4 == 0, 0, p4)

        fk0 = np.where(specparm < 0.125, p4, 0) + np.where(specparm > 0.875, p ** 4, 0)
        fk0 = np.where(fk0 == 0, 1.0 - fs, fk0)

        fk1 = np.where(specparm < 0.125, 1.0 - p - 2.0 * p4, 0) + np.where(
            specparm > 0.875, 1.0 - p - 2.0 * p4, 0
        )
        fk1 = np.where(fk1 == 0, fs, fk1)

        fk2 = np.where(specparm < 0.125, p + p4, 0) + np.where(
            specparm > 0.875, p + p4, 0
        )
        fk2 = np.where(fk2 == 0, 0.0, fk2)

        id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
            specparm > 0.875, ind0 + 1, 0
        )
        id000 = np.where(id000 == 0, ind0, id000)

        id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
            specparm > 0.875, ind0 + 10, 0
        )
        id010 = np.where(id010 == 0, ind0 + 9, id010)

        id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
            specparm > 0.875, ind0, 0
        )
        id100 = np.where(id100 == 0, ind0 + 1, id100)

        id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
            specparm > 0.875, ind0 + 9, 0
        )
        id110 = np.where(id110 == 0, ind0 + 10, id110)

        id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
            specparm > 0.875, ind0 - 1, 0
        )
        id200 = np.where(id200 == 0, ind0, id200)

        id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
            specparm > 0.875, ind0 + 8, 0
        )
        id210 = np.where(id210 == 0, ind0, id210)

        fac000 = fk0 * fac00[:laytrop]
        fac100 = fk1 * fac00[:laytrop]
        fac200 = fk2 * fac00[:laytrop]
        fac010 = fk0 * fac10[:laytrop]
        fac110 = fk1 * fac10[:laytrop]
        fac210 = fk2 * fac10[:laytrop]

        p = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
            specparm1 > 0.875, -fs1, 0
        )
        p = np.where(p == 0, 0, p)

        p4 = np.where(specparm1 < 0.125, p ** 4, 0) + np.where(
            specparm1 > 0.875, p ** 4, 0
        )
        p4 = np.where(p4 == 0, 0, p4)

        fk0 = np.where(specparm1 < 0.125, p4, 0) + np.where(
            specparm1 > 0.875, p ** 4, 0
        )
        fk0 = np.where(fk0 == 0, 1.0 - fs1, fk0)

        fk1 = np.where(specparm1 < 0.125, 1.0 - p - 2.0 * p4, 0) + np.where(
            specparm1 > 0.875, 1.0 - p - 2.0 * p4, 0
        )
        fk1 = np.where(fk1 == 0, fs1, fk1)

        fk2 = np.where(specparm1 < 0.125, p + p4, 0) + np.where(
            specparm1 > 0.875, p + p4, 0
        )
        fk2 = np.where(fk2 == 0, 0.0, fk2)

        id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
            specparm1 > 0.875, ind1 + 1, 0
        )
        id001 = np.where(id001 == 0, ind1, id001)

        id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
            specparm1 > 0.875, ind1 + 10, 0
        )
        id011 = np.where(id011 == 0, ind1 + 9, id011)

        id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
            specparm1 > 0.875, ind1, 0
        )
        id101 = np.where(id101 == 0, ind1 + 1, id101)

        id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
            specparm1 > 0.875, ind1 + 9, 0
        )
        id111 = np.where(id111 == 0, ind1 + 10, id111)

        id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
            specparm1 > 0.875, ind1 - 1, 0
        )
        id201 = np.where(id201 == 0, ind1, id201)

        id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
            specparm1 > 0.875, ind1 + 8, 0
        )
        id211 = np.where(id211 == 0, ind1, id211)

        fac001 = fk0 * fac01[:laytrop]
        fac101 = fk1 * fac01[:laytrop]
        fac201 = fk2 * fac01[:laytrop]
        fac011 = fk0 * fac11[:laytrop]
        fac111 = fk1 * fac11[:laytrop]
        fac211 = fk2 * fac11[:laytrop]

        for ig in range(ng03):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )
            n2om1 = ka_mn2o[ig, jmn2o, indm] + fmn2o * (
                ka_mn2o[ig, jmn2op, indm] - ka_mn2o[ig, jmn2o, indm]
            )
            n2om2 = ka_mn2o[ig, jmn2o, indmp] + fmn2o * (
                ka_mn2o[ig, jmn2op, indmp] - ka_mn2o[ig, jmn2o, indmp]
            )
            absn2o = n2om1 + minorfrac[:laytrop] * (n2om2 - n2om1)

            tau_major = speccomb * (
                fac000 * absa[ig, id000]
                + fac010 * absa[ig, id010]
                + fac100 * absa[ig, id100]
                + fac110 * absa[ig, id110]
                + fac200 * absa[ig, id200]
                + fac210 * absa[ig, id210]
            )

            tau_major1 = speccomb1 * (
                fac001 * absa[ig, id001]
                + fac011 * absa[ig, id011]
                + fac101 * absa[ig, id101]
                + fac111 * absa[ig, id111]
                + fac201 * absa[ig, id201]
                + fac211 * absa[ig, id211]
            )

            taug[ns03 + ig, :laytrop] = (
                tau_major + tau_major1 + tauself + taufor + adjcoln2o * absn2o
            )

            fracs[ns03 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
                fracrefa[ig, jplp] - fracrefa[ig, jpl]
            )

        #  --- ...  upper atmosphere loop

        speccomb = (
            colamt[laytrop:nlay, 0]
            + rfrate[laytrop:nlay, 0, 0] * colamt[laytrop:nlay, 1]
        )
        specparm = colamt[laytrop:nlay, 0] / speccomb
        specmult = 4.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = (
            ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * self.nspb[2]
            + js
            - 1
        )

        speccomb1 = (
            colamt[laytrop:nlay, 0]
            + rfrate[laytrop:nlay, 0, 1] * colamt[laytrop:nlay, 1]
        )
        specparm1 = colamt[laytrop:nlay, 0] / speccomb1
        specmult1 = 4.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (
            ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * self.nspb[2]
            + js1
            - 1
        )

        speccomb_mn2o = colamt[laytrop:nlay, 0] + refrat_m_b * colamt[laytrop:nlay, 1]
        specparm_mn2o = colamt[laytrop:nlay, 0] / speccomb_mn2o
        specmult_mn2o = 4.0 * np.minimum(specparm_mn2o, self.oneminus)
        jmn2o = 1 + specmult_mn2o.astype(np.int32) - 1
        fmn2o = specmult_mn2o % 1.0

        speccomb_planck = (
            colamt[laytrop:nlay, 0] + refrat_planck_b * colamt[laytrop:nlay, 1]
        )
        specparm_planck = colamt[laytrop:nlay, 0] / speccomb_planck
        specmult_planck = 4.0 * np.minimum(specparm_planck, self.oneminus)
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0

        indf = indfor[laytrop:nlay] - 1
        indm = indminor[laytrop:nlay] - 1
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

        p = coldry[laytrop:nlay] * chi_mls[3, jp[laytrop:nlay]]
        ratn2o = colamt[laytrop:nlay, 3] / p
        adjcoln2o = np.where(
            ratn2o > 1.5, (0.5 + (ratn2o - 0.5) ** 0.65) * p, colamt[laytrop:nlay, 3]
        )
        # if ratn2o > 1.5:
        #     adjfac = 0.5 + (ratn2o - 0.5)**0.65
        #     adjcoln2o = adjfac * p
        # else:
        #     adjcoln2o = colamt[laytrop:nlay, 3]

        fk0 = 1.0 - fs
        fk1 = fs
        fac000 = fk0 * fac00[laytrop:nlay]
        fac010 = fk0 * fac10[laytrop:nlay]
        fac100 = fk1 * fac00[laytrop:nlay]
        fac110 = fk1 * fac10[laytrop:nlay]

        fk0 = 1.0 - fs1
        fk1 = fs1
        fac001 = fk0 * fac01[laytrop:nlay]
        fac011 = fk0 * fac11[laytrop:nlay]
        fac101 = fk1 * fac01[laytrop:nlay]
        fac111 = fk1 * fac11[laytrop:nlay]

        for ig in range(ng03):
            taufor = forfac[laytrop:nlay] * (
                forref[ig, indf]
                + forfrac[laytrop:nlay] * (forref[ig, indfp] - forref[ig, indf])
            )
            n2om1 = kb_mn2o[ig, jmn2o, indm] + fmn2o * (
                kb_mn2o[ig, jmn2op, indm] - kb_mn2o[ig, jmn2o, indm]
            )
            n2om2 = kb_mn2o[ig, jmn2o, indmp] + fmn2o * (
                kb_mn2o[ig, jmn2op, indmp] - kb_mn2o[ig, jmn2o, indmp]
            )
            absn2o = n2om1 + minorfrac[laytrop:nlay] * (n2om2 - n2om1)

            tau_major = speccomb * (
                fac000 * absb[ig, id000]
                + fac010 * absb[ig, id010]
                + fac100 * absb[ig, id100]
                + fac110 * absb[ig, id110]
            )

            tau_major1 = speccomb1 * (
                fac001 * absb[ig, id001]
                + fac011 * absb[ig, id011]
                + fac101 * absb[ig, id101]
                + fac111 * absb[ig, id111]
            )

            taug[ns03 + ig, laytrop:nlay] = (
                tau_major + tau_major1 + taufor + adjcoln2o * absn2o
            )

            fracs[ns03 + ig, laytrop:nlay] = fracrefb[ig, jpl] + fpl * (
                fracrefb[ig, jplp] - fracrefb[ig, jpl]
            )

        return taug, fracs

    # Band 4:  630-700 cm-1 (low key - h2o,co2; high key - o3,co2)
    # ----------------------------------
    def taugb04(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 4:  630-700 cm-1 (low key - h2o,co2; high key - o3,co2)     !
        #  ------------------------------------------------------------------  !
        #
        # ===> ...  begin here
        #

        dsc = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))
        chi_mls = dsc["chi_mls"].data

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb04_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data

        refrat_planck_a = chi_mls[0, 10] / chi_mls[1, 10]  # P = 142.5940 mb
        refrat_planck_b = chi_mls[2, 12] / chi_mls[1, 12]  # P = 95.58350 mb

        #  --- ...  lower atmosphere loop
        speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 0] * colamt[:laytrop, 1]
        specparm = colamt[:laytrop, 0] / speccomb
        specmult = 8.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[3] + js - 1

        speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 1] * colamt[:laytrop, 1]
        specparm1 = colamt[:laytrop, 0] / speccomb1
        specmult1 = 8.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[3] + js1 - 1

        speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 1]
        specparm_planck = colamt[:laytrop, 0] / speccomb_planck
        specmult_planck = 8.0 * np.minimum(specparm_planck, self.oneminus)
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indsp = inds + 1
        indfp = indf + 1
        jplp = jpl + 1

        p = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(specparm > 0.875, -fs, 0)
        p = np.where(p == 0, 0, p)

        p4 = np.where(specparm < 0.125, p ** 4, 0) + np.where(
            specparm > 0.875, p ** 4, 0
        )
        p4 = np.where(p4 == 0, 0, p4)

        fk0 = np.where(specparm < 0.125, p4, 0) + np.where(specparm > 0.875, p ** 4, 0)
        fk0 = np.where(fk0 == 0, 1.0 - fs, fk0)

        fk1 = np.where(specparm < 0.125, 1.0 - p - 2.0 * p4, 0) + np.where(
            specparm > 0.875, 1.0 - p - 2.0 * p4, 0
        )
        fk1 = np.where(fk1 == 0, fs, fk1)

        fk2 = np.where(specparm < 0.125, p + p4, 0) + np.where(
            specparm > 0.875, p + p4, 0
        )
        fk2 = np.where(fk2 == 0, 0.0, fk2)

        id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
            specparm > 0.875, ind0 + 1, 0
        )
        id000 = np.where(id000 == 0, ind0, id000)

        id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
            specparm > 0.875, ind0 + 10, 0
        )
        id010 = np.where(id010 == 0, ind0 + 9, id010)

        id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
            specparm > 0.875, ind0, 0
        )
        id100 = np.where(id100 == 0, ind0 + 1, id100)

        id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
            specparm > 0.875, ind0 + 9, 0
        )
        id110 = np.where(id110 == 0, ind0 + 10, id110)

        id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
            specparm > 0.875, ind0 - 1, 0
        )
        id200 = np.where(id200 == 0, ind0, id200)

        id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
            specparm > 0.875, ind0 + 8, 0
        )
        id210 = np.where(id210 == 0, ind0, id210)

        fac000 = fk0 * fac00[:laytrop]
        fac100 = fk1 * fac00[:laytrop]
        fac200 = fk2 * fac00[:laytrop]
        fac010 = fk0 * fac10[:laytrop]
        fac110 = fk1 * fac10[:laytrop]
        fac210 = fk2 * fac10[:laytrop]

        p = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
            specparm1 > 0.875, -fs1, 0
        )
        p = np.where(p == 0, 0, p)

        p4 = np.where(specparm1 < 0.125, p ** 4, 0) + np.where(
            specparm1 > 0.875, p ** 4, 0
        )
        p4 = np.where(p4 == 0, 0, p4)

        fk0 = np.where(specparm1 < 0.125, p4, 0) + np.where(
            specparm1 > 0.875, p ** 4, 0
        )
        fk0 = np.where(fk0 == 0, 1.0 - fs1, fk0)

        fk1 = np.where(specparm1 < 0.125, 1.0 - p - 2.0 * p4, 0) + np.where(
            specparm1 > 0.875, 1.0 - p - 2.0 * p4, 0
        )
        fk1 = np.where(fk1 == 0, fs1, fk1)

        fk2 = np.where(specparm1 < 0.125, p + p4, 0) + np.where(
            specparm1 > 0.875, p + p4, 0
        )
        fk2 = np.where(fk2 == 0, 0.0, fk2)

        id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
            specparm1 > 0.875, ind1 + 1, 0
        )
        id001 = np.where(id001 == 0, ind1, id001)

        id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
            specparm1 > 0.875, ind1 + 10, 0
        )
        id011 = np.where(id011 == 0, ind1 + 9, id011)

        id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
            specparm1 > 0.875, ind1, 0
        )
        id101 = np.where(id101 == 0, ind1 + 1, id101)

        id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
            specparm1 > 0.875, ind1 + 9, 0
        )
        id111 = np.where(id111 == 0, ind1 + 10, id111)

        id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
            specparm1 > 0.875, ind1 - 1, 0
        )
        id201 = np.where(id201 == 0, ind1, id201)

        id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
            specparm1 > 0.875, ind1 + 8, 0
        )
        id211 = np.where(id211 == 0, ind1, id211)

        fac001 = fk0 * fac01[:laytrop]
        fac101 = fk1 * fac01[:laytrop]
        fac201 = fk2 * fac01[:laytrop]
        fac011 = fk0 * fac11[:laytrop]
        fac111 = fk1 * fac11[:laytrop]
        fac211 = fk2 * fac11[:laytrop]

        for ig in range(ng04):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )

            tau_major = speccomb * (
                fac000 * absa[ig, id000]
                + fac010 * absa[ig, id010]
                + fac100 * absa[ig, id100]
                + fac110 * absa[ig, id110]
                + fac200 * absa[ig, id200]
                + fac210 * absa[ig, id210]
            )

            tau_major1 = speccomb1 * (
                fac001 * absa[ig, id001]
                + fac011 * absa[ig, id011]
                + fac101 * absa[ig, id101]
                + fac111 * absa[ig, id111]
                + fac201 * absa[ig, id201]
                + fac211 * absa[ig, id211]
            )

            taug[ns04 + ig, :laytrop] = tau_major + tau_major1 + tauself + taufor

            fracs[ns04 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
                fracrefa[ig, jplp] - fracrefa[ig, jpl]
            )

        #  --- ...  upper atmosphere loop
        speccomb = (
            colamt[laytrop:nlay, 2]
            + rfrate[laytrop:nlay, 5, 0] * colamt[laytrop:nlay, 1]
        )
        specparm = colamt[laytrop:nlay, 2] / speccomb
        specmult = 4.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = (
            ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * self.nspb[3]
            + js
            - 1
        )

        speccomb1 = (
            colamt[laytrop:nlay, 2]
            + rfrate[laytrop:nlay, 5, 1] * colamt[laytrop:nlay, 1]
        )
        specparm1 = colamt[laytrop:nlay, 2] / speccomb1
        specmult1 = 4.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (
            ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * self.nspb[3]
            + js1
            - 1
        )

        speccomb_planck = (
            colamt[laytrop:nlay, 2] + refrat_planck_b * colamt[laytrop:nlay, 1]
        )
        specparm_planck = colamt[laytrop:nlay, 2] / speccomb_planck
        specmult_planck = 4.0 * np.minimum(specparm_planck, self.oneminus)
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0
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
        fac000 = fk0 * fac00[laytrop:nlay]
        fac010 = fk0 * fac10[laytrop:nlay]
        fac100 = fk1 * fac00[laytrop:nlay]
        fac110 = fk1 * fac10[laytrop:nlay]

        fk0 = 1.0 - fs1
        fk1 = fs1
        fac001 = fk0 * fac01[laytrop:nlay]
        fac011 = fk0 * fac11[laytrop:nlay]
        fac101 = fk1 * fac01[laytrop:nlay]
        fac111 = fk1 * fac11[laytrop:nlay]

        for ig in range(ng04):
            tau_major = speccomb * (
                fac000 * absb[ig, id000]
                + fac010 * absb[ig, id010]
                + fac100 * absb[ig, id100]
                + fac110 * absb[ig, id110]
            )
            tau_major1 = speccomb1 * (
                fac001 * absb[ig, id001]
                + fac011 * absb[ig, id011]
                + fac101 * absb[ig, id101]
                + fac111 * absb[ig, id111]
            )

            taug[ns04 + ig, laytrop:nlay] = tau_major + tau_major1

            fracs[ns04 + ig, laytrop:nlay] = fracrefb[ig, jpl] + fpl * (
                fracrefb[ig, jplp] - fracrefb[ig, jpl]
            )

            #  --- ...  empirical modification to code to improve stratospheric cooling rates
            #           for co2. revised to apply weighting for g-point reduction in this band.

        taug[ns04 + 7, laytrop:nlay] = taug[ns04 + 7, laytrop:nlay] * 0.92
        taug[ns04 + 8, laytrop:nlay] = taug[ns04 + 8, laytrop:nlay] * 0.88
        taug[ns04 + 9, laytrop:nlay] = taug[ns04 + 9, laytrop:nlay] * 1.07
        taug[ns04 + 10, laytrop:nlay] = taug[ns04 + 10, laytrop:nlay] * 1.1
        taug[ns04 + 11, laytrop:nlay] = taug[ns04 + 11, laytrop:nlay] * 0.99
        taug[ns04 + 12, laytrop:nlay] = taug[ns04 + 12, laytrop:nlay] * 0.88
        taug[ns04 + 13, laytrop:nlay] = taug[ns04 + 13, laytrop:nlay] * 0.943

        return taug, fracs

    # Band 5:  700-820 cm-1 (low key - h2o,co2; low minor - o3, ccl4)
    #                       (high key - o3,co2)
    def taugb05(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 5:  700-820 cm-1 (low key - h2o,co2; low minor - o3, ccl4)  !
        #                           (high key - o3,co2)                        !
        #  ------------------------------------------------------------------  !
        #
        # ===> ...  begin here
        #
        #  --- ...  minor gas mapping level :
        #     lower - o3, p = 317.34 mbar, t = 240.77 k
        #     lower - ccl4

        #  --- ...  calculate reference ratio to be used in calculation of Planck
        #           fraction in lower/upper atmosphere.

        dsc = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))
        chi_mls = dsc["chi_mls"].data

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb05_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data
        ka_mo3 = ds["ka_mo3"].data
        ccl4 = ds["ccl4"].data

        refrat_planck_a = chi_mls[0, 4] / chi_mls[1, 4]  # P = 473.420 mb
        refrat_planck_b = chi_mls[2, 42] / chi_mls[1, 42]  # P = 0.2369  mb
        refrat_m_a = chi_mls[0, 6] / chi_mls[1, 6]  # P = 317.348 mb

        #  --- ...  lower atmosphere loop

        speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 0] * colamt[:laytrop, 1]
        specparm = colamt[:laytrop, 0] / speccomb
        specmult = 8.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[4] + js - 1

        speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 1] * colamt[:laytrop, 1]
        specparm1 = colamt[:laytrop, 0] / speccomb1
        specmult1 = 8.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[4] + js1 - 1

        speccomb_mo3 = colamt[:laytrop, 0] + refrat_m_a * colamt[:laytrop, 1]
        specparm_mo3 = colamt[:laytrop, 0] / speccomb_mo3
        specmult_mo3 = 8.0 * np.minimum(specparm_mo3, self.oneminus)
        jmo3 = 1 + specmult_mo3.astype(np.int32) - 1
        fmo3 = specmult_mo3 % 1.0

        speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 1]
        specparm_planck = colamt[:laytrop, 0] / speccomb_planck
        specmult_planck = 8.0 * np.minimum(specparm_planck, self.oneminus)
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indm = indminor[:laytrop] - 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jplp = jpl + 1
        jmo3p = jmo3 + 1

        p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(
            specparm > 0.875, -fs, 0
        )
        p0 = np.where(p0 == 0, 0, p0)

        p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        p40 = np.where(p40 == 0, 0, p40)

        fk00 = np.where(specparm < 0.125, p40, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

        fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
            specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
        )
        fk10 = np.where(fk10 == 0, fs, fk10)

        fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
            specparm > 0.875, p0 + p40, 0
        )
        fk20 = np.where(fk20 == 0, 0.0, fk20)

        id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
            specparm > 0.875, ind0 + 1, 0
        )
        id000 = np.where(id000 == 0, ind0, id000)

        id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
            specparm > 0.875, ind0 + 10, 0
        )
        id010 = np.where(id010 == 0, ind0 + 9, id010)

        id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
            specparm > 0.875, ind0, 0
        )
        id100 = np.where(id100 == 0, ind0 + 1, id100)

        id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
            specparm > 0.875, ind0 + 9, 0
        )
        id110 = np.where(id110 == 0, ind0 + 10, id110)

        id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
            specparm > 0.875, ind0 - 1, 0
        )
        id200 = np.where(id200 == 0, ind0, id200)

        id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
            specparm > 0.875, ind0 + 8, 0
        )
        id210 = np.where(id210 == 0, ind0, id210)

        fac000 = fk00 * fac00[:laytrop]
        fac100 = fk10 * fac00[:laytrop]
        fac200 = fk20 * fac00[:laytrop]
        fac010 = fk00 * fac10[:laytrop]
        fac110 = fk10 * fac10[:laytrop]
        fac210 = fk20 * fac10[:laytrop]

        p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
            specparm1 > 0.875, -fs1, 0
        )
        p1 = np.where(p1 == 0, 0, p1)

        p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        p41 = np.where(p41 == 0, 0, p41)

        fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

        fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
            specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
        )
        fk11 = np.where(fk11 == 0, fs1, fk11)

        fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
            specparm1 > 0.875, p1 + p41, 0
        )
        fk21 = np.where(fk21 == 0, 0.0, fk21)

        id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
            specparm1 > 0.875, ind1 + 1, 0
        )
        id001 = np.where(id001 == 0, ind1, id001)

        id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
            specparm1 > 0.875, ind1 + 10, 0
        )
        id011 = np.where(id011 == 0, ind1 + 9, id011)

        id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
            specparm1 > 0.875, ind1, 0
        )
        id101 = np.where(id101 == 0, ind1 + 1, id101)

        id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
            specparm1 > 0.875, ind1 + 9, 0
        )
        id111 = np.where(id111 == 0, ind1 + 10, id111)

        id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
            specparm1 > 0.875, ind1 - 1, 0
        )
        id201 = np.where(id201 == 0, ind1, id201)

        id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
            specparm1 > 0.875, ind1 + 8, 0
        )
        id211 = np.where(id211 == 0, ind1, id211)

        fac001 = fk01 * fac01[:laytrop]
        fac101 = fk11 * fac01[:laytrop]
        fac201 = fk21 * fac01[:laytrop]
        fac011 = fk01 * fac11[:laytrop]
        fac111 = fk11 * fac11[:laytrop]
        fac211 = fk21 * fac11[:laytrop]

        for ig in range(ng05):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )
            o3m1 = ka_mo3[ig, jmo3, indm] + fmo3 * (
                ka_mo3[ig, jmo3p, indm] - ka_mo3[ig, jmo3, indm]
            )
            o3m2 = ka_mo3[ig, jmo3, indmp] + fmo3 * (
                ka_mo3[ig, jmo3p, indmp] - ka_mo3[ig, jmo3, indmp]
            )
            abso3 = o3m1 + minorfrac[:laytrop] * (o3m2 - o3m1)

            taug[ns05 + ig, :laytrop] = (
                speccomb
                * (
                    fac000 * absa[ig, id000]
                    + fac010 * absa[ig, id010]
                    + fac100 * absa[ig, id100]
                    + fac110 * absa[ig, id110]
                    + fac200 * absa[ig, id200]
                    + fac210 * absa[ig, id210]
                )
                + speccomb1
                * (
                    fac001 * absa[ig, id001]
                    + fac011 * absa[ig, id011]
                    + fac101 * absa[ig, id101]
                    + fac111 * absa[ig, id111]
                    + fac201 * absa[ig, id201]
                    + fac211 * absa[ig, id211]
                )
                + tauself
                + taufor
                + abso3 * colamt[:laytrop, 2]
                + wx[:laytrop, 0] * ccl4[ig]
            )

            fracs[ns05 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
                fracrefa[ig, jplp] - fracrefa[ig, jpl]
            )

        #  --- ...  upper atmosphere loop

        speccomb = (
            colamt[laytrop:nlay, 2]
            + rfrate[laytrop:nlay, 5, 0] * colamt[laytrop:nlay, 1]
        )
        specparm = colamt[laytrop:nlay, 2] / speccomb
        specmult = 4.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = (
            ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * self.nspb[4]
            + js
            - 1
        )

        speccomb1 = (
            colamt[laytrop:nlay, 2]
            + rfrate[laytrop:nlay, 5, 1] * colamt[laytrop:nlay, 1]
        )
        specparm1 = colamt[laytrop:nlay, 2] / speccomb1
        specmult1 = 4.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (
            ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * self.nspb[4]
            + js1
            - 1
        )

        speccomb_planck = (
            colamt[laytrop:nlay, 2] + refrat_planck_b * colamt[laytrop:nlay, 1]
        )
        specparm_planck = colamt[laytrop:nlay, 2] / speccomb_planck
        specmult_planck = 4.0 * np.minimum(specparm_planck, self.oneminus)
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0
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

        fac000 = fk00 * fac00[laytrop:nlay]
        fac010 = fk00 * fac10[laytrop:nlay]
        fac100 = fk10 * fac00[laytrop:nlay]
        fac110 = fk10 * fac10[laytrop:nlay]

        fac001 = fk01 * fac01[laytrop:nlay]
        fac011 = fk01 * fac11[laytrop:nlay]
        fac101 = fk11 * fac01[laytrop:nlay]
        fac111 = fk11 * fac11[laytrop:nlay]

        for ig in range(ng05):
            taug[ns05 + ig, laytrop:nlay] = (
                speccomb
                * (
                    fac000 * absb[ig, id000]
                    + fac010 * absb[ig, id010]
                    + fac100 * absb[ig, id100]
                    + fac110 * absb[ig, id110]
                )
                + speccomb1
                * (
                    fac001 * absb[ig, id001]
                    + fac011 * absb[ig, id011]
                    + fac101 * absb[ig, id101]
                    + fac111 * absb[ig, id111]
                )
                + wx[laytrop:nlay, 0] * ccl4[ig]
            )

            fracs[ns05 + ig, laytrop:nlay] = fracrefb[ig, jpl] + fpl * (
                fracrefb[ig, jplp] - fracrefb[ig, jpl]
            )

        return taug, fracs

    # Band 6:  820-980 cm-1 (low key - h2o; low minor - co2)
    #                       (high key - none; high minor - cfc11, cfc12)
    def taugb06(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 6:  820-980 cm-1 (low key - h2o; low minor - co2)           !
        #                           (high key - none; high minor - cfc11, cfc12)
        #  ------------------------------------------------------------------  !

        #  --- ...  minor gas mapping level:
        #     lower - co2, p = 706.2720 mb, t = 294.2 k
        #     upper - cfc11, cfc12

        dsc = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))
        chi_mls = dsc["chi_mls"].data

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb06_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        fracrefa = ds["fracrefa"].data
        ka_mco2 = ds["ka_mco2"].data
        cfc11adj = ds["cfc11adj"].data
        cfc12 = ds["cfc12"].data

        #  --- ...  lower atmosphere loop
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[5]
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[5]

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indm = indminor[:laytrop] - 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1

        #  --- ...  in atmospheres where the amount of co2 is too great to be considered
        #           a minor species, adjust the column amount of co2 by an empirical factor
        #           to obtain the proper contribution.

        temp = coldry[:laytrop] * chi_mls[1, jp[:laytrop] + 1]
        ratco2 = colamt[:laytrop, 1] / temp
        adjcolco2 = np.where(
            ratco2 > 3.0, (2.0 + (ratco2 - 2.0) ** 0.77) * temp, colamt[:laytrop, 1]
        )

        for ig in range(ng06):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )
            absco2 = ka_mco2[ig, indm] + minorfrac[:laytrop] * (
                ka_mco2[ig, indmp] - ka_mco2[ig, indm]
            )

            taug[ns06 + ig, :laytrop] = (
                colamt[:laytrop, 0]
                * (
                    fac00[:laytrop] * absa[ig, ind0]
                    + fac10[:laytrop] * absa[ig, ind0p]
                    + fac01[:laytrop] * absa[ig, ind1]
                    + fac11[:laytrop] * absa[ig, ind1p]
                )
                + tauself
                + taufor
                + adjcolco2 * absco2
                + wx[:laytrop, 1] * cfc11adj[ig]
                + wx[:laytrop, 2] * cfc12[ig]
            )

            fracs[ns06 + ig, :laytrop] = fracrefa[ig]

        #  --- ...  upper atmosphere loop
        #           nothing important goes on above laytrop in this band.

        for ig in range(ng06):
            taug[ns06 + ig, laytrop:nlay] = (
                wx[laytrop:nlay, 1] * cfc11adj[ig] + wx[laytrop:nlay, 2] * cfc12[ig]
            )
            fracs[ns06 + ig, laytrop:nlay] = fracrefa[ig]

        return taug, fracs

    # Band 7:  980-1080 cm-1 (low key - h2o,o3; low minor - co2)
    #                        (high key - o3; high minor - co2)
    def taugb07(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 7:  980-1080 cm-1 (low key - h2o,o3; low minor - co2)       !
        #                            (high key - o3; high minor - co2)         !
        #  ------------------------------------------------------------------  !

        #  --- ...  minor gas mapping level :
        #     lower - co2, p = 706.2620 mbar, t= 278.94 k
        #     upper - co2, p = 12.9350 mbar, t = 234.01 k

        #  --- ...  calculate reference ratio to be used in calculation of Planck
        #           fraction in lower atmosphere.

        dsc = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))
        chi_mls = dsc["chi_mls"].data

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb07_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data
        ka_mco2 = ds["ka_mco2"].data
        kb_mco2 = ds["kb_mco2"].data

        refrat_planck_a = chi_mls[0, 2] / chi_mls[2, 2]  # P = 706.2620 mb
        refrat_m_a = chi_mls[0, 2] / chi_mls[2, 2]  # P = 706.2720 mb

        #  --- ...  lower atmosphere loop
        speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 1, 0] * colamt[:laytrop, 2]
        specparm = colamt[:laytrop, 0] / speccomb
        specmult = 8.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[6] + js - 1

        speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 1, 1] * colamt[:laytrop, 2]
        specparm1 = colamt[:laytrop, 0] / speccomb1
        specmult1 = 8.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[6] + js1 - 1

        speccomb_mco2 = colamt[:laytrop, 0] + refrat_m_a * colamt[:laytrop, 2]
        specparm_mco2 = colamt[:laytrop, 0] / speccomb_mco2
        specmult_mco2 = 8.0 * np.minimum(specparm_mco2, self.oneminus)
        jmco2 = 1 + specmult_mco2.astype(np.int32) - 1
        fmco2 = specmult_mco2 % 1.0

        speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 2]
        specparm_planck = colamt[:laytrop, 0] / speccomb_planck
        specmult_planck = 8.0 * np.minimum(specparm_planck, self.oneminus)
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indm = indminor[:laytrop] - 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jplp = jpl + 1
        jmco2p = jmco2 + 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1

        #  --- ...  in atmospheres where the amount of CO2 is too great to be considered
        #           a minor species, adjust the column amount of CO2 by an empirical factor
        #           to obtain the proper contribution.

        temp = coldry[:laytrop] * chi_mls[1, jp[:laytrop]]
        ratco2 = colamt[:laytrop, 1] / temp
        adjcolco2 = np.where(
            ratco2 > 3.0, (3.0 + (ratco2 - 3.0) ** 0.79) * temp, colamt[:laytrop, 1]
        )

        p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(
            specparm > 0.875, -fs, 0
        )
        p0 = np.where(p0 == 0, 0, p0)

        p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        p40 = np.where(p40 == 0, 0, p40)

        fk00 = np.where(specparm < 0.125, p40, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

        fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
            specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
        )
        fk10 = np.where(fk10 == 0, fs, fk10)

        fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
            specparm > 0.875, p0 + p40, 0
        )
        fk20 = np.where(fk20 == 0, 0.0, fk20)

        id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
            specparm > 0.875, ind0 + 1, 0
        )
        id000 = np.where(id000 == 0, ind0, id000)

        id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
            specparm > 0.875, ind0 + 10, 0
        )
        id010 = np.where(id010 == 0, ind0 + 9, id010)

        id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
            specparm > 0.875, ind0, 0
        )
        id100 = np.where(id100 == 0, ind0 + 1, id100)

        id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
            specparm > 0.875, ind0 + 9, 0
        )
        id110 = np.where(id110 == 0, ind0 + 10, id110)

        id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
            specparm > 0.875, ind0 - 1, 0
        )
        id200 = np.where(id200 == 0, ind0, id200)

        id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
            specparm > 0.875, ind0 + 8, 0
        )
        id210 = np.where(id210 == 0, ind0, id210)

        fac000 = fk00 * fac00[:laytrop]
        fac100 = fk10 * fac00[:laytrop]
        fac200 = fk20 * fac00[:laytrop]
        fac010 = fk00 * fac10[:laytrop]
        fac110 = fk10 * fac10[:laytrop]
        fac210 = fk20 * fac10[:laytrop]

        p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
            specparm1 > 0.875, -fs1, 0
        )
        p1 = np.where(p1 == 0, 0, p1)

        p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        p41 = np.where(p41 == 0, 0, p41)

        fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

        fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
            specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
        )
        fk11 = np.where(fk11 == 0, fs1, fk11)

        fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
            specparm1 > 0.875, p1 + p41, 0
        )
        fk21 = np.where(fk21 == 0, 0.0, fk21)

        id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
            specparm1 > 0.875, ind1 + 1, 0
        )
        id001 = np.where(id001 == 0, ind1, id001)

        id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
            specparm1 > 0.875, ind1 + 10, 0
        )
        id011 = np.where(id011 == 0, ind1 + 9, id011)

        id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
            specparm1 > 0.875, ind1, 0
        )
        id101 = np.where(id101 == 0, ind1 + 1, id101)

        id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
            specparm1 > 0.875, ind1 + 9, 0
        )
        id111 = np.where(id111 == 0, ind1 + 10, id111)

        id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
            specparm1 > 0.875, ind1 - 1, 0
        )
        id201 = np.where(id201 == 0, ind1, id201)

        id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
            specparm1 > 0.875, ind1 + 8, 0
        )
        id211 = np.where(id211 == 0, ind1, id211)

        fac001 = fk01 * fac01[:laytrop]
        fac101 = fk11 * fac01[:laytrop]
        fac201 = fk21 * fac01[:laytrop]
        fac011 = fk01 * fac11[:laytrop]
        fac111 = fk11 * fac11[:laytrop]
        fac211 = fk21 * fac11[:laytrop]

        for ig in range(ng07):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )
            co2m1 = ka_mco2[ig, jmco2, indm] + fmco2 * (
                ka_mco2[ig, jmco2p, indm] - ka_mco2[ig, jmco2, indm]
            )
            co2m2 = ka_mco2[ig, jmco2, indmp] + fmco2 * (
                ka_mco2[ig, jmco2p, indmp] - ka_mco2[ig, jmco2, indmp]
            )
            absco2 = co2m1 + minorfrac[:laytrop] * (co2m2 - co2m1)

            taug[ns07 + ig, :laytrop] = (
                speccomb
                * (
                    fac000 * absa[ig, id000]
                    + fac010 * absa[ig, id010]
                    + fac100 * absa[ig, id100]
                    + fac110 * absa[ig, id110]
                    + fac200 * absa[ig, id200]
                    + fac210 * absa[ig, id210]
                )
                + speccomb1
                * (
                    fac001 * absa[ig, id001]
                    + fac011 * absa[ig, id011]
                    + fac101 * absa[ig, id101]
                    + fac111 * absa[ig, id111]
                    + fac201 * absa[ig, id201]
                    + fac211 * absa[ig, id211]
                )
                + tauself
                + taufor
                + adjcolco2 * absco2
            )

            fracs[ns07 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
                fracrefa[ig, jplp] - fracrefa[ig, jpl]
            )

        #  --- ...  upper atmosphere loop

        #  --- ...  in atmospheres where the amount of co2 is too great to be considered
        #           a minor species, adjust the column amount of co2 by an empirical factor
        #           to obtain the proper contribution.

        temp = coldry[laytrop:nlay] * chi_mls[1, jp[laytrop:nlay]]
        ratco2 = colamt[laytrop:nlay, 1] / temp
        adjcolco2 = np.where(
            ratco2 > 3.0, (2.0 + (ratco2 - 2.0) ** 0.79) * temp, colamt[laytrop:nlay, 1]
        )

        ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * self.nspb[6]
        ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * self.nspb[6]

        indm = indminor[laytrop:nlay] - 1
        indmp = indm + 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1

        for ig in range(ng07):
            absco2 = kb_mco2[ig, indm] + minorfrac[laytrop:nlay] * (
                kb_mco2[ig, indmp] - kb_mco2[ig, indm]
            )

            taug[ns07 + ig, laytrop:nlay] = (
                colamt[laytrop:nlay, 2]
                * (
                    fac00[laytrop:nlay] * absb[ig, ind0]
                    + fac10[laytrop:nlay] * absb[ig, ind0p]
                    + fac01[laytrop:nlay] * absb[ig, ind1]
                    + fac11[laytrop:nlay] * absb[ig, ind1p]
                )
                + adjcolco2 * absco2
            )

            fracs[ns07 + ig, laytrop:nlay] = fracrefb[ig]

        #  --- ...  empirical modification to code to improve stratospheric cooling rates
        #           for o3.  revised to apply weighting for g-point reduction in this band.

        taug[ns07 + 5, laytrop:nlay] = taug[ns07 + 5, laytrop:nlay] * 0.92
        taug[ns07 + 6, laytrop:nlay] = taug[ns07 + 6, laytrop:nlay] * 0.88
        taug[ns07 + 7, laytrop:nlay] = taug[ns07 + 7, laytrop:nlay] * 1.07
        taug[ns07 + 8, laytrop:nlay] = taug[ns07 + 8, laytrop:nlay] * 1.1
        taug[ns07 + 9, laytrop:nlay] = taug[ns07 + 9, laytrop:nlay] * 0.99
        taug[ns07 + 10, laytrop:nlay] = taug[ns07 + 10, laytrop:nlay] * 0.855

        return taug, fracs

    # Band 8:  1080-1180 cm-1 (low key - h2o; low minor - co2,o3,n2o)
    #                         (high key - o3; high minor - co2, n2o)
    def taugb08(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 8:  1080-1180 cm-1 (low key - h2o; low minor - co2,o3,n2o)  !
        #                             (high key - o3; high minor - co2, n2o)   !
        #  ------------------------------------------------------------------  !
        #  --- ...  minor gas mapping level:
        #     lower - co2, p = 1053.63 mb, t = 294.2 k
        #     lower - o3,  p = 317.348 mb, t = 240.77 k
        #     lower - n2o, p = 706.2720 mb, t= 278.94 k
        #     lower - cfc12,cfc11
        #     upper - co2, p = 35.1632 mb, t = 223.28 k
        #     upper - n2o, p = 8.716e-2 mb, t = 226.03 k

        dsc = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))
        chi_mls = dsc["chi_mls"].data

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb08_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data
        ka_mo3 = ds["ka_mo3"].data
        ka_mco2 = ds["ka_mco2"].data
        kb_mco2 = ds["kb_mco2"].data
        cfc12 = ds["cfc12"].data
        ka_mn2o = ds["ka_mn2o"].data
        kb_mn2o = ds["kb_mn2o"].data
        cfc22adj = ds["cfc22adj"].data

        #  --- ...  lower atmosphere loop
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[7]
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[7]

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indm = indminor[:laytrop] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1

        #  --- ...  in atmospheres where the amount of co2 is too great to be considered
        #           a minor species, adjust the column amount of co2 by an empirical factor
        #           to obtain the proper contribution.

        temp = coldry[:laytrop] * chi_mls[1, jp[:laytrop]]
        ratco2 = colamt[:laytrop, 1] / temp
        adjcolco2 = np.where(
            ratco2 > 3.0, (2.0 + (ratco2 - 2.0) ** 0.65) * temp, colamt[:laytrop, 1]
        )

        for ig in range(ng08):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )
            absco2 = ka_mco2[ig, indm] + minorfrac[:laytrop] * (
                ka_mco2[ig, indmp] - ka_mco2[ig, indm]
            )
            abso3 = ka_mo3[ig, indm] + minorfrac[:laytrop] * (
                ka_mo3[ig, indmp] - ka_mo3[ig, indm]
            )
            absn2o = ka_mn2o[ig, indm] + minorfrac[:laytrop] * (
                ka_mn2o[ig, indmp] - ka_mn2o[ig, indm]
            )

            taug[ns08 + ig, :laytrop] = (
                colamt[:laytrop, 0]
                * (
                    fac00[:laytrop] * absa[ig, ind0]
                    + fac10[:laytrop] * absa[ig, ind0p]
                    + fac01[:laytrop] * absa[ig, ind1]
                    + fac11[:laytrop] * absa[ig, ind1p]
                )
                + tauself
                + taufor
                + adjcolco2 * absco2
                + colamt[:laytrop, 2] * abso3
                + colamt[:laytrop, 3] * absn2o
                + wx[:laytrop, 2] * cfc12[ig]
                + wx[:laytrop, 3] * cfc22adj[ig]
            )

            fracs[ns08 + ig, :laytrop] = fracrefa[ig]

        #  --- ...  upper atmosphere loop

        ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * self.nspb[7]
        ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * self.nspb[7]

        indm = indminor[laytrop:nlay] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indmp = indm + 1

        #  --- ...  in atmospheres where the amount of co2 is too great to be considered
        #           a minor species, adjust the column amount of co2 by an empirical factor
        #           to obtain the proper contribution.

        temp = coldry[laytrop:nlay] * chi_mls[1, jp[laytrop:nlay]]
        ratco2 = colamt[laytrop:nlay, 1] / temp
        adjcolco2 = np.where(
            ratco2 > 3.0, (2.0 + (ratco2 - 2.0) ** 0.65) * temp, colamt[laytrop:nlay, 1]
        )

        for ig in range(ng08):
            absco2 = kb_mco2[ig, indm] + minorfrac[laytrop:nlay] * (
                kb_mco2[ig, indmp] - kb_mco2[ig, indm]
            )
            absn2o = kb_mn2o[ig, indm] + minorfrac[laytrop:nlay] * (
                kb_mn2o[ig, indmp] - kb_mn2o[ig, indm]
            )

            taug[ns08 + ig, laytrop:nlay] = (
                colamt[laytrop:nlay, 2]
                * (
                    fac00[laytrop:nlay] * absb[ig, ind0]
                    + fac10[laytrop:nlay] * absb[ig, ind0p]
                    + fac01[laytrop:nlay] * absb[ig, ind1]
                    + fac11[laytrop:nlay] * absb[ig, ind1p]
                )
                + adjcolco2 * absco2
                + colamt[laytrop:nlay, 3] * absn2o
                + wx[laytrop:nlay, 2] * cfc12[ig]
                + wx[laytrop:nlay, 3] * cfc22adj[ig]
            )

            fracs[ns08 + ig, laytrop:nlay] = fracrefb[ig]

        return taug, fracs

    # Band 9:  1180-1390 cm-1 (low key - h2o,ch4; low minor - n2o)
    #                         (high key - ch4; high minor - n2o)
    def taugb09(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 9:  1180-1390 cm-1 (low key - h2o,ch4; low minor - n2o)     !
        #                             (high key - ch4; high minor - n2o)       !
        #  ------------------------------------------------------------------  !

        #  --- ...  minor gas mapping level :
        #     lower - n2o, p = 706.272 mbar, t = 278.94 k
        #     upper - n2o, p = 95.58 mbar, t = 215.7 k

        dsc = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))
        chi_mls = dsc["chi_mls"].data

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb09_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data
        ka_mn2o = ds["ka_mn2o"].data
        kb_mn2o = ds["kb_mn2o"].data

        #  --- ...  calculate reference ratio to be used in calculation of Planck
        #           fraction in lower/upper atmosphere.

        refrat_planck_a = chi_mls[0, 8] / chi_mls[5, 8]  # P = 212 mb
        refrat_m_a = chi_mls[0, 2] / chi_mls[5, 2]  # P = 706.272 mb

        #  --- ...  lower atmosphere loop
        speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 3, 0] * colamt[:laytrop, 4]
        specparm = colamt[:laytrop, 0] / speccomb
        specmult = 8.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[8] + js - 1

        speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 3, 1] * colamt[:laytrop, 4]
        specparm1 = colamt[:laytrop, 0] / speccomb1
        specmult1 = 8.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[8] + js1 - 1

        speccomb_mn2o = colamt[:laytrop, 0] + refrat_m_a * colamt[:laytrop, 4]
        specparm_mn2o = colamt[:laytrop, 0] / speccomb_mn2o
        specmult_mn2o = 8.0 * np.minimum(specparm_mn2o, self.oneminus)
        jmn2o = 1 + specmult_mn2o.astype(np.int32) - 1
        fmn2o = specmult_mn2o % 1.0

        speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 4]
        specparm_planck = colamt[:laytrop, 0] / speccomb_planck
        specmult_planck = 8.0 * np.minimum(specparm_planck, self.oneminus)
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indm = indminor[:laytrop] - 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jplp = jpl + 1
        jmn2op = jmn2o + 1

        #  --- ...  in atmospheres where the amount of n2o is too great to be considered
        #           a minor species, adjust the column amount of n2o by an empirical factor
        #           to obtain the proper contribution.

        temp = coldry[:laytrop] * chi_mls[3, jp[:laytrop]]
        ratn2o = colamt[:laytrop, 3] / temp
        adjcoln2o = np.where(
            ratn2o > 1.5, (0.5 + (ratn2o - 0.5) ** 0.65) * temp, colamt[:laytrop, 3]
        )

        p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(
            specparm > 0.875, -fs, 0
        )
        p0 = np.where(p0 == 0, 0, p0)

        p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        p40 = np.where(p40 == 0, 0, p40)

        fk00 = np.where(specparm < 0.125, p40, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

        fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
            specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
        )
        fk10 = np.where(fk10 == 0, fs, fk10)

        fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
            specparm > 0.875, p0 + p40, 0
        )
        fk20 = np.where(fk20 == 0, 0.0, fk20)

        id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
            specparm > 0.875, ind0 + 1, 0
        )
        id000 = np.where(id000 == 0, ind0, id000)

        id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
            specparm > 0.875, ind0 + 10, 0
        )
        id010 = np.where(id010 == 0, ind0 + 9, id010)

        id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
            specparm > 0.875, ind0, 0
        )
        id100 = np.where(id100 == 0, ind0 + 1, id100)

        id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
            specparm > 0.875, ind0 + 9, 0
        )
        id110 = np.where(id110 == 0, ind0 + 10, id110)

        id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
            specparm > 0.875, ind0 - 1, 0
        )
        id200 = np.where(id200 == 0, ind0, id200)

        id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
            specparm > 0.875, ind0 + 8, 0
        )
        id210 = np.where(id210 == 0, ind0, id210)

        fac000 = fk00 * fac00[:laytrop]
        fac100 = fk10 * fac00[:laytrop]
        fac200 = fk20 * fac00[:laytrop]
        fac010 = fk00 * fac10[:laytrop]
        fac110 = fk10 * fac10[:laytrop]
        fac210 = fk20 * fac10[:laytrop]

        p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
            specparm1 > 0.875, -fs1, 0
        )
        p1 = np.where(p1 == 0, 0, p1)

        p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        p41 = np.where(p41 == 0, 0, p41)

        fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

        fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
            specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
        )
        fk11 = np.where(fk11 == 0, fs1, fk11)

        fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
            specparm1 > 0.875, p1 + p41, 0
        )
        fk21 = np.where(fk21 == 0, 0.0, fk21)

        id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
            specparm1 > 0.875, ind1 + 1, 0
        )
        id001 = np.where(id001 == 0, ind1, id001)

        id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
            specparm1 > 0.875, ind1 + 10, 0
        )
        id011 = np.where(id011 == 0, ind1 + 9, id011)

        id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
            specparm1 > 0.875, ind1, 0
        )
        id101 = np.where(id101 == 0, ind1 + 1, id101)

        id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
            specparm1 > 0.875, ind1 + 9, 0
        )
        id111 = np.where(id111 == 0, ind1 + 10, id111)

        id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
            specparm1 > 0.875, ind1 - 1, 0
        )
        id201 = np.where(id201 == 0, ind1, id201)

        id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
            specparm1 > 0.875, ind1 + 8, 0
        )
        id211 = np.where(id211 == 0, ind1, id211)

        fac001 = fk01 * fac01[:laytrop]
        fac101 = fk11 * fac01[:laytrop]
        fac201 = fk21 * fac01[:laytrop]
        fac011 = fk01 * fac11[:laytrop]
        fac111 = fk11 * fac11[:laytrop]
        fac211 = fk21 * fac11[:laytrop]

        for ig in range(ng09):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )
            n2om1 = ka_mn2o[ig, jmn2o, indm] + fmn2o * (
                ka_mn2o[ig, jmn2op, indm] - ka_mn2o[ig, jmn2o, indm]
            )
            n2om2 = ka_mn2o[ig, jmn2o, indmp] + fmn2o * (
                ka_mn2o[ig, jmn2op, indmp] - ka_mn2o[ig, jmn2o, indmp]
            )
            absn2o = n2om1 + minorfrac[:laytrop] * (n2om2 - n2om1)

            taug[ns09 + ig, :laytrop] = (
                speccomb
                * (
                    fac000 * absa[ig, id000]
                    + fac010 * absa[ig, id010]
                    + fac100 * absa[ig, id100]
                    + fac110 * absa[ig, id110]
                    + fac200 * absa[ig, id200]
                    + fac210 * absa[ig, id210]
                )
                + speccomb1
                * (
                    fac001 * absa[ig, id001]
                    + fac011 * absa[ig, id011]
                    + fac101 * absa[ig, id101]
                    + fac111 * absa[ig, id111]
                    + fac201 * absa[ig, id201]
                    + fac211 * absa[ig, id211]
                )
                + tauself
                + taufor
                + adjcoln2o * absn2o
            )

            fracs[ns09 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
                fracrefa[ig, jplp] - fracrefa[ig, jpl]
            )

        #  --- ...  upper atmosphere loop
        ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * self.nspb[8]
        ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * self.nspb[8]

        indm = indminor[laytrop:nlay] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indmp = indm + 1

        #  --- ...  in atmospheres where the amount of n2o is too great to be considered
        #           a minor species, adjust the column amount of n2o by an empirical factor
        #           to obtain the proper contribution.

        temp = coldry[laytrop:nlay] * chi_mls[3, jp[laytrop:nlay]]
        ratn2o = colamt[laytrop:nlay, 3] / temp
        adjcoln2o = np.where(
            ratn2o > 1.5, (0.5 + (ratn2o - 0.5) ** 0.65) * temp, colamt[laytrop:nlay, 3]
        )

        for ig in range(ng09):
            absn2o = kb_mn2o[ig, indm] + minorfrac[laytrop:nlay] * (
                kb_mn2o[ig, indmp] - kb_mn2o[ig, indm]
            )

            taug[ns09 + ig, laytrop:nlay] = (
                colamt[laytrop:nlay, 4]
                * (
                    fac00[laytrop:nlay] * absb[ig, ind0]
                    + fac10[laytrop:nlay] * absb[ig, ind0p]
                    + fac01[laytrop:nlay] * absb[ig, ind1]
                    + fac11[laytrop:nlay] * absb[ig, ind1p]
                )
                + adjcoln2o * absn2o
            )

            fracs[ns09 + ig, laytrop:nlay] = fracrefb[ig]

        return taug, fracs

    # Band 10:  1390-1480 cm-1 (low key - h2o; high key - h2o)
    def taugb10(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 10:  1390-1480 cm-1 (low key - h2o; high key - h2o)         !
        #  ------------------------------------------------------------------  !

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb10_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data

        #  --- ...  lower atmosphere loop
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[9]
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[9]

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1

        for ig in range(ng10):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )

            taug[ns10 + ig, :laytrop] = (
                colamt[:laytrop, 0]
                * (
                    fac00[:laytrop] * absa[ig, ind0]
                    + fac10[:laytrop] * absa[ig, ind0p]
                    + fac01[:laytrop] * absa[ig, ind1]
                    + fac11[:laytrop] * absa[ig, ind1p]
                )
                + tauself
                + taufor
            )

            fracs[ns10 + ig, :laytrop] = fracrefa[ig]

        #  --- ...  upper atmosphere loop

        ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * self.nspb[9]
        ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * self.nspb[9]

        indf = indfor[laytrop:nlay] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indfp = indf + 1

        for ig in range(ng10):
            taufor = forfac[laytrop:nlay] * (
                forref[ig, indf]
                + forfrac[laytrop:nlay] * (forref[ig, indfp] - forref[ig, indf])
            )

            taug[ns10 + ig, laytrop:nlay] = (
                colamt[laytrop:nlay, 0]
                * (
                    fac00[laytrop:nlay] * absb[ig, ind0]
                    + fac10[laytrop:nlay] * absb[ig, ind0p]
                    + fac01[laytrop:nlay] * absb[ig, ind1]
                    + fac11[laytrop:nlay] * absb[ig, ind1p]
                )
                + taufor
            )

            fracs[ns10 + ig, laytrop:nlay] = fracrefb[ig]

        return taug, fracs

    # Band 11:  1480-1800 cm-1 (low - h2o; low minor - o2)
    #                          (high key - h2o; high minor - o2)
    def taugb11(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 11:  1480-1800 cm-1 (low - h2o; low minor - o2)             !
        #                              (high key - h2o; high minor - o2)       !
        #  ------------------------------------------------------------------  !

        #  --- ...  minor gas mapping level :
        #     lower - o2, p = 706.2720 mbar, t = 278.94 k
        #     upper - o2, p = 4.758820 mbarm t = 250.85 k

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb11_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data
        ka_mo2 = ds["ka_mo2"].data
        kb_mo2 = ds["kb_mo2"].data

        #  --- ...  lower atmosphere loop
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[10]
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[10]

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indm = indminor[:laytrop] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1

        scaleo2 = colamt[:laytrop, 5] * scaleminor[:laytrop]

        for ig in range(ng11):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )
            tauo2 = scaleo2 * (
                ka_mo2[ig, indm]
                + minorfrac[:laytrop] * (ka_mo2[ig, indmp] - ka_mo2[ig, indm])
            )

            taug[ns11 + ig, :laytrop] = (
                colamt[:laytrop, 0]
                * (
                    fac00[:laytrop] * absa[ig, ind0]
                    + fac10[:laytrop] * absa[ig, ind0p]
                    + fac01[:laytrop] * absa[ig, ind1]
                    + fac11[:laytrop] * absa[ig, ind1p]
                )
                + tauself
                + taufor
                + tauo2
            )

            fracs[ns11 + ig, :laytrop] = fracrefa[ig]

        #  --- ...  upper atmosphere loop
        ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * self.nspb[10]
        ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * self.nspb[10]

        indf = indfor[laytrop:nlay] - 1
        indm = indminor[laytrop:nlay] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indfp = indf + 1
        indmp = indm + 1

        scaleo2 = colamt[laytrop:nlay, 5] * scaleminor[laytrop:nlay]

        for ig in range(ng11):
            taufor = forfac[laytrop:nlay] * (
                forref[ig, indf]
                + forfrac[laytrop:nlay] * (forref[ig, indfp] - forref[ig, indf])
            )
            tauo2 = scaleo2 * (
                kb_mo2[ig, indm]
                + minorfrac[laytrop:nlay] * (kb_mo2[ig, indmp] - kb_mo2[ig, indm])
            )

            taug[ns11 + ig, laytrop:nlay] = (
                colamt[laytrop:nlay, 0]
                * (
                    fac00[laytrop:nlay] * absb[ig, ind0]
                    + fac10[laytrop:nlay] * absb[ig, ind0p]
                    + fac01[laytrop:nlay] * absb[ig, ind1]
                    + fac11[laytrop:nlay] * absb[ig, ind1p]
                )
                + taufor
                + tauo2
            )

            fracs[ns11 + ig, laytrop:nlay] = fracrefb[ig]

        return taug, fracs

    # Band 12:  1800-2080 cm-1 (low - h2o,co2; high - nothing)
    def taugb12(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 12:  1800-2080 cm-1 (low - h2o,co2; high - nothing)         !
        #  ------------------------------------------------------------------  !

        dsc = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))
        chi_mls = dsc["chi_mls"].data

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb12_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        fracrefa = ds["fracrefa"].data

        #  --- ...  calculate reference ratio to be used in calculation of Planck
        #           fraction in lower/upper atmosphere.

        refrat_planck_a = chi_mls[0, 9] / chi_mls[1, 9]  # P =   174.164 mb

        #  --- ...  lower atmosphere loop

        speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 0] * colamt[:laytrop, 1]
        specparm = colamt[:laytrop, 0] / speccomb
        specmult = 8.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[11] + js - 1

        speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 1] * colamt[:laytrop, 1]
        specparm1 = colamt[:laytrop, 0] / speccomb1
        specmult1 = 8.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[11] + js1 - 1

        speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 1]
        specparm_planck = colamt[:laytrop, 0] / speccomb_planck
        specparm_planck = np.where(
            specparm_planck >= self.oneminus, self.oneminus, specparm_planck
        )
        specmult_planck = 8.0 * specparm_planck
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indsp = inds + 1
        indfp = indf + 1
        jplp = jpl + 1

        p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(
            specparm > 0.875, -fs, 0
        )
        p0 = np.where(p0 == 0, 0, p0)

        p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        p40 = np.where(p40 == 0, 0, p40)

        fk00 = np.where(specparm < 0.125, p40, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

        fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
            specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
        )
        fk10 = np.where(fk10 == 0, fs, fk10)

        fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
            specparm > 0.875, p0 + p40, 0
        )
        fk20 = np.where(fk20 == 0, 0.0, fk20)

        id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
            specparm > 0.875, ind0 + 1, 0
        )
        id000 = np.where(id000 == 0, ind0, id000)

        id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
            specparm > 0.875, ind0 + 10, 0
        )
        id010 = np.where(id010 == 0, ind0 + 9, id010)

        id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
            specparm > 0.875, ind0, 0
        )
        id100 = np.where(id100 == 0, ind0 + 1, id100)

        id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
            specparm > 0.875, ind0 + 9, 0
        )
        id110 = np.where(id110 == 0, ind0 + 10, id110)

        id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
            specparm > 0.875, ind0 - 1, 0
        )
        id200 = np.where(id200 == 0, ind0, id200)

        id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
            specparm > 0.875, ind0 + 8, 0
        )
        id210 = np.where(id210 == 0, ind0, id210)

        fac000 = fk00 * fac00[:laytrop]
        fac100 = fk10 * fac00[:laytrop]
        fac200 = fk20 * fac00[:laytrop]
        fac010 = fk00 * fac10[:laytrop]
        fac110 = fk10 * fac10[:laytrop]
        fac210 = fk20 * fac10[:laytrop]

        p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
            specparm1 > 0.875, -fs1, 0
        )
        p1 = np.where(p1 == 0, 0, p1)

        p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        p41 = np.where(p41 == 0, 0, p41)

        fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

        fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
            specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
        )
        fk11 = np.where(fk11 == 0, fs1, fk11)

        fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
            specparm1 > 0.875, p1 + p41, 0
        )
        fk21 = np.where(fk21 == 0, 0.0, fk21)

        id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
            specparm1 > 0.875, ind1 + 1, 0
        )
        id001 = np.where(id001 == 0, ind1, id001)

        id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
            specparm1 > 0.875, ind1 + 10, 0
        )
        id011 = np.where(id011 == 0, ind1 + 9, id011)

        id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
            specparm1 > 0.875, ind1, 0
        )
        id101 = np.where(id101 == 0, ind1 + 1, id101)

        id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
            specparm1 > 0.875, ind1 + 9, 0
        )
        id111 = np.where(id111 == 0, ind1 + 10, id111)

        id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
            specparm1 > 0.875, ind1 - 1, 0
        )
        id201 = np.where(id201 == 0, ind1, id201)

        id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
            specparm1 > 0.875, ind1 + 8, 0
        )
        id211 = np.where(id211 == 0, ind1, id211)

        fac001 = fk01 * fac01[:laytrop]
        fac101 = fk11 * fac01[:laytrop]
        fac201 = fk21 * fac01[:laytrop]
        fac011 = fk01 * fac11[:laytrop]
        fac111 = fk11 * fac11[:laytrop]
        fac211 = fk21 * fac11[:laytrop]

        for ig in range(ng12):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )

            taug[ns12 + ig, :laytrop] = (
                speccomb
                * (
                    fac000 * absa[ig, id000]
                    + fac010 * absa[ig, id010]
                    + fac100 * absa[ig, id100]
                    + fac110 * absa[ig, id110]
                    + fac200 * absa[ig, id200]
                    + fac210 * absa[ig, id210]
                )
                + speccomb1
                * (
                    fac001 * absa[ig, id001]
                    + fac011 * absa[ig, id011]
                    + fac101 * absa[ig, id101]
                    + fac111 * absa[ig, id111]
                    + fac201 * absa[ig, id201]
                    + fac211 * absa[ig, id211]
                )
                + tauself
                + taufor
            )

            fracs[ns12 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
                fracrefa[ig, jplp] - fracrefa[ig, jpl]
            )

        #  --- ...  upper atmosphere loop
        for ig in range(ng12):
            taug[ns12 + ig, laytrop:nlay] = 0.0
            fracs[ns12 + ig, laytrop:nlay] = 0.0

        return taug, fracs

    # Band 13:  2080-2250 cm-1 (low key-h2o,n2o; high minor-o3 minor)
    def taugb13(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 13:  2080-2250 cm-1 (low key-h2o,n2o; high minor-o3 minor)  !
        #  ------------------------------------------------------------------  !

        #  --- ...  minor gas mapping levels :
        #     lower - co2, p = 1053.63 mb, t = 294.2 k
        #     lower - co, p = 706 mb, t = 278.94 k
        #     upper - o3, p = 95.5835 mb, t = 215.7 k

        dsc = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))
        chi_mls = dsc["chi_mls"].data

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb13_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data
        ka_mco2 = ds["ka_mco2"].data
        ka_mco = ds["ka_mco"].data
        kb_mo3 = ds["kb_mo3"].data

        #  --- ...  calculate reference ratio to be used in calculation of Planck
        #           fraction in lower/upper atmosphere.

        refrat_planck_a = chi_mls[0, 4] / chi_mls[3, 4]  # P = 473.420 mb (Level 5)
        refrat_m_a = chi_mls[0, 0] / chi_mls[3, 0]  # P = 1053. (Level 1)
        refrat_m_a3 = chi_mls[0, 2] / chi_mls[3, 2]  # P = 706. (Level 3)

        #  --- ...  lower atmosphere loop

        speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 2, 0] * colamt[:laytrop, 3]
        specparm = colamt[:laytrop, 0] / speccomb
        specmult = 8.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[12] + js - 1

        speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 2, 1] * colamt[:laytrop, 3]
        specparm1 = colamt[:laytrop, 0] / speccomb1
        specmult1 = 8.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[12] + js1 - 1

        speccomb_mco2 = colamt[:laytrop, 0] + refrat_m_a * colamt[:laytrop, 3]
        specparm_mco2 = colamt[:laytrop, 0] / speccomb_mco2
        specmult_mco2 = 8.0 * np.minimum(specparm_mco2, self.oneminus)
        jmco2 = 1 + specmult_mco2.astype(np.int32) - 1
        fmco2 = specmult_mco2 % 1.0

        #  --- ...  in atmospheres where the amount of co2 is too great to be considered
        #           a minor species, adjust the column amount of co2 by an empirical factor
        #           to obtain the proper contribution.

        speccomb_mco = colamt[:laytrop, 0] + refrat_m_a3 * colamt[:laytrop, 3]
        specparm_mco = colamt[:laytrop, 0] / speccomb_mco
        specmult_mco = 8.0 * np.minimum(specparm_mco, self.oneminus)
        jmco = 1 + specmult_mco.astype(np.int32) - 1
        fmco = specmult_mco % 1.0

        speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 3]
        specparm_planck = colamt[:laytrop, 0] / speccomb_planck
        specmult_planck = 8.0 * np.minimum(specparm_planck, self.oneminus)
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indm = indminor[:laytrop] - 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jplp = jpl + 1
        jmco2p = jmco2 + 1
        jmcop = jmco + 1

        #  --- ...  in atmospheres where the amount of co2 is too great to be considered
        #           a minor species, adjust the column amount of co2 by an empirical factor
        #           to obtain the proper contribution.

        temp = coldry[:laytrop] * 3.55e-4
        ratco2 = colamt[:laytrop, 1] / temp
        adjcolco2 = np.where(
            ratco2 > 3.0, (2.0 + (ratco2 - 2.0) ** 0.68) * temp, colamt[:laytrop, 1]
        )

        p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(
            specparm > 0.875, -fs, 0
        )
        p0 = np.where(p0 == 0, 0, p0)

        p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        p40 = np.where(p40 == 0, 0, p40)

        fk00 = np.where(specparm < 0.125, p40, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

        fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
            specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
        )
        fk10 = np.where(fk10 == 0, fs, fk10)

        fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
            specparm > 0.875, p0 + p40, 0
        )
        fk20 = np.where(fk20 == 0, 0.0, fk20)

        id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
            specparm > 0.875, ind0 + 1, 0
        )
        id000 = np.where(id000 == 0, ind0, id000)

        id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
            specparm > 0.875, ind0 + 10, 0
        )
        id010 = np.where(id010 == 0, ind0 + 9, id010)

        id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
            specparm > 0.875, ind0, 0
        )
        id100 = np.where(id100 == 0, ind0 + 1, id100)

        id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
            specparm > 0.875, ind0 + 9, 0
        )
        id110 = np.where(id110 == 0, ind0 + 10, id110)

        id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
            specparm > 0.875, ind0 - 1, 0
        )
        id200 = np.where(id200 == 0, ind0, id200)

        id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
            specparm > 0.875, ind0 + 8, 0
        )
        id210 = np.where(id210 == 0, ind0, id210)

        fac000 = fk00 * fac00[:laytrop]
        fac100 = fk10 * fac00[:laytrop]
        fac200 = fk20 * fac00[:laytrop]
        fac010 = fk00 * fac10[:laytrop]
        fac110 = fk10 * fac10[:laytrop]
        fac210 = fk20 * fac10[:laytrop]

        p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
            specparm1 > 0.875, -fs1, 0
        )
        p1 = np.where(p1 == 0, 0, p1)

        p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        p41 = np.where(p41 == 0, 0, p41)

        fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

        fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
            specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
        )
        fk11 = np.where(fk11 == 0, fs1, fk11)

        fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
            specparm1 > 0.875, p1 + p41, 0
        )
        fk21 = np.where(fk21 == 0, 0.0, fk21)

        id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
            specparm1 > 0.875, ind1 + 1, 0
        )
        id001 = np.where(id001 == 0, ind1, id001)

        id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
            specparm1 > 0.875, ind1 + 10, 0
        )
        id011 = np.where(id011 == 0, ind1 + 9, id011)

        id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
            specparm1 > 0.875, ind1, 0
        )
        id101 = np.where(id101 == 0, ind1 + 1, id101)

        id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
            specparm1 > 0.875, ind1 + 9, 0
        )
        id111 = np.where(id111 == 0, ind1 + 10, id111)

        id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
            specparm1 > 0.875, ind1 - 1, 0
        )
        id201 = np.where(id201 == 0, ind1, id201)

        id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
            specparm1 > 0.875, ind1 + 8, 0
        )
        id211 = np.where(id211 == 0, ind1, id211)

        fac001 = fk01 * fac01[:laytrop]
        fac101 = fk11 * fac01[:laytrop]
        fac201 = fk21 * fac01[:laytrop]
        fac011 = fk01 * fac11[:laytrop]
        fac111 = fk11 * fac11[:laytrop]
        fac211 = fk21 * fac11[:laytrop]

        for ig in range(ng13):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )
            co2m1 = ka_mco2[ig, jmco2, indm] + fmco2 * (
                ka_mco2[ig, jmco2p, indm] - ka_mco2[ig, jmco2, indm]
            )
            co2m2 = ka_mco2[ig, jmco2, indmp] + fmco2 * (
                ka_mco2[ig, jmco2p, indmp] - ka_mco2[ig, jmco2, indmp]
            )
            absco2 = co2m1 + minorfrac[:laytrop] * (co2m2 - co2m1)
            com1 = ka_mco[ig, jmco, indm] + fmco * (
                ka_mco[ig, jmcop, indm] - ka_mco[ig, jmco, indm]
            )
            com2 = ka_mco[ig, jmco, indmp] + fmco * (
                ka_mco[ig, jmcop, indmp] - ka_mco[ig, jmco, indmp]
            )
            absco = com1 + minorfrac[:laytrop] * (com2 - com1)

            taug[ns13 + ig, :laytrop] = (
                speccomb
                * (
                    fac000 * absa[ig, id000]
                    + fac010 * absa[ig, id010]
                    + fac100 * absa[ig, id100]
                    + fac110 * absa[ig, id110]
                    + fac200 * absa[ig, id200]
                    + fac210 * absa[ig, id210]
                )
                + speccomb1
                * (
                    fac001 * absa[ig, id001]
                    + fac011 * absa[ig, id011]
                    + fac101 * absa[ig, id101]
                    + fac111 * absa[ig, id111]
                    + fac201 * absa[ig, id201]
                    + fac211 * absa[ig, id211]
                )
                + tauself
                + taufor
                + adjcolco2 * absco2
                + colamt[:laytrop, 6] * absco
            )

            fracs[ns13 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
                fracrefa[ig, jplp] - fracrefa[ig, jpl]
            )

        #  --- ...  upper atmosphere loop
        indm = indminor[laytrop:nlay] - 1
        indmp = indm + 1

        for ig in range(ng13):
            abso3 = kb_mo3[ig, indm] + minorfrac[laytrop:nlay] * (
                kb_mo3[ig, indmp] - kb_mo3[ig, indm]
            )

            taug[ns13 + ig, laytrop:nlay] = colamt[laytrop:nlay, 2] * abso3

            fracs[ns13 + ig, laytrop:nlay] = fracrefb[ig]

        return taug, fracs, taufor

    # Band 14:  2250-2380 cm-1 (low - co2; high - co2)
    def taugb14(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
        taufor,
    ):
        #  ------------------------------------------------------------------  !
        #     band 14:  2250-2380 cm-1 (low - co2; high - co2)                 !
        #  ------------------------------------------------------------------  !

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb14_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data

        #  --- ...  lower atmosphere loop

        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[13]
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[13]

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        ind0p = ind0 + 1
        ind1p = ind1 + 1
        indsp = inds + 1
        indfp = indf + 1

        for ig in range(ng14):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )

            taug[ns14 + ig, :laytrop] = (
                colamt[:laytrop, 1]
                * (
                    fac00[:laytrop] * absa[ig, ind0]
                    + fac10[:laytrop] * absa[ig, ind0p]
                    + fac01[:laytrop] * absa[ig, ind1]
                    + fac11[:laytrop] * absa[ig, ind1p]
                )
                + tauself
                + taufor
            )

            fracs[ns14 + ig, :laytrop] = fracrefa[ig]

        #  --- ...  upper atmosphere loop

        ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * self.nspb[13]
        ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * self.nspb[13]

        ind0p = ind0 + 1
        ind1p = ind1 + 1

        for ig in range(ng14):
            taug[ns14 + ig, laytrop:nlay] = colamt[laytrop:nlay, 1] * (
                fac00[laytrop:nlay] * absb[ig, ind0]
                + fac10[laytrop:nlay] * absb[ig, ind0p]
                + fac01[laytrop:nlay] * absb[ig, ind1]
                + fac11[laytrop:nlay] * absb[ig, ind1p]
            )

            fracs[ns14 + ig, laytrop:nlay] = fracrefb[ig]

        return taug, fracs

    # Band 15:  2380-2600 cm-1 (low - n2o,co2; low minor - n2)
    #                          (high - nothing)
    def taugb15(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 15:  2380-2600 cm-1 (low - n2o,co2; low minor - n2)         !
        #                              (high - nothing)                        !
        #  ------------------------------------------------------------------  !

        #  --- ...  minor gas mapping level :
        #     lower - nitrogen continuum, P = 1053., T = 294.

        dsc = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))
        chi_mls = dsc["chi_mls"].data

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb15_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        fracrefa = ds["fracrefa"].data
        ka_mn2 = ds["ka_mn2"].data

        #  --- ...  calculate reference ratio to be used in calculation of Planck
        #           fraction in lower atmosphere.

        refrat_planck_a = chi_mls[3, 0] / chi_mls[1, 0]  # P = 1053. mb (Level 1)
        refrat_m_a = chi_mls[3, 0] / chi_mls[1, 0]  # P = 1053. mb

        #  --- ...  lower atmosphere loop
        speccomb = colamt[:laytrop, 3] + rfrate[:laytrop, 4, 0] * colamt[:laytrop, 1]
        specparm = colamt[:laytrop, 3] / speccomb
        specmult = 8.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[14] + js - 1

        speccomb1 = colamt[:laytrop, 3] + rfrate[:laytrop, 4, 1] * colamt[:laytrop, 1]
        specparm1 = colamt[:laytrop, 3] / speccomb1
        specmult1 = 8.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[14] + js1 - 1

        speccomb_mn2 = colamt[:laytrop, 3] + refrat_m_a * colamt[:laytrop, 1]
        specparm_mn2 = colamt[:laytrop, 3] / speccomb_mn2
        specmult_mn2 = 8.0 * np.minimum(specparm_mn2, self.oneminus)
        jmn2 = 1 + specmult_mn2.astype(np.int32) - 1
        fmn2 = specmult_mn2 % 1.0

        speccomb_planck = colamt[:laytrop, 3] + refrat_planck_a * colamt[:laytrop, 1]
        specparm_planck = colamt[:laytrop, 3] / speccomb_planck
        specmult_planck = 8.0 * np.minimum(specparm_planck, self.oneminus)
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0

        scalen2 = colbrd[:laytrop] * scaleminor[:laytrop]

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indm = indminor[:laytrop] - 1
        indsp = inds + 1
        indfp = indf + 1
        indmp = indm + 1
        jplp = jpl + 1
        jmn2p = jmn2 + 1

        p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(
            specparm > 0.875, -fs, 0
        )
        p0 = np.where(p0 == 0, 0, p0)

        p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        p40 = np.where(p40 == 0, 0, p40)

        fk00 = np.where(specparm < 0.125, p40, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

        fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
            specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
        )
        fk10 = np.where(fk10 == 0, fs, fk10)

        fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
            specparm > 0.875, p0 + p40, 0
        )
        fk20 = np.where(fk20 == 0, 0.0, fk20)

        id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
            specparm > 0.875, ind0 + 1, 0
        )
        id000 = np.where(id000 == 0, ind0, id000)

        id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
            specparm > 0.875, ind0 + 10, 0
        )
        id010 = np.where(id010 == 0, ind0 + 9, id010)

        id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
            specparm > 0.875, ind0, 0
        )
        id100 = np.where(id100 == 0, ind0 + 1, id100)

        id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
            specparm > 0.875, ind0 + 9, 0
        )
        id110 = np.where(id110 == 0, ind0 + 10, id110)

        id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
            specparm > 0.875, ind0 - 1, 0
        )
        id200 = np.where(id200 == 0, ind0, id200)

        id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
            specparm > 0.875, ind0 + 8, 0
        )
        id210 = np.where(id210 == 0, ind0, id210)

        fac000 = fk00 * fac00[:laytrop]
        fac100 = fk10 * fac00[:laytrop]
        fac200 = fk20 * fac00[:laytrop]
        fac010 = fk00 * fac10[:laytrop]
        fac110 = fk10 * fac10[:laytrop]
        fac210 = fk20 * fac10[:laytrop]

        p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
            specparm1 > 0.875, -fs1, 0
        )
        p1 = np.where(p1 == 0, 0, p1)

        p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        p41 = np.where(p41 == 0, 0, p41)

        fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

        fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
            specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
        )
        fk11 = np.where(fk11 == 0, fs1, fk11)

        fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
            specparm1 > 0.875, p1 + p41, 0
        )
        fk21 = np.where(fk21 == 0, 0.0, fk21)

        id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
            specparm1 > 0.875, ind1 + 1, 0
        )
        id001 = np.where(id001 == 0, ind1, id001)

        id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
            specparm1 > 0.875, ind1 + 10, 0
        )
        id011 = np.where(id011 == 0, ind1 + 9, id011)

        id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
            specparm1 > 0.875, ind1, 0
        )
        id101 = np.where(id101 == 0, ind1 + 1, id101)

        id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
            specparm1 > 0.875, ind1 + 9, 0
        )
        id111 = np.where(id111 == 0, ind1 + 10, id111)

        id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
            specparm1 > 0.875, ind1 - 1, 0
        )
        id201 = np.where(id201 == 0, ind1, id201)

        id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
            specparm1 > 0.875, ind1 + 8, 0
        )
        id211 = np.where(id211 == 0, ind1, id211)

        fac001 = fk01 * fac01[:laytrop]
        fac101 = fk11 * fac01[:laytrop]
        fac201 = fk21 * fac01[:laytrop]
        fac011 = fk01 * fac11[:laytrop]
        fac111 = fk11 * fac11[:laytrop]
        fac211 = fk21 * fac11[:laytrop]

        for ig in range(ng15):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )
            n2m1 = ka_mn2[ig, jmn2, indm] + fmn2 * (
                ka_mn2[ig, jmn2p, indm] - ka_mn2[ig, jmn2, indm]
            )
            n2m2 = ka_mn2[ig, jmn2, indmp] + fmn2 * (
                ka_mn2[ig, jmn2p, indmp] - ka_mn2[ig, jmn2, indmp]
            )
            taun2 = scalen2 * (n2m1 + minorfrac[:laytrop] * (n2m2 - n2m1))

            taug[ns15 + ig, :laytrop] = (
                speccomb
                * (
                    fac000 * absa[ig, id000]
                    + fac010 * absa[ig, id010]
                    + fac100 * absa[ig, id100]
                    + fac110 * absa[ig, id110]
                    + fac200 * absa[ig, id200]
                    + fac210 * absa[ig, id210]
                )
                + speccomb1
                * (
                    fac001 * absa[ig, id001]
                    + fac011 * absa[ig, id011]
                    + fac101 * absa[ig, id101]
                    + fac111 * absa[ig, id111]
                    + fac201 * absa[ig, id201]
                    + fac211 * absa[ig, id211]
                )
                + tauself
                + taufor
                + taun2
            )

            fracs[ns15 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
                fracrefa[ig, jplp] - fracrefa[ig, jpl]
            )

        #  --- ...  upper atmosphere loop
        for ig in range(ng15):
            taug[ns15 + ig, laytrop:nlay] = 0.0

            fracs[ns15 + ig, laytrop:nlay] = 0.0

        return taug, fracs

    # Band 16:  2600-3250 cm-1 (low key- h2o,ch4; high key - ch4)
    def taugb16(
        self,
        laytrop,
        pavel,
        coldry,
        colamt,
        colbrd,
        wx,
        tauaer,
        rfrate,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        selffac,
        selffrac,
        indself,
        forfac,
        forfrac,
        indfor,
        minorfrac,
        scaleminor,
        scaleminorn2,
        indminor,
        nlay,
        taug,
        fracs,
    ):
        #  ------------------------------------------------------------------  !
        #     band 16:  2600-3250 cm-1 (low key- h2o,ch4; high key - ch4)      !
        #  ------------------------------------------------------------------  !

        dsc = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_ref_data.nc"))
        chi_mls = dsc["chi_mls"].data

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radlw_kgb16_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        fracrefa = ds["fracrefa"].data
        fracrefb = ds["fracrefb"].data

        #  --- ...  calculate reference ratio to be used in calculation of Planck
        #           fraction in lower atmosphere.

        refrat_planck_a = chi_mls[0, 5] / chi_mls[5, 5]  # P = 387. mb (Level 6)

        #  --- ...  lower atmosphere loop
        speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 3, 0] * colamt[:laytrop, 4]
        specparm = colamt[:laytrop, 0] / speccomb
        specmult = 8.0 * np.minimum(specparm, self.oneminus)
        js = 1 + specmult.astype(np.int32)
        fs = specmult % 1.0
        ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * self.nspa[15] + js - 1

        speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 3, 1] * colamt[:laytrop, 4]
        specparm1 = colamt[:laytrop, 0] / speccomb1
        specmult1 = 8.0 * np.minimum(specparm1, self.oneminus)
        js1 = 1 + specmult1.astype(np.int32)
        fs1 = specmult1 % 1.0
        ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * self.nspa[15] + js1 - 1

        speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 4]
        specparm_planck = colamt[:laytrop, 0] / speccomb_planck
        specmult_planck = 8.0 * np.minimum(specparm_planck, self.oneminus)
        jpl = 1 + specmult_planck.astype(np.int32) - 1
        fpl = specmult_planck % 1.0

        inds = indself[:laytrop] - 1
        indf = indfor[:laytrop] - 1
        indsp = inds + 1
        indfp = indf + 1
        jplp = jpl + 1

        p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(
            specparm > 0.875, -fs, 0
        )
        p0 = np.where(p0 == 0, 0, p0)

        p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        p40 = np.where(p40 == 0, 0, p40)

        fk00 = np.where(specparm < 0.125, p40, 0) + np.where(
            specparm > 0.875, p0 ** 4, 0
        )
        fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

        fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
            specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
        )
        fk10 = np.where(fk10 == 0, fs, fk10)

        fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
            specparm > 0.875, p0 + p40, 0
        )
        fk20 = np.where(fk20 == 0, 0.0, fk20)

        id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
            specparm > 0.875, ind0 + 1, 0
        )
        id000 = np.where(id000 == 0, ind0, id000)

        id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
            specparm > 0.875, ind0 + 10, 0
        )
        id010 = np.where(id010 == 0, ind0 + 9, id010)

        id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
            specparm > 0.875, ind0, 0
        )
        id100 = np.where(id100 == 0, ind0 + 1, id100)

        id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
            specparm > 0.875, ind0 + 9, 0
        )
        id110 = np.where(id110 == 0, ind0 + 10, id110)

        id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
            specparm > 0.875, ind0 - 1, 0
        )
        id200 = np.where(id200 == 0, ind0, id200)

        id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
            specparm > 0.875, ind0 + 8, 0
        )
        id210 = np.where(id210 == 0, ind0, id210)

        fac000 = fk00 * fac00[:laytrop]
        fac100 = fk10 * fac00[:laytrop]
        fac200 = fk20 * fac00[:laytrop]
        fac010 = fk00 * fac10[:laytrop]
        fac110 = fk10 * fac10[:laytrop]
        fac210 = fk20 * fac10[:laytrop]

        p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
            specparm1 > 0.875, -fs1, 0
        )
        p1 = np.where(p1 == 0, 0, p1)

        p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        p41 = np.where(p41 == 0, 0, p41)

        fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(
            specparm1 > 0.875, p1 ** 4, 0
        )
        fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

        fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
            specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
        )
        fk11 = np.where(fk11 == 0, fs1, fk11)

        fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
            specparm1 > 0.875, p1 + p41, 0
        )
        fk21 = np.where(fk21 == 0, 0.0, fk21)

        id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
            specparm1 > 0.875, ind1 + 1, 0
        )
        id001 = np.where(id001 == 0, ind1, id001)

        id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
            specparm1 > 0.875, ind1 + 10, 0
        )
        id011 = np.where(id011 == 0, ind1 + 9, id011)

        id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
            specparm1 > 0.875, ind1, 0
        )
        id101 = np.where(id101 == 0, ind1 + 1, id101)

        id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
            specparm1 > 0.875, ind1 + 9, 0
        )
        id111 = np.where(id111 == 0, ind1 + 10, id111)

        id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
            specparm1 > 0.875, ind1 - 1, 0
        )
        id201 = np.where(id201 == 0, ind1, id201)

        id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
            specparm1 > 0.875, ind1 + 8, 0
        )
        id211 = np.where(id211 == 0, ind1, id211)

        fac001 = fk01 * fac01[:laytrop]
        fac101 = fk11 * fac01[:laytrop]
        fac201 = fk21 * fac01[:laytrop]
        fac011 = fk01 * fac11[:laytrop]
        fac111 = fk11 * fac11[:laytrop]
        fac211 = fk21 * fac11[:laytrop]

        for ig in range(ng16):
            tauself = selffac[:laytrop] * (
                selfref[ig, inds]
                + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
            )
            taufor = forfac[:laytrop] * (
                forref[ig, indf]
                + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
            )

            taug[ns16 + ig, :laytrop] = (
                speccomb
                * (
                    fac000 * absa[ig, id000]
                    + fac010 * absa[ig, id010]
                    + fac100 * absa[ig, id100]
                    + fac110 * absa[ig, id110]
                    + fac200 * absa[ig, id200]
                    + fac210 * absa[ig, id210]
                )
                + speccomb1
                * (
                    fac001 * absa[ig, id001]
                    + fac011 * absa[ig, id011]
                    + fac101 * absa[ig, id101]
                    + fac111 * absa[ig, id111]
                    + fac201 * absa[ig, id201]
                    + fac211 * absa[ig, id211]
                )
                + tauself
                + taufor
            )

            fracs[ns16 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
                fracrefa[ig, jplp] - fracrefa[ig, jpl]
            )

        #  --- ...  upper atmosphere loop
        ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * self.nspb[15]
        ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * self.nspb[15]

        ind0p = ind0 + 1
        ind1p = ind1 + 1

        for ig in range(ng16):
            taug[ns16 + ig, laytrop:nlay] = colamt[laytrop:nlay, 4] * (
                fac00[laytrop:nlay] * absb[ig, ind0]
                + fac10[laytrop:nlay] * absb[ig, ind0p]
                + fac01[laytrop:nlay] * absb[ig, ind1]
                + fac11[laytrop:nlay] * absb[ig, ind1p]
            )

            fracs[ns16 + ig, laytrop:nlay] = fracrefb[ig]

        return taug, fracs

    def mcica_subcol(self, cldf, nlay, ipseed, dz, de_lgth, iplon):
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  input variables:                                                size !
        #   cldf    - real, layer cloud fraction                           nlay !
        #   nlay    - integer, number of model vertical layers               1  !
        #   ipseed  - integer, permute seed for random num generator         1  !
        #    ** note : if the cloud generator is called multiple times, need    !
        #              to permute the seed between each call; if between calls  !
        #              for lw and sw, use values differ by the number of g-pts. !
        #   dz      - real, layer thickness (km)                           nlay !
        #   de_lgth - real, layer cloud decorrelation length (km)            1  !
        #                                                                       !
        #  output variables:                                                    !
        #   lcloudy - logical, sub-colum cloud profile flag array    ngptlw*nlay!
        #                                                                       !
        #  other control flags from module variables:                           !
        #     iovrlw    : control flag for cloud overlapping method             !
        #                 =0:random; =1:maximum/random: =2:maximum; =3:decorr   !
        #                                                                       !
        #  =====================    end of definitions    ====================  !

        lcloudy = np.zeros((ngptlw, nlay), dtype=bool)
        cdfunc = np.zeros((ngptlw, nlay))
        rand1d = np.zeros(ngptlw)
        rand2d = np.zeros(nlay * ngptlw)
        fac_lcf = np.zeros(nlay)
        cdfun2 = np.zeros((ngptlw, nlay))

        #
        # ===> ...  begin here
        #
        #  --- ...  advance randum number generator by ipseed values

        #  --- ...  sub-column set up according to overlapping assumption

        if self.iovrlw == 0:
            # random overlap, pick a random value at every level
            print("Not Implemented!!")

        elif self.iovrlw == 1:  # max-ran overlap
            ds = xr.open_dataset(self.rand_file)
            rand2d = ds["rand2d"][iplon, :].data

            k1 = 0
            for n in range(ngptlw):
                for k in range(nlay):
                    cdfunc[n, k] = rand2d[k1]
                    k1 += 1

            #  ---  first pick a random number for bottom (or top) layer.
            #       then walk up the column: (aer's code)
            #       if layer below is cloudy, use the same rand num in the layer below
            #       if layer below is clear,  use a new random number

            #  ---  from bottom up
            for k in range(1, nlay):
                k1 = k - 1
                tem1 = 1.0 - cldf[k1]

                for n in range(ngptlw):
                    if cdfunc[n, k1] > tem1:
                        cdfunc[n, k] = cdfunc[n, k1]
                    else:
                        cdfunc[n, k] = cdfunc[n, k] * tem1

        elif (
            self.iovrlw == 2
        ):  # maximum overlap, pick same random numebr at every level
            print("Not Implemented!!")

        elif self.iovrlw == 3:  # decorrelation length overlap
            print("Not Implemented!!")

        #  --- ...  generate subcolumns for homogeneous clouds

        for k in range(nlay):
            tem1 = 1.0 - cldf[k]

            for n in range(ngptlw):
                lcloudy[n, k] = cdfunc[n, k] >= tem1

        return lcloudy
