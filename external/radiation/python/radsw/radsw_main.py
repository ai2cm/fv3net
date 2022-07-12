import numpy as np
import xarray as xr
import os
import sys
import warnings

sys.path.insert(0, "..")
from radsw.radsw_param import (
    ntbmx,
    nbdsw,
    ngptsw,
    maxgas,
    nbandssw,
    nbhgh,
    nblow,
    NGB,
    nspa,
    nspb,
    ng,
    ngs,
    oneminus,
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
from radphysparam import iswmode, iswrgas, iswrate, iswcice, iswcliq
from phys_const import con_amd, con_amw, con_amo3, con_g, con_cp, con_avgd
from util import compare_data
from config import *


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

    def return_initdata(self):
        outdict = {"heatfac": self.heatfac, "exp_tbl": self.exp_tbl}
        return outdict

    def swrad(
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
        sfcalb,
        dzlyr,
        delpin,
        de_lgth,
        cosz,
        solcon,
        NDAY,
        idxday,
        npts,
        nlay,
        nlp1,
        lprnt,
        lhswb,
        lhsw0,
        lflxprf,
        lfdncmp,
        sw_rand_file,
    ):

        self.lhswb = lhswb
        self.lhsw0 = lhsw0
        self.lflxprf = lflxprf
        self.lfdncmp = lfdncmp
        self.rand_file = sw_rand_file

        # outputs
        hswc = np.zeros((npts, nlay))
        cldtau = np.zeros((npts, nlay))

        upfxc_t = np.zeros(npts)
        dnfxc_t = np.zeros(npts)
        upfx0_t = np.zeros(npts)

        upfxc_s = np.zeros(npts)
        dnfxc_s = np.zeros(npts)
        upfx0_s = np.zeros(npts)
        dnfx0_s = np.zeros(npts)

        upfxc_f = np.zeros((npts, nlp1))
        dnfxc_f = np.zeros((npts, nlp1))
        upfx0_f = np.zeros((npts, nlp1))
        dnfx0_f = np.zeros((npts, nlp1))

        uvbf0 = np.zeros(npts)
        uvbfc = np.zeros(npts)
        nirbm = np.zeros(npts)
        nirdf = np.zeros(npts)
        visbm = np.zeros(npts)
        visdf = np.zeros(npts)

        hswb = np.zeros((npts, nlay, nbdsw))
        hsw0 = np.zeros((npts, nlay))

        # locals
        cldfmc = np.zeros((nlay, ngptsw))
        taug = np.zeros((nlay, ngptsw))
        taur = np.zeros((nlay, ngptsw))

        fxupc = np.zeros((nlp1, nbdsw))
        fxdnc = np.zeros((nlp1, nbdsw))
        fxup0 = np.zeros((nlp1, nbdsw))
        fxdn0 = np.zeros((nlp1, nbdsw))

        tauae = np.zeros((nlay, nbdsw))
        ssaae = np.zeros((nlay, nbdsw))
        asyae = np.zeros((nlay, nbdsw))
        taucw = np.zeros((nlay, nbdsw))
        ssacw = np.zeros((nlay, nbdsw))
        asycw = np.zeros((nlay, nbdsw))

        sfluxzen = np.zeros(ngptsw)

        cldfrc = np.zeros(nlay)
        delp = np.zeros(nlay)
        pavel = np.zeros(nlay)
        tavel = np.zeros(nlay)
        coldry = np.zeros(nlay)
        colmol = np.zeros(nlay)
        h2ovmr = np.zeros(nlay)
        o3vmr = np.zeros(nlay)
        temcol = np.zeros(nlay)
        cliqp = np.zeros(nlay)
        reliq = np.zeros(nlay)
        cicep = np.zeros(nlay)
        reice = np.zeros(nlay)
        cdat1 = np.zeros(nlay)
        cdat2 = np.zeros(nlay)
        cdat3 = np.zeros(nlay)
        cdat4 = np.zeros(nlay)
        cfrac = np.zeros(nlay)
        fac00 = np.zeros(nlay)
        fac01 = np.zeros(nlay)
        fac10 = np.zeros(nlay)
        fac11 = np.zeros(nlay)
        forfac = np.zeros(nlay)
        forfrac = np.zeros(nlay)
        selffac = np.zeros(nlay)
        selffrac = np.zeros(nlay)
        rfdelp = np.zeros(nlay)
        dz = np.zeros(nlay)

        fnet = np.zeros(nlp1)
        flxdc = np.zeros(nlp1)
        flxuc = np.zeros(nlp1)
        flxd0 = np.zeros(nlp1)
        flxu0 = np.zeros(nlp1)

        albbm = np.zeros(2)
        albdf = np.zeros(2)
        sfbmc = np.zeros(2)
        sfbm0 = np.zeros(2)
        sfdfc = np.zeros(2)
        sfdf0 = np.zeros(2)

        colamt = np.zeros((nlay, maxgas))

        ipseed = np.zeros(npts)

        indfor = np.zeros(nlay, np.int32)
        indself = np.zeros(nlay, np.int32)
        jp = np.zeros(nlay, np.int32)
        jt = np.zeros(nlay, np.int32)
        jt1 = np.zeros(nlay, np.int32)

        # Compute solar constant adjustment factor (s0fac) according to solcon.
        #      ***  s0, the solar constant at toa in w/m**2, is hard-coded with
        #           each spectra band, the total flux is about 1368.22 w/m**2.

        s0fac = solcon / self.s0

        # -# Change random number seed value for each radiation invocation
        #    (isubcsw =1 or 2).

        if self.isubcsw == 1:  # advance prescribed permutation seed
            for i in range(npts):
                ipseed[i] = self.ipsdsw0 + i + 1
        elif self.isubcsw == 2:  # use input array of permutaion seeds
            for i in range(npts):
                ipseed[i] = icseed[i]

        if lprnt:
            print(
                f"In radsw, isubcsw, ipsdsw0,ipseed = {self.isubcsw}, {self.ipsdsw0}, {ipseed}"
            )

        #  --- ...  loop over each daytime grid point

        for ipt in range(NDAY):
            j1 = idxday[ipt] - 1

            cosz1 = cosz[j1]
            sntz1 = 1.0 / cosz[j1]
            ssolar = s0fac * cosz[j1]
            if self.iovrsw == 3:
                delgth = de_lgth[j1]  # clouds decorr-length
            else:
                delgth = 0

            # -# Prepare surface albedo: bm,df - dir,dif; 1,2 - nir,uvv.
            albbm[0] = sfcalb[j1, 0]
            albdf[0] = sfcalb[j1, 1]
            albbm[1] = sfcalb[j1, 2]
            albdf[1] = sfcalb[j1, 3]

            # -# Prepare atmospheric profile for use in rrtm.
            #           the vertical index of internal array is from surface to top

            tem1 = 100.0 * con_g
            tem2 = 1.0e-20 * 1.0e3 * con_avgd

            for k in range(nlay):
                pavel[k] = plyr[j1, k]
                tavel[k] = tlyr[j1, k]
                delp[k] = delpin[j1, k]
                dz[k] = dzlyr[j1, k]

                #  --- ...  set absorber amount
                # ncep model use
                h2ovmr[k] = max(
                    0.0, qlyr[j1, k] * self.amdw / (1.0 - qlyr[j1, k])
                )  # input specific humidity
                o3vmr[k] = max(0.0, olyr[j1, k] * self.amdo3)  # input mass mixing ratio

                tem0 = (1.0 - h2ovmr[k]) * con_amd + h2ovmr[k] * con_amw
                coldry[k] = tem2 * delp[k] / (tem1 * tem0 * (1.0 + h2ovmr[k]))
                temcol[k] = 1.0e-12 * coldry[k]

                colamt[k, 0] = max(0.0, coldry[k] * h2ovmr[k])  # h2o
                colamt[k, 1] = max(temcol[k], coldry[k] * gasvmr[j1, k, 0])  # co2
                colamt[k, 2] = max(0.0, coldry[k] * o3vmr[k])  # o3
                colmol[k] = coldry[k] + colamt[k, 0]

            if lprnt:
                if ipt == 1:
                    print(f"pavel = {pavel}")
                    print(f"tavel = {tavel}")
                    print(f"delp = {delp}")
                    print(f"h2ovmr = {h2ovmr*1000}")
                    print(f"o3vmr = {o3vmr*1000000}")

            #  --- ...  set up gas column amount, convert from volume mixing ratio
            #           to molec/cm2 based on coldry (scaled to 1.0e-20)

            if iswrgas > 0:
                for k in range(nlay):
                    colamt[k, 3] = max(temcol[k], coldry[k] * gasvmr[j1, k, 1])  # n2o
                    colamt[k, 4] = max(temcol[k], coldry[k] * gasvmr[j1, k, 2])  # ch4
                    colamt[k, 5] = max(temcol[k], coldry[k] * gasvmr[j1, k, 3])  # o2
            else:
                for k in range(nlay):
                    colamt[k, 3] = temcol[k]  # n2o
                    colamt[k, 4] = temcol[k]  # ch4
                    colamt[k, 5] = temcol[k]  # o2

            #  --- ...  set aerosol optical properties

            for ib in range(nbdsw):
                for k in range(nlay):
                    tauae[k, ib] = aerosols[j1, k, ib, 0]
                    ssaae[k, ib] = aerosols[j1, k, ib, 1]
                    asyae[k, ib] = aerosols[j1, k, ib, 2]

            if iswcliq > 0:  # use prognostic cloud method
                for k in range(nlay):
                    cfrac[k] = clouds[j1, k, 0]  # cloud fraction
                    cliqp[k] = clouds[j1, k, 1]  # cloud liq path
                    reliq[k] = clouds[j1, k, 2]  # liq partical effctive radius
                    cicep[k] = clouds[j1, k, 3]  # cloud ice path
                    reice[k] = clouds[j1, k, 4]  # ice partical effctive radius
                    cdat1[k] = clouds[j1, k, 5]  # cloud rain drop path
                    cdat2[k] = clouds[j1, k, 6]  # rain partical effctive radius
                    cdat3[k] = clouds[j1, k, 7]  # cloud snow path
                    cdat4[k] = clouds[j1, k, 8]  # snow partical effctive radius
            else:  # use diagnostic cloud method
                for k in range(nlay):
                    cfrac[k] = clouds[j1, k, 0]  # cloud fraction
                    cdat1[k] = clouds[j1, k, 1]  # cloud optical depth
                    cdat2[k] = clouds[j1, k, 2]  # cloud single scattering albedo
                    cdat3[k] = clouds[j1, k, 3]  # cloud asymmetry factor

            # -# Compute fractions of clear sky view:
            #    - random overlapping
            #    - max/ran overlapping
            #    - maximum overlapping

            zcf0 = 1.0
            zcf1 = 1.0
            if self.iovrsw == 0:  # random overlapping
                for k in range(nlay):
                    zcf0 = zcf0 * (1.0 - cfrac[k])
            elif self.iovrsw == 1:  # max/ran overlapping
                for k in range(nlay):
                    if cfrac[k] > self.ftiny:  # cloudy layer
                        zcf1 = min(zcf1, 1.0 - cfrac[k])
                    elif zcf1 < 1.0:  # clear layer
                        zcf0 = zcf0 * zcf1
                        zcf1 = 1.0
                zcf0 = zcf0 * zcf1
            elif self.iovrsw >= 2:
                for k in range(nlay):
                    zcf0 = min(
                        zcf0, 1.0 - cfrac[k]
                    )  # used only as clear/cloudy indicator

            if zcf0 <= self.ftiny:
                zcf0 = 0.0
            if zcf0 > self.oneminus:
                zcf0 = 1.0
            zcf1 = 1.0 - zcf0

            # -# For cloudy sky column, call cldprop() to compute the cloud
            #    optical properties for each cloudy layer.

            if zcf1 > 0.0:  # cloudy sky column
                taucw, ssacw, asycw, cldfrc, cldfmc = self.cldprop(
                    cfrac,
                    cliqp,
                    reliq,
                    cicep,
                    reice,
                    cdat1,
                    cdat2,
                    cdat3,
                    cdat4,
                    zcf1,
                    nlay,
                    ipseed[j1],
                    dz,
                    delgth,
                    ipt,
                )

                #  --- ...  save computed layer cloud optical depth for output
                #           rrtm band 10 is approx to the 0.55 mu spectrum

                for k in range(nlay):
                    cldtau[j1, k] = taucw[k, 9]
            else:
                cldfrc[:] = 0.0
                cldfmc[:, :] = 0.0
                for i in range(nbdsw):
                    for k in range(nlay):
                        taucw[k, i] = 0.0
                        ssacw[k, i] = 0.0
                        asycw[k, i] = 0.0

            # -# Call setcoef() to compute various coefficients needed in
            #    radiative transfer calculations.
            (
                laytrop,
                jp,
                jt,
                jt1,
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
            ) = self.setcoef(pavel, tavel, h2ovmr, nlay, nlp1)

            # -# Call taumol() to calculate optical depths for gaseous absorption
            #    and rayleigh scattering
            sfluxzen, taug, taur = self.taumol(
                colamt,
                colmol,
                fac00,
                fac01,
                fac10,
                fac11,
                jp,
                jt,
                jt1,
                laytrop,
                forfac,
                forfrac,
                indfor,
                selffac,
                selffrac,
                indself,
                nlay,
            )

            # -# Call the 2-stream radiation transfer model:
            #    - if physparam::isubcsw .le.0, using standard cloud scheme,
            #      call spcvrtc().
            #    - if physparam::isubcsw .gt.0, using mcica cloud scheme,
            #      call spcvrtm().

            (
                fxupc,
                fxdnc,
                fxup0,
                fxdn0,
                ftoauc,
                ftoau0,
                ftoadc,
                fsfcuc,
                fsfcu0,
                fsfcdc,
                fsfcd0,
                sfbmc,
                sfdfc,
                sfbm0,
                sfdf0,
                suvbfc,
                suvbf0,
            ) = self.spcvrtm(
                ssolar,
                cosz1,
                sntz1,
                albbm,
                albdf,
                sfluxzen,
                cldfmc,
                zcf1,
                zcf0,
                taug,
                taur,
                tauae,
                ssaae,
                asyae,
                taucw,
                ssacw,
                asycw,
                nlay,
                nlp1,
            )

            # -# Save outputs.
            #  --- ...  sum up total spectral fluxes for total-sky

            for k in range(nlp1):
                flxuc[k] = 0.0
                flxdc[k] = 0.0

                for ib in range(nbdsw):
                    flxuc[k] = flxuc[k] + fxupc[k, ib]
                    flxdc[k] = flxdc[k] + fxdnc[k, ib]

            # --- ...  optional clear sky fluxes

            if self.lhsw0 or self.lflxprf:
                for k in range(nlp1):
                    flxu0[k] = 0.0
                    flxd0[k] = 0.0

                    for ib in range(nbdsw):
                        flxu0[k] = flxu0[k] + fxup0[k, ib]
                        flxd0[k] = flxd0[k] + fxdn0[k, ib]

            #  --- ...  prepare for final outputs
            for k in range(nlay):
                rfdelp[k] = self.heatfac / delp[k]

            if self.lfdncmp:
                # --- ...  optional uv-b surface downward flux
                uvbf0[j1] = suvbf0
                uvbfc[j1] = suvbfc

                # --- ...  optional beam and diffuse sfc fluxes
                nirbm[j1] = sfbmc[0]
                nirdf[j1] = sfdfc[0]
                visbm[j1] = sfbmc[1]
                visdf[j1] = sfdfc[1]

            #  --- ...  toa and sfc fluxes

            upfxc_t[j1] = ftoauc
            dnfxc_t[j1] = ftoadc
            upfx0_t[j1] = ftoau0

            upfxc_s[j1] = fsfcuc
            dnfxc_s[j1] = fsfcdc
            upfx0_s[j1] = fsfcu0
            dnfx0_s[j1] = fsfcd0

            #  --- ...  compute heating rates

            fnet[0] = flxdc[0] - flxuc[0]

            for k in range(1, nlp1):
                fnet[k] = flxdc[k] - flxuc[k]
                hswc[j1, k - 1] = (fnet[k] - fnet[k - 1]) * rfdelp[k - 1]

            # --- ...  optional flux profiles

            if self.lflxprf:
                for k in range(nlp1):
                    upfxc_f[j1, k] = flxuc[k]
                    dnfxc_f[j1, k] = flxdc[k]
                    upfx0_f[j1, k] = flxu0[k]
                    dnfx0_f[j1, k] = flxd0[k]

            # --- ...  optional clear sky heating rates

            if self.lhsw0:
                fnet[0] = flxd0[0] - flxu0[0]

                for k in range(1, nlp1):
                    fnet[k] = flxd0[k] - flxu0[k]
                    hsw0[j1, k - 1] = (fnet[k] - fnet[k - 1]) * rfdelp[k - 1]

            # --- ...  optional spectral band heating rates

            if self.lhswb:
                for mb in range(nbdsw):
                    fnet[0] = fxdnc[0, mb] - fxupc[0, mb]

                    for k in range(nlay):
                        fnet[k + 1] = fxdnc[k + 1, mb] - fxupc[k + 1, mb]
                        hswb[j1, k, mb] = (fnet[k + 1] - fnet[k]) * rfdelp[k]

        return (
            hswc,
            upfxc_t,
            dnfxc_t,
            upfx0_t,
            upfxc_s,
            dnfxc_s,
            upfx0_s,
            dnfx0_s,
            cldtau,
            hsw0,
            uvbf0,
            uvbfc,
            nirbm,
            nirdf,
            visbm,
            visdf,
        )

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
        cf1,
        nlay,
        ipseed,
        dz,
        delgth,
        ipt,
    ):
        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_cldprtb_data.nc"))
        extliq1 = ds["extliq1"].data
        extliq2 = ds["extliq2"].data
        ssaliq1 = ds["ssaliq1"].data
        ssaliq2 = ds["ssaliq2"].data
        asyliq1 = ds["asyliq1"].data
        asyliq2 = ds["asyliq2"].data
        extice2 = ds["extice2"].data
        ssaice2 = ds["ssaice2"].data
        asyice2 = ds["asyice2"].data
        extice3 = ds["extice3"].data
        ssaice3 = ds["ssaice3"].data
        asyice3 = ds["asyice3"].data
        abari = ds["abari"].data
        bbari = ds["bbari"].data
        cbari = ds["cbari"].data
        dbari = ds["dbari"].data
        ebari = ds["ebari"].data
        fbari = ds["fbari"].data
        b0s = ds["b0s"].data
        b1s = ds["b1s"].data
        b0r = ds["b0r"].data
        c0s = ds["c0s"].data
        c0r = ds["c0r"].data
        a0r = ds["a0r"].data
        a1r = ds["a1r"].data
        a0s = ds["a0s"].data
        a1s = ds["a1s"].data

        #  ---  outputs:
        cldfmc = np.zeros((nlay, ngptsw))
        taucw = np.zeros((nlay, nbdsw))
        ssacw = np.ones((nlay, nbdsw))
        asycw = np.zeros((nlay, nbdsw))
        cldfrc = np.zeros(nlay)

        #  ---  locals:
        tauliq = np.zeros(nbandssw)
        tauice = np.zeros(nbandssw)
        ssaliq = np.zeros(nbandssw)
        ssaice = np.zeros(nbandssw)
        ssaran = np.zeros(nbandssw)
        ssasnw = np.zeros(nbandssw)
        asyliq = np.zeros(nbandssw)
        asyice = np.zeros(nbandssw)
        asyran = np.zeros(nbandssw)
        asysnw = np.zeros(nbandssw)
        cldf = np.zeros(nlay)

        lcloudy = np.zeros((nlay, ngptsw), dtype=bool)

        # Compute cloud radiative properties for a cloudy column.

        if iswcliq > 0:

            for k in range(nlay):
                if cfrac[k] > self.ftiny:

                    #  - Compute optical properties for rain and snow.
                    #    For rain: tauran/ssaran/asyran
                    #    For snow: tausnw/ssasnw/asysnw
                    #  - Calculation of absorption coefficients due to water clouds
                    #    For water clouds: tauliq/ssaliq/asyliq
                    #  - Calculation of absorption coefficients due to ice clouds
                    #    For ice clouds: tauice/ssaice/asyice
                    #  - For Prognostic cloud scheme: sum up the cloud optical property:
                    #     \f$ taucw=tauliq+tauice+tauran+tausnw \f$
                    #     \f$ ssacw=ssaliq+ssaice+ssaran+ssasnw \f$
                    #     \f$ asycw=asyliq+asyice+asyran+asysnw \f$

                    cldran = cdat1[k]
                    cldsnw = cdat3[k]
                    refsnw = cdat4[k]
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

                    for ib in range(nbandssw):
                        ssaran[ib] = tauran * (1.0 - b0r[ib])
                        ssasnw[ib] = tausnw * (1.0 - (b0s[ib] + b1s[ib] * dgesnw))
                        asyran[ib] = ssaran[ib] * c0r[ib]
                        asysnw[ib] = ssasnw[ib] * c0s[ib]

                    cldliq = cliqp[k]
                    cldice = cicep[k]
                    refliq = reliq[k]
                    refice = reice[k]

                    #  --- ...  calculation of absorption coefficients due to water clouds.

                    if cldliq <= 0.0:
                        for ib in range(nbandssw):
                            tauliq[ib] = 0.0
                            ssaliq[ib] = 0.0
                            asyliq[ib] = 0.0
                    else:
                        factor = refliq - 1.5
                        index = max(1, min(57, int(factor))) - 1
                        fint = factor - float(index + 1)

                        if iswcliq == 1:
                            for ib in range(nbandssw):
                                extcoliq = max(
                                    0.0,
                                    extliq1[index, ib]
                                    + fint
                                    * (extliq1[index + 1, ib] - extliq1[index, ib]),
                                )
                                ssacoliq = max(
                                    0.0,
                                    min(
                                        1.0,
                                        ssaliq1[index, ib]
                                        + fint
                                        * (ssaliq1[index + 1, ib] - ssaliq1[index, ib]),
                                    ),
                                )

                                asycoliq = max(
                                    0.0,
                                    min(
                                        1.0,
                                        asyliq1[index, ib]
                                        + fint
                                        * (asyliq1[index + 1, ib] - asyliq1[index, ib]),
                                    ),
                                )

                                tauliq[ib] = cldliq * extcoliq
                                ssaliq[ib] = tauliq[ib] * ssacoliq
                                asyliq[ib] = ssaliq[ib] * asycoliq
                        elif iswcliq == 2:  # use updated coeffs
                            for ib in range(nbandssw):
                                extcoliq = max(
                                    0.0,
                                    extliq2[index, ib]
                                    + fint
                                    * (extliq2[index + 1, ib] - extliq2[index, ib]),
                                )
                                ssacoliq = max(
                                    0.0,
                                    min(
                                        1.0,
                                        ssaliq2[index, ib]
                                        + fint
                                        * (ssaliq2[index + 1, ib] - ssaliq2[index, ib]),
                                    ),
                                )

                                asycoliq = max(
                                    0.0,
                                    min(
                                        1.0,
                                        asyliq2[index, ib]
                                        + fint
                                        * (asyliq2[index + 1, ib] - asyliq2[index, ib]),
                                    ),
                                )

                                tauliq[ib] = cldliq * extcoliq
                                ssaliq[ib] = tauliq[ib] * ssacoliq
                                asyliq[ib] = ssaliq[ib] * asycoliq

                    #  --- ...  calculation of absorption coefficients due to ice clouds.

                    if cldice <= 0.0:
                        for ib in range(nbandssw):
                            tauice[ib] = 0.0
                            ssaice[ib] = 0.0
                            asyice[ib] = 0.0
                    else:

                        #  --- ...  ebert and curry approach for all particle sizes though somewhat
                        #           unjustified for large ice particles

                        if iswcice == 1:
                            refice = min(130.0, max(13.0, refice))

                            for ib in range(nbandssw):
                                ia = (
                                    self.idxebc[ib] - 1
                                )  # eb_&_c band index for ice cloud coeff

                                extcoice = max(0.0, abari[ia] + bbari[ia] / refice)
                                ssacoice = max(
                                    0.0, min(1.0, 1.0 - cbari[ia] - dbari[ia] * refice)
                                )
                                asycoice = max(
                                    0.0, min(1.0, ebari[ia] + fbari[ia] * refice)
                                )

                                tauice[ib] = cldice * extcoice
                                ssaice[ib] = tauice[ib] * ssacoice
                                asyice[ib] = ssaice[ib] * asycoice

                        #  --- ...  streamer approach for ice effective radius between 5.0 and 131.0 microns
                        elif iswcice == 2:
                            refice = min(131.0, max(5.0, refice))

                            factor = (refice - 2.0) / 3.0
                            index = max(1, min(42, int(factor))) - 1
                            fint = factor - float(index + 1)

                            for ib in range(nbandssw):
                                extcoice = max(
                                    0.0,
                                    extice2[index, ib]
                                    + fint
                                    * (extice2[index + 1, ib] - extice2[index, ib]),
                                )
                                ssacoice = max(
                                    0.0,
                                    min(
                                        1.0,
                                        ssaice2[index, ib]
                                        + fint
                                        * (ssaice2[index + 1, ib] - ssaice2[index, ib]),
                                    ),
                                )
                                asycoice = max(
                                    0.0,
                                    min(
                                        1.0,
                                        asyice2[index, ib]
                                        + fint
                                        * (asyice2[index + 1, ib] - asyice2[index, ib]),
                                    ),
                                )

                                tauice[ib] = cldice * extcoice
                                ssaice[ib] = tauice[ib] * ssacoice
                                asyice[ib] = ssaice[ib] * asycoice

                        #  --- ...  fu's approach for ice effective radius between 4.8 and 135 microns
                        #           (generalized effective size from 5 to 140 microns)
                        elif iswcice == 3:
                            dgeice = max(5.0, min(140.0, 1.0315 * refice))

                            factor = (dgeice - 2.0) / 3.0
                            index = max(1, min(45, int(factor))) - 1
                            fint = factor - float(index + 1)

                            for ib in range(nbandssw):
                                extcoice = max(
                                    0.0,
                                    extice3[index, ib]
                                    + fint
                                    * (extice3[index + 1, ib] - extice3[index, ib]),
                                )
                                ssacoice = max(
                                    0.0,
                                    min(
                                        1.0,
                                        ssaice3[index, ib]
                                        + fint
                                        * (ssaice3[index + 1, ib] - ssaice3[index, ib]),
                                    ),
                                )
                                asycoice = max(
                                    0.0,
                                    min(
                                        1.0,
                                        asyice3[index, ib]
                                        + fint
                                        * (asyice3[index + 1, ib] - asyice3[index, ib]),
                                    ),
                                )

                                tauice[ib] = cldice * extcoice
                                ssaice[ib] = tauice[ib] * ssacoice
                                asyice[ib] = ssaice[ib] * asycoice

                    for ib in range(nbdsw):
                        jb = nblow + ib - 16
                        taucw[k, ib] = tauliq[jb] + tauice[jb] + tauran + tausnw
                        ssacw[k, ib] = ssaliq[jb] + ssaice[jb] + ssaran[jb] + ssasnw[jb]
                        asycw[k, ib] = asyliq[jb] + asyice[jb] + asyran[jb] + asysnw[jb]

        else:  #  lab_if_iswcliq

            for k in range(nlay):
                if cfrac[k] > self.ftiny:
                    for ib in range(nbdsw):
                        taucw[k, ib] = cdat1[k]
                        ssacw[k, ib] = cdat1[k] * cdat2[k]
                        asycw[k, ib] = ssacw[k, ib] * cdat3[k]

        # -# if physparam::isubcsw > 0, call mcica_subcol() to distribute
        #    cloud properties to each g-point.

        if self.isubcsw > 0:  # mcica sub-col clouds approx
            cldf = cfrac
            cldf = np.where(cldf < self.ftiny, 0.0, cldf)

            #  --- ...  call sub-column cloud generator

            lcloudy = self.mcica_subcol(cldf, nlay, ipseed, dz, delgth, ipt)

            for ig in range(ngptsw):
                for k in range(nlay):
                    if lcloudy[k, ig]:
                        cldfmc[k, ig] = 1.0
                    else:
                        cldfmc[k, ig] = 0.0

        else:  # non-mcica, normalize cloud
            for k in range(nlay):
                cldfrc[k] = cfrac[k] / cf1

        return taucw, ssacw, asycw, cldfrc, cldfmc

    def mcica_subcol(self, cldf, nlay, ipseed, dz, de_lgth, ipt):

        ds = xr.open_dataset(self.rand_file)
        rand2d = ds["rand2d"][ipt, :].data

        #  ---  outputs:
        lcloudy = np.zeros((nlay, ngptsw))

        #  ---  locals:
        cdfunc = np.zeros((nlay, ngptsw))

        #  --- ...  sub-column set up according to overlapping assumption

        if self.iovrsw == 1:  # max-ran overlap

            k1 = 0
            for n in range(ngptsw):
                for k in range(nlay):
                    cdfunc[k, n] = rand2d[k1]
                    k1 = k1 + 1

            #  ---  first pick a random number for bottom/top layer.
            #       then walk up the column: (aer's code)
            #       if layer below is cloudy, use the same rand num in the layer below
            #       if layer below is clear,  use a new random number

            #  ---  from bottom up
            for k in range(1, nlay):
                k1 = k - 1
                tem1 = 1.0 - cldf[k1]

                for n in range(ngptsw):
                    if cdfunc[k1, n] > tem1:
                        cdfunc[k, n] = cdfunc[k1, n]
                    else:
                        cdfunc[k, n] = cdfunc[k, n] * tem1

        #  --- ...  generate subcolumns for homogeneous clouds

        for k in range(nlay):
            tem1 = 1.0 - cldf[k]

            for n in range(ngptsw):
                lcloudy[k, n] = cdfunc[k, n] >= tem1

        return lcloudy

    def setcoef(self, pavel, tavel, h2ovmr, nlay, nlp1):
        #  ===================  program usage description  ===================  !
        #                                                                       !
        # purpose:  compute various coefficients needed in radiative transfer   !
        #    calculations.                                                      !
        #                                                                       !
        # subprograms called:  none                                             !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                       -size- !
        #   pavel     - real, layer pressures (mb)                         nlay !
        #   tavel     - real, layer temperatures (k)                       nlay !
        #   h2ovmr    - real, layer w.v. volum mixing ratio (kg/kg)        nlay !
        #   nlay/nlp1 - integer, total number of vertical layers, levels    1   !
        #                                                                       !
        #  outputs:                                                             !
        #   laytrop   - integer, tropopause layer index (unitless)          1   !
        #   jp        - real, indices of lower reference pressure          nlay !
        #   jt, jt1   - real, indices of lower reference temperatures      nlay !
        #                 at levels of jp and jp+1                              !
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
        #                                                                       !
        #  ======================    end of definitions    ===================  !   #

        #  ---  outputs:
        indself = np.zeros(nlay, dtype=np.int32)
        indfor = np.zeros(nlay, dtype=np.int32)
        jp = np.zeros(nlay, dtype=np.int32)
        jt = np.zeros(nlay, dtype=np.int32)
        jt1 = np.zeros(nlay, dtype=np.int32)

        fac00 = np.zeros(nlay)
        fac01 = np.zeros(nlay)
        fac10 = np.zeros(nlay)
        fac11 = np.zeros(nlay)
        selffac = np.zeros(nlay)
        selffrac = np.zeros(nlay)
        forfac = np.zeros(nlay)
        forfrac = np.zeros(nlay)

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_ref_data.nc"))
        preflog = ds["preflog"].data
        tref = ds["tref"].data

        laytrop = nlay

        for k in range(nlay):

            forfac[k] = pavel[k] * self.stpfac / (tavel[k] * (1.0 + h2ovmr[k]))

            #  --- ...  find the two reference pressures on either side of the
            #           layer pressure.  store them in jp and jp1.  store in fp the
            #           fraction of the difference (in ln(pressure)) between these
            #           two values that the layer pressure lies.

            plog = np.log(pavel[k])
            jp[k] = max(1, min(58, int(36.0 - 5.0 * (plog + 0.04)))) - 1
            jp1 = jp[k] + 1
            fp = 5.0 * (preflog[jp[k]] - plog)

            #  --- ...  determine, for each reference pressure (jp and jp1), which
            #          reference temperature (these are different for each reference
            #          pressure) is nearest the layer temperature but does not exceed it.
            #          store these indices in jt and jt1, resp. store in ft (resp. ft1)
            #          the fraction of the way between jt (jt1) and the next highest
            #          reference temperature that the layer temperature falls.

            tem1 = (tavel[k] - tref[jp[k]]) / 15.0
            tem2 = (tavel[k] - tref[jp1]) / 15.0
            jt[k] = max(1, min(4, int(3.0 + tem1))) - 1
            jt1[k] = max(1, min(4, int(3.0 + tem2))) - 1
            ft = tem1 - float(jt[k] - 2)
            ft1 = tem2 - float(jt1[k] - 2)

            #  --- ...  we have now isolated the layer ln pressure and temperature,
            #           between two reference pressures and two reference temperatures
            #           (for each reference pressure).  we multiply the pressure
            #           fraction fp with the appropriate temperature fractions to get
            #           the factors that will be needed for the interpolation that yields
            #           the optical depths (performed in routines taugbn for band n).

            fp1 = 1.0 - fp
            fac10[k] = fp1 * ft
            fac00[k] = fp1 * (1.0 - ft)
            fac11[k] = fp * ft1
            fac01[k] = fp * (1.0 - ft1)

            #  --- ...  if the pressure is less than ~100mb, perform a different
            #           set of species interpolations.

            if plog > 4.56:

                laytrop = k + 1

                #  --- ...  set up factors needed to separately include the water vapor
                #           foreign-continuum in the calculation of absorption coefficient.

                tem1 = (332.0 - tavel[k]) / 36.0
                indfor[k] = min(2, max(1, int(tem1)))
                forfrac[k] = tem1 - float(indfor[k])

                #  --- ...  set up factors needed to separately include the water vapor
                #           self-continuum in the calculation of absorption coefficient.

                tem2 = (tavel[k] - 188.0) / 7.2
                indself[k] = min(9, max(1, int(tem2) - 7))
                selffrac[k] = tem2 - float(indself[k] + 7)
                selffac[k] = h2ovmr[k] * forfac[k]

            else:

                #  --- ...  set up factors needed to separately include the water vapor
                #           foreign-continuum in the calculation of absorption coefficient.

                tem1 = (tavel[k] - 188.0) / 36.0
                indfor[k] = 3
                forfrac[k] = tem1 - 1.0

                indself[k] = 0
                selffrac[k] = 0.0
                selffac[k] = 0.0

        jp += 1
        jt += 1
        jt1 += 1

        return (
            laytrop,
            jp,
            jt,
            jt1,
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
        )

    def spcvrtm(
        self,
        ssolar,
        cosz,
        sntz,
        albbm,
        albdf,
        sfluxzen,
        cldfmc,
        cf1,
        cf0,
        taug,
        taur,
        tauae,
        ssaae,
        asyae,
        taucw,
        ssacw,
        asycw,
        nlay,
        nlp1,
    ):
        #  ===================  program usage description  ===================  !
        #                                                                       !
        #   purpose:  computes the shortwave radiative fluxes using two-stream  !
        #             method of h. barker and mcica, the monte-carlo independent!
        #             column approximation, for the representation of sub-grid  !
        #             cloud variability (i.e. cloud overlap).                   !
        #                                                                       !
        #   subprograms called:  vrtqdr                                         !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                        size  !
        #    ssolar  - real, incoming solar flux at top                    1    !
        #    cosz    - real, cosine solar zenith angle                     1    !
        #    sntz    - real, secant solar zenith angle                     1    !
        #    albbm   - real, surface albedo for direct beam radiation      2    !
        #    albdf   - real, surface albedo for diffused radiation         2    !
        #    sfluxzen- real, spectral distribution of incoming solar flux ngptsw!
        #    cldfmc  - real, layer cloud fraction for g-point        nlay*ngptsw!
        #    cf1     - real, >0: cloudy sky, otherwise: clear sky          1    !
        #    cf0     - real, =1-cf1                                        1    !
        #    taug    - real, spectral optical depth for gases        nlay*ngptsw!
        #    taur    - real, optical depth for rayleigh scattering   nlay*ngptsw!
        #    tauae   - real, aerosols optical depth                  nlay*nbdsw !
        #    ssaae   - real, aerosols single scattering albedo       nlay*nbdsw !
        #    asyae   - real, aerosols asymmetry factor               nlay*nbdsw !
        #    taucw   - real, weighted cloud optical depth            nlay*nbdsw !
        #    ssacw   - real, weighted cloud single scat albedo       nlay*nbdsw !
        #    asycw   - real, weighted cloud asymmetry factor         nlay*nbdsw !
        #    nlay,nlp1 - integer,  number of layers/levels                 1    !
        #                                                                       !
        #  output variables:                                                    !
        #    fxupc   - real, tot sky upward flux                     nlp1*nbdsw !
        #    fxdnc   - real, tot sky downward flux                   nlp1*nbdsw !
        #    fxup0   - real, clr sky upward flux                     nlp1*nbdsw !
        #    fxdn0   - real, clr sky downward flux                   nlp1*nbdsw !
        #    ftoauc  - real, tot sky toa upwd flux                         1    !
        #    ftoau0  - real, clr sky toa upwd flux                         1    !
        #    ftoadc  - real, toa downward (incoming) solar flux            1    !
        #    fsfcuc  - real, tot sky sfc upwd flux                         1    !
        #    fsfcu0  - real, clr sky sfc upwd flux                         1    !
        #    fsfcdc  - real, tot sky sfc dnwd flux                         1    !
        #    fsfcd0  - real, clr sky sfc dnwd flux                         1    !
        #    sfbmc   - real, tot sky sfc dnwd beam flux (nir/uv+vis)       2    !
        #    sfdfc   - real, tot sky sfc dnwd diff flux (nir/uv+vis)       2    !
        #    sfbm0   - real, clr sky sfc dnwd beam flux (nir/uv+vis)       2    !
        #    sfdf0   - real, clr sky sfc dnwd diff flux (nir/uv+vis)       2    !
        #    suvbfc  - real, tot sky sfc dnwd uv-b flux                    1    !
        #    suvbf0  - real, clr sky sfc dnwd uv-b flux                    1    !
        #                                                                       !
        #  internal variables:                                                  !
        #    zrefb   - real, direct beam reflectivity for clear/cloudy    nlp1  !
        #    zrefd   - real, diffuse reflectivity for clear/cloudy        nlp1  !
        #    ztrab   - real, direct beam transmissivity for clear/cloudy  nlp1  !
        #    ztrad   - real, diffuse transmissivity for clear/cloudy      nlp1  !
        #    zldbt   - real, layer beam transmittance for clear/cloudy    nlp1  !
        #    ztdbt   - real, lev total beam transmittance for clr/cld     nlp1  !
        #                                                                       !
        #  control parameters in module "physparam"                             !
        #    iswmode - control flag for 2-stream transfer schemes               !
        #              = 1 delta-eddington    (joseph et al., 1976)             !
        #              = 2 pifm               (zdunkowski et al., 1980)         !
        #              = 3 discrete ordinates (liou, 1973)                      !
        #                                                                       !
        #  *******************************************************************  !
        #  original code description                                            !
        #                                                                       !
        #  method:                                                              !
        #  -------                                                              !
        #     standard delta-eddington, p.i.f.m., or d.o.m. layer calculations. !
        #     kmodts  = 1 eddington (joseph et al., 1976)                       !
        #             = 2 pifm (zdunkowski et al., 1980)                        !
        #             = 3 discrete ordinates (liou, 1973)                       !
        #                                                                       !
        #  modifications:                                                       !
        #  --------------                                                       !
        #   original: h. barker                                                 !
        #   revision: merge with rrtmg_sw: j.-j.morcrette, ecmwf, feb 2003      !
        #   revision: add adjustment for earth/sun distance:mjiacono,aer,oct2003!
        #   revision: bug fix for use of palbp and palbd: mjiacono, aer, nov2003!
        #   revision: bug fix to apply delta scaling to clear sky: aer, dec2004 !
        #   revision: code modified so that delta scaling is not done in cloudy !
        #             profiles if routine cldprop is used; delta scaling can be !
        #             applied by swithcing code below if cldprop is not used to !
        #             get cloud properties. aer, jan 2005                       !
        #   revision: uniform formatting for rrtmg: mjiacono, aer, jul 2006     !
        #   revision: use exponential lookup table for transmittance: mjiacono, !
        #             aer, aug 2007                                             !
        #                                                                       !
        #  *******************************************************************  !
        #  ======================  end of description block  =================  !

        #  ---  constant parameters:
        zcrit = 0.9999995  # thresold for conservative scattering
        zsr3 = np.sqrt(3.0)
        od_lo = 0.06
        eps1 = 1.0e-8

        #  ---  outputs:
        fxupc = np.zeros((nlp1, nbdsw))
        fxdnc = np.zeros((nlp1, nbdsw))
        fxup0 = np.zeros((nlp1, nbdsw))
        fxdn0 = np.zeros((nlp1, nbdsw))

        sfbmc = np.zeros(2)
        sfdfc = np.zeros(2)
        sfbm0 = np.zeros(2)
        sfdf0 = np.zeros(2)

        #  ---  locals:
        ztaus = np.zeros(nlay)
        zssas = np.zeros(nlay)
        zasys = np.zeros(nlay)
        zldbt0 = np.zeros(nlay)

        zrefb = np.zeros(nlp1)
        zrefd = np.zeros(nlp1)
        ztrab = np.zeros(nlp1)
        ztrad = np.zeros(nlp1)
        ztdbt = np.zeros(nlp1)
        zldbt = np.zeros(nlp1)
        zfu = np.zeros(nlp1)
        zfd = np.zeros(nlp1)

        #  --- ...  loop over all g-points in each band

        for jg in range(ngptsw):
            jb = NGB[jg] - 1
            ib = jb + 1 - nblow
            ibd = self.idxsfc[jb - 15] - 1  # spectral band index

            zsolar = ssolar * sfluxzen[jg]

            #  --- ...  set up toa direct beam and surface values (beam and diff)

            ztdbt[nlp1 - 1] = 1.0
            ztdbt0 = 1.0

            zldbt[0] = 0.0
            if ibd != -1:
                zrefb[0] = albbm[ibd]
                zrefd[0] = albdf[ibd]
            else:
                zrefb[0] = 0.5 * (albbm[0] + albbm[1])
                zrefd[0] = 0.5 * (albdf[0] + albdf[1])

            ztrab[0] = 0.0
            ztrad[0] = 0.0

            # -# Compute clear-sky optical parameters, layer reflectance and
            #    transmittance.
            #    - Set up toa direct beam and surface values (beam and diff)
            #    - Delta scaling for clear-sky condition
            #    - General two-stream expressions for physparam::iswmode
            #    - Compute homogeneous reflectance and transmittance for both
            #      conservative and non-conservative scattering
            #    - Pre-delta-scaling clear and cloudy direct beam transmittance
            #    - Call swflux() to compute the upward and downward radiation fluxes

            for k in range(nlay - 1, -1, -1):
                kp = k + 1

                ztau0 = max(self.ftiny, taur[k, jg] + taug[k, jg] + tauae[k, ib])
                zssa0 = taur[k, jg] + tauae[k, ib] * ssaae[k, ib]
                zasy0 = asyae[k, ib] * ssaae[k, ib] * tauae[k, ib]
                zssaw = min(self.oneminus, zssa0 / ztau0)
                zasyw = zasy0 / max(self.ftiny, zssa0)

                #  --- ...  saving clear-sky quantities for later total-sky usage
                ztaus[k] = ztau0
                zssas[k] = zssa0
                zasys[k] = zasy0

                #  --- ...  delta scaling for clear-sky condition
                za1 = zasyw * zasyw
                za2 = zssaw * za1

                ztau1 = (1.0 - za2) * ztau0
                zssa1 = (zssaw - za2) / (1.0 - za2)
                zasy1 = zasyw / (1.0 + zasyw)  # to reduce truncation error
                zasy3 = 0.75 * zasy1

                #  --- ...  general two-stream expressions
                if iswmode == 1:
                    zgam1 = 1.75 - zssa1 * (1.0 + zasy3)
                    zgam2 = -0.25 + zssa1 * (1.0 - zasy3)
                    zgam3 = 0.5 - zasy3 * cosz
                elif iswmode == 2:  # pifm
                    zgam1 = 2.0 - zssa1 * (1.25 + zasy3)
                    zgam2 = 0.75 * zssa1 * (1.0 - zasy1)
                    zgam3 = 0.5 - zasy3 * cosz
                elif iswmode == 3:  # discrete ordinates
                    zgam1 = zsr3 * (2.0 - zssa1 * (1.0 + zasy1)) * 0.5
                    zgam2 = zsr3 * zssa1 * (1.0 - zasy1) * 0.5
                    zgam3 = (1.0 - zsr3 * zasy1 * cosz) * 0.5

                zgam4 = 1.0 - zgam3

                #  --- ...  compute homogeneous reflectance and transmittance

                if zssaw >= zcrit:  # for conservative scattering
                    za1 = zgam1 * cosz - zgam3
                    za2 = zgam1 * ztau1

                    #  --- ...  use exponential lookup table for transmittance, or expansion
                    #           of exponential for low optical depth

                    zb1 = min(ztau1 * sntz, 500.0)
                    if zb1 <= od_lo:
                        zb2 = 1.0 - zb1 + 0.5 * zb1 * zb1
                    else:
                        ftind = zb1 / (self.bpade + zb1)
                        itind = int(ftind * ntbmx + 0.5)
                        zb2 = self.exp_tbl[itind]

                    #      ...  collimated beam
                    zrefb[kp] = max(
                        0.0, min(1.0, (za2 - za1 * (1.0 - zb2)) / (1.0 + za2))
                    )
                    ztrab[kp] = max(0.0, min(1.0, 1.0 - zrefb[kp]))

                    #      ...      isotropic incidence
                    zrefd[kp] = max(0.0, min(1.0, za2 / (1.0 + za2)))
                    ztrad[kp] = max(0.0, min(1.0, 1.0 - zrefd[kp]))

                else:  # for non-conservative scattering
                    za1 = zgam1 * zgam4 + zgam2 * zgam3
                    za2 = zgam1 * zgam3 + zgam2 * zgam4
                    zrk = np.sqrt((zgam1 - zgam2) * (zgam1 + zgam2))
                    zrk2 = 2.0 * zrk

                    zrp = zrk * cosz
                    zrp1 = 1.0 + zrp
                    zrm1 = 1.0 - zrp
                    zrpp1 = 1.0 - zrp * zrp
                    zrpp = np.copysign(
                        max(self.flimit, abs(zrpp1)), zrpp1
                    )  # avoid numerical singularity
                    zrkg1 = zrk + zgam1
                    zrkg3 = zrk * zgam3
                    zrkg4 = zrk * zgam4

                    zr1 = zrm1 * (za2 + zrkg3)
                    zr2 = zrp1 * (za2 - zrkg3)
                    zr3 = zrk2 * (zgam3 - za2 * cosz)
                    zr4 = zrpp * zrkg1
                    zr5 = zrpp * (zrk - zgam1)

                    zt1 = zrp1 * (za1 + zrkg4)
                    zt2 = zrm1 * (za1 - zrkg4)
                    zt3 = zrk2 * (zgam4 + za1 * cosz)

                    #  --- ...  use exponential lookup table for transmittance, or expansion
                    #           of exponential for low optical depth

                    zb1 = min(zrk * ztau1, 500.0)
                    if zb1 <= od_lo:
                        zexm1 = 1.0 - zb1 + 0.5 * zb1 * zb1
                    else:
                        ftind = zb1 / (self.bpade + zb1)
                        itind = int(ftind * ntbmx + 0.5)
                        zexm1 = self.exp_tbl[itind]

                    zexp1 = 1.0 / zexm1

                    zb2 = min(sntz * ztau1, 500.0)
                    if zb2 <= od_lo:
                        zexm2 = 1.0 - zb2 + 0.5 * zb2 * zb2
                    else:
                        ftind = zb2 / (self.bpade + zb2)
                        itind = int(ftind * ntbmx + 0.5)
                        zexm2 = self.exp_tbl[itind]

                    zexp2 = 1.0 / zexm2
                    ze1r45 = zr4 * zexp1 + zr5 * zexm1

                    #      ...      collimated beam
                    if ze1r45 >= -eps1 and ze1r45 <= eps1:
                        zrefb[kp] = eps1
                        ztrab[kp] = zexm2
                    else:
                        zden1 = zssa1 / ze1r45
                        zrefb[kp] = max(
                            0.0,
                            min(1.0, (zr1 * zexp1 - zr2 * zexm1 - zr3 * zexm2) * zden1),
                        )
                        ztrab[kp] = max(
                            0.0,
                            min(
                                1.0,
                                zexm2
                                * (
                                    1.0
                                    - (zt1 * zexp1 - zt2 * zexm1 - zt3 * zexp2) * zden1
                                ),
                            ),
                        )

                    #      ...      diffuse beam
                    zden1 = zr4 / (ze1r45 * zrkg1)
                    zrefd[kp] = max(0.0, min(1.0, zgam2 * (zexp1 - zexm1) * zden1))
                    ztrad[kp] = max(0.0, min(1.0, zrk2 * zden1))

                #  --- ...  direct beam transmittance. use exponential lookup table
                #           for transmittance, or expansion of exponential for low
                #           optical depth

                zr1 = ztau1 * sntz
                if zr1 <= od_lo:
                    zexp3 = 1.0 - zr1 + 0.5 * zr1 * zr1
                else:
                    ftind = zr1 / (self.bpade + zr1)
                    itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                    zexp3 = self.exp_tbl[itind]

                ztdbt[k] = zexp3 * ztdbt[kp]
                zldbt[kp] = zexp3

                #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                #           (must use 'orig', unscaled cloud optical depth)

                zr1 = ztau0 * sntz
                if zr1 <= od_lo:
                    zexp4 = 1.0 - zr1 + 0.5 * zr1 * zr1
                else:
                    ftind = zr1 / (self.bpade + zr1)
                    itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                    zexp4 = self.exp_tbl[itind]

                zldbt0[k] = zexp4
                ztdbt0 = zexp4 * ztdbt0

            zfu, zfd = self.vrtqdr(zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1)

            #  --- ...  compute upward and downward fluxes at levels
            for k in range(nlp1):
                fxup0[k, ib] = fxup0[k, ib] + zsolar * zfu[k]
                fxdn0[k, ib] = fxdn0[k, ib] + zsolar * zfd[k]

            # --- ...  surface downward beam/diffuse flux components
            zb1 = zsolar * ztdbt0
            zb2 = zsolar * (zfd[0] - ztdbt0)

            if ibd != -1:
                sfbm0[ibd] = sfbm0[ibd] + zb1
                sfdf0[ibd] = sfdf0[ibd] + zb2
            else:
                zf1 = 0.5 * zb1
                zf2 = 0.5 * zb2
                sfbm0[0] = sfbm0[0] + zf1
                sfdf0[0] = sfdf0[0] + zf2
                sfbm0[1] = sfbm0[1] + zf1
                sfdf0[1] = sfdf0[1] + zf2

            # -# Compute total sky optical parameters, layer reflectance and
            #    transmittance.
            #    - Set up toa direct beam and surface values (beam and diff)
            #    - Delta scaling for total-sky condition
            #    - General two-stream expressions for physparam::iswmode
            #    - Compute homogeneous reflectance and transmittance for
            #      conservative scattering and non-conservative scattering
            #    - Pre-delta-scaling clear and cloudy direct beam transmittance
            #    - Call swflux() to compute the upward and downward radiation fluxes

            if cf1 > self.eps:

                #  --- ...  set up toa direct beam and surface values (beam and diff)
                ztdbt0 = 1.0
                zldbt[0] = 0.0

                for k in range(nlay - 1, -1, -1):
                    kp = k + 1
                    if cldfmc[k, jg] > self.ftiny:  # it is a cloudy-layer

                        ztau0 = ztaus[k] + taucw[k, ib]
                        zssa0 = zssas[k] + ssacw[k, ib]
                        zasy0 = zasys[k] + asycw[k, ib]
                        zssaw = min(self.oneminus, zssa0 / ztau0)
                        zasyw = zasy0 / max(self.ftiny, zssa0)

                        #  --- ...  delta scaling for total-sky condition
                        za1 = zasyw * zasyw
                        za2 = zssaw * za1

                        ztau1 = (1.0 - za2) * ztau0
                        zssa1 = (zssaw - za2) / (1.0 - za2)
                        zasy1 = zasyw / (1.0 + zasyw)
                        zasy3 = 0.75 * zasy1

                        #  --- ...  general two-stream expressions
                        if iswmode == 1:
                            zgam1 = 1.75 - zssa1 * (1.0 + zasy3)
                            zgam2 = -0.25 + zssa1 * (1.0 - zasy3)
                            zgam3 = 0.5 - zasy3 * cosz
                        elif iswmode == 2:  # pifm
                            zgam1 = 2.0 - zssa1 * (1.25 + zasy3)
                            zgam2 = 0.75 * zssa1 * (1.0 - zasy1)
                            zgam3 = 0.5 - zasy3 * cosz
                        elif iswmode == 3:  # discrete ordinates
                            zgam1 = zsr3 * (2.0 - zssa1 * (1.0 + zasy1)) * 0.5
                            zgam2 = zsr3 * zssa1 * (1.0 - zasy1) * 0.5
                            zgam3 = (1.0 - zsr3 * zasy1 * cosz) * 0.5

                        zgam4 = 1.0 - zgam3

                        #  --- ...  compute homogeneous reflectance and transmittance

                        if zssaw >= zcrit:  # for conservative scattering
                            za1 = zgam1 * cosz - zgam3
                            za2 = zgam1 * ztau1

                            #  --- ...  use exponential lookup table for transmittance, or expansion
                            #           of exponential for low optical depth

                            zb1 = min(ztau1 * sntz, 500.0)
                            if zb1 <= od_lo:
                                zb2 = 1.0 - zb1 + 0.5 * zb1 * zb1
                            else:
                                ftind = zb1 / (self.bpade + zb1)
                                itind = int(ftind * ntbmx + 0.5)
                                zb2 = self.exp_tbl[itind]

                            #      ...  collimated beam
                            zrefb[kp] = max(
                                0.0, min(1.0, (za2 - za1 * (1.0 - zb2)) / (1.0 + za2))
                            )
                            ztrab[kp] = max(0.0, min(1.0, 1.0 - zrefb[kp]))

                            #      ...  isotropic incidence
                            zrefd[kp] = max(0.0, min(1.0, za2 / (1.0 + za2)))
                            ztrad[kp] = max(0.0, min(1.0, 1.0 - zrefd(kp)))

                        else:  # for non-conservative scattering
                            za1 = zgam1 * zgam4 + zgam2 * zgam3
                            za2 = zgam1 * zgam3 + zgam2 * zgam4
                            zrk = np.sqrt((zgam1 - zgam2) * (zgam1 + zgam2))
                            zrk2 = 2.0 * zrk

                            zrp = zrk * cosz
                            zrp1 = 1.0 + zrp
                            zrm1 = 1.0 - zrp
                            zrpp1 = 1.0 - zrp * zrp
                            zrpp = np.copysign(
                                max(self.flimit, abs(zrpp1)), zrpp1
                            )  # avoid numerical singularity
                            zrkg1 = zrk + zgam1
                            zrkg3 = zrk * zgam3
                            zrkg4 = zrk * zgam4

                            zr1 = zrm1 * (za2 + zrkg3)
                            zr2 = zrp1 * (za2 - zrkg3)
                            zr3 = zrk2 * (zgam3 - za2 * cosz)
                            zr4 = zrpp * zrkg1
                            zr5 = zrpp * (zrk - zgam1)

                            zt1 = zrp1 * (za1 + zrkg4)
                            zt2 = zrm1 * (za1 - zrkg4)
                            zt3 = zrk2 * (zgam4 + za1 * cosz)

                            #  --- ...  use exponential lookup table for transmittance, or expansion
                            #           of exponential for low optical depth

                            zb1 = min(zrk * ztau1, 500.0)
                            if zb1 <= od_lo:
                                zexm1 = 1.0 - zb1 + 0.5 * zb1 * zb1
                            else:
                                ftind = zb1 / (self.bpade + zb1)
                                itind = int(ftind * ntbmx + 0.5)
                                zexm1 = self.exp_tbl[itind]

                            zexp1 = 1.0 / zexm1

                            zb2 = min(ztau1 * sntz, 500.0)
                            if zb2 <= od_lo:
                                zexm2 = 1.0 - zb2 + 0.5 * zb2 * zb2
                            else:
                                ftind = zb2 / (self.bpade + zb2)
                                itind = int(ftind * ntbmx + 0.5)
                                zexm2 = self.exp_tbl[itind]

                            zexp2 = 1.0 / zexm2
                            ze1r45 = zr4 * zexp1 + zr5 * zexm1

                            #      ...  collimated beam
                            if ze1r45 >= -eps1 and ze1r45 <= eps1:
                                zrefb[kp] = eps1
                                ztrab[kp] = zexm2
                            else:
                                zden1 = zssa1 / ze1r45
                                zrefb[kp] = max(
                                    0.0,
                                    min(
                                        1.0,
                                        (zr1 * zexp1 - zr2 * zexm1 - zr3 * zexm2)
                                        * zden1,
                                    ),
                                )
                                ztrab[kp] = max(
                                    0.0,
                                    min(
                                        1.0,
                                        zexm2
                                        * (
                                            1.0
                                            - (zt1 * zexp1 - zt2 * zexm1 - zt3 * zexp2)
                                            * zden1
                                        ),
                                    ),
                                )

                            #      ...  diffuse beam
                            zden1 = zr4 / (ze1r45 * zrkg1)
                            zrefd[kp] = max(
                                0.0, min(1.0, zgam2 * (zexp1 - zexm1) * zden1)
                            )
                            ztrad[kp] = max(0.0, min(1.0, zrk2 * zden1))

                        #  --- ...  direct beam transmittance. use exponential lookup table
                        #           for transmittance, or expansion of exponential for low
                        #           optical depth

                        zr1 = ztau1 * sntz
                        if zr1 <= od_lo:
                            zexp3 = 1.0 - zr1 + 0.5 * zr1 * zr1
                        else:
                            ftind = zr1 / (self.bpade + zr1)
                            itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                            zexp3 = self.exp_tbl[itind]

                        zldbt[kp] = zexp3
                        ztdbt[k] = zexp3 * ztdbt[kp]

                        #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                        #           (must use 'orig', unscaled cloud optical depth)

                        zr1 = ztau0 * sntz
                        if zr1 <= od_lo:
                            zexp4 = 1.0 - zr1 + 0.5 * zr1 * zr1
                        else:
                            ftind = zr1 / (self.bpade + zr1)
                            itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                            zexp4 = self.exp_tbl[itind]

                        ztdbt0 = zexp4 * ztdbt0

                    else:  # if_cldfmc_block  ---  it is a clear layer

                        #  --- ...  direct beam transmittance
                        ztdbt[k] = zldbt[kp] * ztdbt[kp]

                        #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                        ztdbt0 = zldbt0[k] * ztdbt0

                #  --- ...  perform vertical quadrature

                zfu, zfd = self.vrtqdr(
                    zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1
                )

                #  --- ...  compute upward and downward fluxes at levels
                for k in range(nlp1):
                    fxupc[k, ib] = fxupc[k, ib] + zsolar * zfu[k]
                    fxdnc[k, ib] = fxdnc[k, ib] + zsolar * zfd[k]

                #  -# Process and save outputs.
                # --- ...  surface downward beam/diffused flux components
                zb1 = zsolar * ztdbt0
                zb2 = zsolar * (zfd[0] - ztdbt0)

                if ibd != -1:
                    sfbmc[ibd] = sfbmc[ibd] + zb1
                    sfdfc[ibd] = sfdfc[ibd] + zb2
                else:
                    zf1 = 0.5 * zb1
                    zf2 = 0.5 * zb2
                    sfbmc[0] = sfbmc[0] + zf1
                    sfdfc[0] = sfdfc[0] + zf2
                    sfbmc[1] = sfbmc[1] + zf1
                    sfdfc[1] = sfdfc[1] + zf2

        #  --- ...  end of g-point loop
        ftoadc = 0
        ftoauc = 0
        ftoau0 = 0
        fsfcu0 = 0
        fsfcuc = 0
        fsfcd0 = 0
        fsfcdc = 0

        for ib in range(nbdsw):
            ftoadc = ftoadc + fxdn0[nlp1 - 1, ib]
            ftoau0 = ftoau0 + fxup0[nlp1 - 1, ib]
            fsfcu0 = fsfcu0 + fxup0[0, ib]
            fsfcd0 = fsfcd0 + fxdn0[0, ib]

        # --- ...  uv-b surface downward flux
        ibd = self.nuvb - nblow
        suvbf0 = fxdn0[0, ibd]

        if cf1 <= self.eps:  # clear column, set total-sky=clear-sky fluxes
            for ib in range(nbdsw):
                for k in range(nlp1):
                    fxupc[k, ib] = fxup0[k, ib]
                    fxdnc[k, ib] = fxdn0[k, ib]

            ftoauc = ftoau0
            fsfcuc = fsfcu0
            fsfcdc = fsfcd0

            # --- ...  surface downward beam/diffused flux components
            sfbmc[0] = sfbm0[0]
            sfdfc[0] = sfdf0[0]
            sfbmc[1] = sfbm0[1]
            sfdfc[1] = sfdf0[1]

            # --- ...  uv-b surface downward flux
            suvbfc = suvbf0
        else:  # cloudy column, compute total-sky fluxes
            for ib in range(nbdsw):
                ftoauc = ftoauc + fxupc[nlp1 - 1, ib]
                fsfcuc = fsfcuc + fxupc[0, ib]
                fsfcdc = fsfcdc + fxdnc[0, ib]

            # --- ...  uv-b surface downward flux
            suvbfc = fxdnc[0, ibd]

        return (
            fxupc,
            fxdnc,
            fxup0,
            fxdn0,
            ftoauc,
            ftoau0,
            ftoadc,
            fsfcuc,
            fsfcu0,
            fsfcdc,
            fsfcd0,
            sfbmc,
            sfdfc,
            sfbm0,
            sfdf0,
            suvbfc,
            suvbf0,
        )

    def vrtqdr(self, zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1):

        #  ---  outputs:
        zfu = np.zeros(nlp1)
        zfd = np.zeros(nlp1)

        #  ---  locals:
        zrupb = np.zeros(nlp1)
        zrupd = np.zeros(nlp1)
        zrdnd = np.zeros(nlp1)
        ztdn = np.zeros(nlp1)

        # -# Link lowest layer with surface.
        zrupb[0] = zrefb[0]  # direct beam
        zrupd[0] = zrefd[0]  # diffused

        # -# Pass from bottom to top.
        for k in range(nlay):
            kp = k + 1

            zden1 = 1.0 / (1.0 - zrupd[k] * zrefd[kp])
            zrupb[kp] = (
                zrefb[kp]
                + (
                    ztrad[kp]
                    * ((ztrab[kp] - zldbt[kp]) * zrupd[k] + zldbt[kp] * zrupb[k])
                )
                * zden1
            )
            zrupd[kp] = zrefd[kp] + ztrad[kp] * ztrad[kp] * zrupd[k] * zden1

        # -# Upper boundary conditions
        ztdn[nlp1 - 1] = 1.0
        zrdnd[nlp1 - 1] = 0.0
        ztdn[nlay - 1] = ztrab[nlp1 - 1]
        zrdnd[nlay - 1] = zrefd[nlp1 - 1]

        # -# Pass from top to bottom
        for k in range(nlay - 1, 0, -1):
            zden1 = 1.0 / (1.0 - zrefd[k] * zrdnd[k])
            ztdn[k - 1] = (
                ztdbt[k] * ztrab[k]
                + (ztrad[k] * ((ztdn[k] - ztdbt[k]) + ztdbt[k] * zrefb[k] * zrdnd[k]))
                * zden1
            )
            zrdnd[k - 1] = zrefd[k] + ztrad[k] * ztrad[k] * zrdnd[k] * zden1

        # -# Up and down-welling fluxes at levels.
        for k in range(nlp1):
            zden1 = 1.0 / (1.0 - zrdnd[k] * zrupd[k])
            zfu[k] = (ztdbt[k] * zrupb[k] + (ztdn[k] - ztdbt[k]) * zrupd[k]) * zden1
            zfd[k] = (
                ztdbt[k] + (ztdn[k] - ztdbt[k] + ztdbt[k] * zrupb[k] * zrdnd[k]) * zden1
            )

        return zfu, zfd

    def taumol(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
    ):
        #  ==================   program usage description   ==================  !
        #                                                                       !
        #  description:                                                         !
        #    calculate optical depths for gaseous absorption and rayleigh       !
        #    scattering.                                                        !
        #                                                                       !
        #  subroutines called: taugb## (## = 16 - 29)                           !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                         size !
        #    colamt  - real, column amounts of absorbing gases the index        !
        #                    are for h2o, co2, o3, n2o, ch4, and o2,            !
        #                    respectively (molecules/cm**2)          nlay*maxgas!
        #    colmol  - real, total column amount (dry air+water vapor)     nlay !
        #    facij   - real, for each layer, these are factors that are         !
        #                    needed to compute the interpolation factors        !
        #                    that multiply the appropriate reference k-         !
        #                    values.  a value of 0/1 for i,j indicates          !
        #                    that the corresponding factor multiplies           !
        #                    reference k-value for the lower/higher of the      !
        #                    two appropriate temperatures, and altitudes,       !
        #                    respectively.                                 naly !
        #    jp      - real, the index of the lower (in altitude) of the        !
        #                    two appropriate ref pressure levels needed         !
        #                    for interpolation.                            nlay !
        #    jt, jt1 - integer, the indices of the lower of the two approp      !
        #                    ref temperatures needed for interpolation (for     !
        #                    pressure levels jp and jp+1, respectively)    nlay !
        #    laytrop - integer, tropopause layer index                       1  !
        #    forfac  - real, scale factor needed to foreign-continuum.     nlay !
        #    forfrac - real, factor needed for temperature interpolation   nlay !
        #    indfor  - integer, index of the lower of the two appropriate       !
        #                    reference temperatures needed for foreign-         !
        #                    continuum interpolation                       nlay !
        #    selffac - real, scale factor needed to h2o self-continuum.    nlay !
        #    selffrac- real, factor needed for temperature interpolation        !
        #                    of reference h2o self-continuum data          nlay !
        #    indself - integer, index of the lower of the two appropriate       !
        #                    reference temperatures needed for the self-        !
        #                    continuum interpolation                       nlay !
        #    nlay    - integer, number of vertical layers                    1  !
        #                                                                       !
        #  output:                                                              !
        #    sfluxzen- real, spectral distribution of incoming solar flux ngptsw!
        #    taug    - real, spectral optical depth for gases        nlay*ngptsw!
        #    taur    - real, opt depth for rayleigh scattering       nlay*ngptsw!
        #                                                                       !
        #  ===================================================================  !
        #  ************     original subprogram description    ***************  !
        #                                                                       !
        #                  optical depths developed for the                     !
        #                                                                       !
        #                rapid radiative transfer model (rrtm)                  !
        #                                                                       !
        #            atmospheric and environmental research, inc.               !
        #                        131 hartwell avenue                            !
        #                        lexington, ma 02421                            !
        #                                                                       !
        #                                                                       !
        #                           eli j. mlawer                               !
        #                         jennifer delamere                             !
        #                         steven j. taubman                             !
        #                         shepard a. clough                             !
        #                                                                       !
        #                                                                       !
        #                                                                       !
        #                       email:  mlawer@aer.com                          !
        #                       email:  jdelamer@aer.com                        !
        #                                                                       !
        #        the authors wish to acknowledge the contributions of the       !
        #        following people:  patrick d. brown, michael j. iacono,        !
        #        ronald e. farren, luke chen, robert bergstrom.                 !
        #                                                                       !
        #  *******************************************************************  !
        #                                                                       !
        #  taumol                                                               !
        #                                                                       !
        #    this file contains the subroutines taugbn (where n goes from       !
        #    16 to 29).  taugbn calculates the optical depths and Planck        !
        #    fractions per g-value and layer for band n.                        !
        #                                                                       !
        #  output:  optical depths (unitless)                                   !
        #           fractions needed to compute planck functions at every layer !
        #           and g-value                                                 !
        #                                                                       !
        #  modifications:                                                       !
        #                                                                       !
        # revised: adapted to f90 coding, j.-j.morcrette, ecmwf, feb 2003       !
        # revised: modified for g-point reduction, mjiacono, aer, dec 2003      !
        # revised: reformatted for consistency with rrtmg_lw, mjiacono, aer,    !
        #          jul 2006                                                     !
        #                                                                       !
        #  *******************************************************************  !
        #  ======================  end of description block  =================  !

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_sflux_data.nc"))
        self.strrat = ds["strrat"].data
        specwt = ds["specwt"].data
        layreffr = ds["layreffr"].data
        ix1 = ds["ix1"].data
        ix2 = ds["ix2"].data
        ibx = ds["ibx"].data
        sfluxref01 = ds["sfluxref01"].data
        sfluxref02 = ds["sfluxref02"].data
        sfluxref03 = ds["sfluxref03"].data
        scalekur = ds["scalekur"].data

        id0 = np.zeros((nlay, nbhgh), dtype=np.int32)
        id1 = np.zeros((nlay, nbhgh), dtype=np.int32)
        sfluxzen = np.zeros(ngptsw)

        taug = np.zeros((nlay, ngptsw))
        taur = np.zeros((nlay, ngptsw))

        for b in range(nbhgh - nblow + 1):
            jb = nblow + b - 1

            #  --- ...  indices for layer optical depth

            for k in range(laytrop):
                id0[k, jb] = ((jp[k] - 1) * 5 + (jt[k] - 1)) * nspa[b] - 1
                id1[k, jb] = (jp[k] * 5 + (jt1[k] - 1)) * nspa[b] - 1

            for k in range(laytrop, nlay):
                id0[k, jb] = ((jp[k] - 13) * 5 + (jt[k] - 1)) * nspb[b] - 1
                id1[k, jb] = ((jp[k] - 12) * 5 + (jt1[k] - 1)) * nspb[b] - 1

            #  --- ...  calculate spectral flux at toa
            ibd = ibx[b] - 1
            njb = ng[b]
            ns = ngs[b]

            if jb in [15, 19, 22, 24, 25, 28]:
                for j in range(njb):
                    sfluxzen[ns + j] = sfluxref01[j, 0, ibd]
            elif jb == 26:
                for j in range(njb):
                    sfluxzen[ns + j] = scalekur * sfluxref01[j, 0, ibd]
            else:
                if jb == 16 or jb == 27:
                    ks = nlay - 1
                    for k in range(laytrop - 1, nlay - 1):
                        if (jp[k] < layreffr[b]) and jp[k + 1] >= layreffr[b]:
                            ks = k + 1
                            break

                    colm1 = colamt[ks, ix1[b] - 1]
                    colm2 = colamt[ks, ix2[b] - 1]
                    speccomb = colm1 + self.strrat[b] * colm2
                    specmult = specwt[b] * min(oneminus, colm1 / speccomb)
                    js = 1 + int(specmult) - 1
                    fs = np.mod(specmult, 1.0)

                    for j in range(njb):
                        sfluxzen[ns + j] = sfluxref02[j, js, ibd] + fs * (
                            sfluxref02[j, js + 1, ibd] - sfluxref02[j, js, ibd]
                        )
                else:
                    ks = laytrop - 1
                    for k in range(laytrop - 1):
                        if jp[k] < layreffr[b] and jp[k + 1] >= layreffr[b]:
                            ks = k + 1
                            break
                    colm1 = colamt[ks, ix1[b] - 1]
                    colm2 = colamt[ks, ix2[b] - 1]
                    speccomb = colm1 + self.strrat[b] * colm2
                    specmult = specwt[b] * min(oneminus, colm1 / speccomb)
                    js = 1 + int(specmult) - 1
                    fs = np.mod(specmult, 1.0)

                    for j in range(njb):
                        sfluxzen[ns + j] = sfluxref03[j, js, ibd] + fs * (
                            sfluxref03[j, js + 1, ibd] - sfluxref03[j, js, ibd]
                        )

        taug, taur = self.taumol16(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol17(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol18(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol19(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol20(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol21(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol22(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol23(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol24(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol25(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol26(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol27(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol28(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        taug, taur = self.taumol29(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
            id0,
            id1,
            taug,
            taur,
        )

        return sfluxzen, taug, taur

    def taumol16(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb16_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        rayl = ds["rayl"].data

        #  --- ... compute the optical depth by interpolating in ln(pressure),
        #          temperature, and appropriate species.  below laytrop, the water
        #          vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            tauray = colmol[k] * rayl

            for j in range(NG16):
                taur[k, NS16 + j] = tauray

        for k in range(laytrop):
            speccomb = colamt[k, 0] + self.strrat[0] * colamt[k, 4]
            specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

            js = 1 + int(specmult)
            fs = np.mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00[k]
            fac010 = fs1 * fac10[k]
            fac100 = fs * fac00[k]
            fac110 = fs * fac10[k]
            fac001 = fs1 * fac01[k]
            fac011 = fs1 * fac11[k]
            fac101 = fs * fac01[k]
            fac111 = fs * fac11[k]

            ind01 = id0[k, 15] + js
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[k, 15] + js
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10
            inds = indself[k] - 1
            indf = indfor[k] - 1
            indsp = inds + 1
            indfp = indf + 1

            for j in range(NG16):
                taug[k, NS16 + j] = speccomb * (
                    fac000 * absa[ind01, j]
                    + fac100 * absa[ind02, j]
                    + fac010 * absa[ind03, j]
                    + fac110 * absa[ind04, j]
                    + fac001 * absa[ind11, j]
                    + fac101 * absa[ind12, j]
                    + fac011 * absa[ind13, j]
                    + fac111 * absa[ind14, j]
                ) + colamt[k, 0] * (
                    selffac[k]
                    * (
                        selfref[inds, j]
                        + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                    )
                    + forfac[k]
                    * (
                        forref[indf, j]
                        + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                    )
                )

            for k in range(laytrop, nlay):
                ind01 = id0[k, 15] + 1
                ind02 = ind01 + 1
                ind11 = id1[k, 15] + 1
                ind12 = ind11 + 1

                for j in range(NG16):
                    taug[k, NS16 + j] = colamt[k, 4] * (
                        fac00[k] * absb[ind01, j]
                        + fac10[k] * absb[ind02, j]
                        + fac01[k] * absb[ind11, j]
                        + fac11[k] * absb[ind12, j]
                    )
        return taug, taur

    # The subroutine computes the optical depth in band 17:  3250-4000
    # cm-1 (low - h2o,co2; high - h2o,co2)

    def taumol17(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 17:  3250-4000 cm-1 (low - h2o,co2; high - h2o,co2)         !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb17_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        rayl = ds["rayl"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            tauray = colmol[k] * rayl

            for j in range(NG17):
                taur[k, NS17 + j] = tauray

        for k in range(laytrop):
            speccomb = colamt[k, 0] + self.strrat[1] * colamt[k, 1]
            specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

            js = 1 + int(specmult)
            fs = np.mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00[k]
            fac010 = fs1 * fac10[k]
            fac100 = fs * fac00[k]
            fac110 = fs * fac10[k]
            fac001 = fs1 * fac01[k]
            fac011 = fs1 * fac11[k]
            fac101 = fs * fac01[k]
            fac111 = fs * fac11[k]

            ind01 = id0[k, 16] + js
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[k, 16] + js
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself[k] - 1
            indf = indfor[k] - 1
            indsp = inds + 1
            indfp = indf + 1

            for j in range(NG17):
                taug[k, NS17 + j] = speccomb * (
                    fac000 * absa[ind01, j]
                    + fac100 * absa[ind02, j]
                    + fac010 * absa[ind03, j]
                    + fac110 * absa[ind04, j]
                    + fac001 * absa[ind11, j]
                    + fac101 * absa[ind12, j]
                    + fac011 * absa[ind13, j]
                    + fac111 * absa[ind14, j]
                ) + colamt[k, 0] * (
                    selffac[k]
                    * (
                        selfref[inds, j]
                        + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                    )
                    + forfac[k]
                    * (
                        forref[indf, j]
                        + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                    )
                )

        for k in range(laytrop, nlay):
            speccomb = colamt[k, 0] + self.strrat[1] * colamt[k, 1]
            specmult = 4.0 * min(oneminus, colamt[k, 0] / speccomb)

            js = 1 + int(specmult)
            fs = np.mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00[k]
            fac010 = fs1 * fac10[k]
            fac100 = fs * fac00[k]
            fac110 = fs * fac10[k]
            fac001 = fs1 * fac01[k]
            fac011 = fs1 * fac11[k]
            fac101 = fs * fac01[k]
            fac111 = fs * fac11[k]

            ind01 = id0[k, 16] + js
            ind02 = ind01 + 1
            ind03 = ind01 + 5
            ind04 = ind01 + 6
            ind11 = id1[k, 16] + js
            ind12 = ind11 + 1
            ind13 = ind11 + 5
            ind14 = ind11 + 6

            indf = indfor[k] - 1
            indfp = indf + 1

            for j in range(NG17):
                taug[k, NS17 + j] = speccomb * (
                    fac000 * absb[ind01, j]
                    + fac100 * absb[ind02, j]
                    + fac010 * absb[ind03, j]
                    + fac110 * absb[ind04, j]
                    + fac001 * absb[ind11, j]
                    + fac101 * absb[ind12, j]
                    + fac011 * absb[ind13, j]
                    + fac111 * absb[ind14, j]
                ) + colamt[k, 0] * forfac[k] * (
                    forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                )

        return taug, taur

    # The subroutine computes the optical depth in band 18:  4000-4650
    # cm-1 (low - h2o,ch4; high - ch4)

    def taumol18(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 18:  4000-4650 cm-1 (low - h2o,ch4; high - ch4)             !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb18_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        rayl = ds["rayl"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            tauray = colmol[k] * rayl

            for j in range(NG18):
                taur[k, NS18 + j] = tauray

        for k in range(laytrop):
            speccomb = colamt[k, 0] + self.strrat[2] * colamt[k, 4]
            specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

            js = 1 + int(specmult)
            fs = np.mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00[k]
            fac010 = fs1 * fac10[k]
            fac100 = fs * fac00[k]
            fac110 = fs * fac10[k]
            fac001 = fs1 * fac01[k]
            fac011 = fs1 * fac11[k]
            fac101 = fs * fac01[k]
            fac111 = fs * fac11[k]

            ind01 = id0[k, 17] + js
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[k, 17] + js
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself[k] - 1
            indf = indfor[k] - 1
            indsp = inds + 1
            indfp = indf + 1

            for j in range(NG18):
                taug[k, NS18 + j] = speccomb * (
                    fac000 * absa[ind01, j]
                    + fac100 * absa[ind02, j]
                    + fac010 * absa[ind03, j]
                    + fac110 * absa[ind04, j]
                    + fac001 * absa[ind11, j]
                    + fac101 * absa[ind12, j]
                    + fac011 * absa[ind13, j]
                    + fac111 * absa[ind14, j]
                ) + colamt[k, 0] * (
                    selffac[k]
                    * (
                        selfref[inds, j]
                        + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                    )
                    + forfac[k]
                    * (
                        forref[indf, j]
                        + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                    )
                )

        for k in range(laytrop, nlay):
            ind01 = id0[k, 17] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 17] + 1
            ind12 = ind11 + 1

            for j in range(NG18):
                taug[k, NS18 + j] = colamt[k, 4] * (
                    fac00[k] * absb[ind01, j]
                    + fac10[k] * absb[ind02, j]
                    + fac01[k] * absb[ind11, j]
                    + fac11[k] * absb[ind12, j]
                )

        return taug, taur

    # The subroutine computes the optical depth in band 19:  4650-5150
    # cm-1 (low - h2o,co2; high - co2)

    def taumol19(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 19:  4650-5150 cm-1 (low - h2o,co2; high - co2)             !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb19_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        rayl = ds["rayl"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            tauray = colmol[k] * rayl

            for j in range(NG19):
                taur[k, NS19 + j] = tauray

        for k in range(laytrop):
            speccomb = colamt[k, 0] + self.strrat[3] * colamt[k, 1]
            specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

            js = 1 + int(specmult)
            fs = np.mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00[k]
            fac010 = fs1 * fac10[k]
            fac100 = fs * fac00[k]
            fac110 = fs * fac10[k]
            fac001 = fs1 * fac01[k]
            fac011 = fs1 * fac11[k]
            fac101 = fs * fac01[k]
            fac111 = fs * fac11[k]

            ind01 = id0[k, 18] + js
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[k, 18] + js
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself[k] - 1
            indf = indfor[k] - 1
            indsp = inds + 1
            indfp = indf + 1

            for j in range(NG19):
                taug[k, NS19 + j] = speccomb * (
                    fac000 * absa[ind01, j]
                    + fac100 * absa[ind02, j]
                    + fac010 * absa[ind03, j]
                    + fac110 * absa[ind04, j]
                    + fac001 * absa[ind11, j]
                    + fac101 * absa[ind12, j]
                    + fac011 * absa[ind13, j]
                    + fac111 * absa[ind14, j]
                ) + colamt[k, 0] * (
                    selffac[k]
                    * (
                        selfref[inds, j]
                        + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                    )
                    + forfac[k]
                    * (
                        forref[indf, j]
                        + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                    )
                )

        for k in range(laytrop, nlay):
            ind01 = id0[k, 18] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 18] + 1
            ind12 = ind11 + 1

            for j in range(NG19):
                taug[k, NS19 + j] = colamt[k, 1] * (
                    fac00[k] * absb[ind01, j]
                    + fac10[k] * absb[ind02, j]
                    + fac01[k] * absb[ind11, j]
                    + fac11[k] * absb[ind12, j]
                )

        return taug, taur

    # The subroutine computes the optical depth in band 20:  5150-6150
    # cm-1 (low - h2o; high - h2o)

    def taumol20(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 20:  5150-6150 cm-1 (low - h2o; high - h2o)                 !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb20_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        absch4 = ds["absch4"].data
        rayl = ds["rayl"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            tauray = colmol[k] * rayl

            for j in range(NG20):
                taur[k, NS20 + j] = tauray

        for k in range(laytrop):
            ind01 = id0[k, 19] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 19] + 1
            ind12 = ind11 + 1

            inds = indself[k] - 1
            indf = indfor[k] - 1
            indsp = inds + 1
            indfp = indf + 1

            for j in range(NG20):
                taug[k, NS20 + j] = (
                    colamt[k, 0]
                    * (
                        (
                            fac00[k] * absa[ind01, j]
                            + fac10[k] * absa[ind02, j]
                            + fac01[k] * absa[ind11, j]
                            + fac11[k] * absa[ind12, j]
                        )
                        + selffac[k]
                        * (
                            selfref[inds, j]
                            + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                        )
                        + forfac[k]
                        * (
                            forref[indf, j]
                            + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                        )
                    )
                    + colamt[k, 4] * absch4[j]
                )

        for k in range(laytrop, nlay):
            ind01 = id0[k, 19] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 19] + 1
            ind12 = ind11 + 1

            indf = indfor[k] - 1
            indfp = indf + 1

            for j in range(NG20):
                taug[k, NS20 + j] = (
                    colamt[k, 0]
                    * (
                        fac00[k] * absb[ind01, j]
                        + fac10[k] * absb[ind02, j]
                        + fac01[k] * absb[ind11, j]
                        + fac11[k] * absb[ind12, j]
                        + forfac[k]
                        * (
                            forref[indf, j]
                            + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                        )
                    )
                    + colamt[k, 4] * absch4[j]
                )

        return taug, taur

    # The subroutine computes the optical depth in band 21:  6150-7700
    # cm-1 (low - h2o,co2; high - h2o,co2)

    def taumol21(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 21:  6150-7700 cm-1 (low - h2o,co2; high - h2o,co2)         !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb21_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        rayl = ds["rayl"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            tauray = colmol[k] * rayl

            for j in range(NG21):
                taur[k, NS21 + j] = tauray

        for k in range(laytrop):
            speccomb = colamt[k, 0] + self.strrat[5] * colamt[k, 1]
            specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

            js = 1 + int(specmult)
            fs = np.mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00[k]
            fac010 = fs1 * fac10[k]
            fac100 = fs * fac00[k]
            fac110 = fs * fac10[k]
            fac001 = fs1 * fac01[k]
            fac011 = fs1 * fac11[k]
            fac101 = fs * fac01[k]
            fac111 = fs * fac11[k]

            ind01 = id0[k, 20] + js
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[k, 20] + js
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself[k] - 1
            indf = indfor[k] - 1
            indsp = inds + 1
            indfp = indf + 1

            for j in range(NG21):
                taug[k, NS21 + j] = speccomb * (
                    fac000 * absa[ind01, j]
                    + fac100 * absa[ind02, j]
                    + fac010 * absa[ind03, j]
                    + fac110 * absa[ind04, j]
                    + fac001 * absa[ind11, j]
                    + fac101 * absa[ind12, j]
                    + fac011 * absa[ind13, j]
                    + fac111 * absa[ind14, j]
                ) + colamt[k, 0] * (
                    selffac[k]
                    * (
                        selfref[inds, j]
                        + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                    )
                    + forfac[k]
                    * (
                        forref[indf, j]
                        + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                    )
                )

        for k in range(laytrop, nlay):
            speccomb = colamt[k, 0] + self.strrat[5] * colamt[k, 1]
            specmult = 4.0 * min(oneminus, colamt[k, 0] / speccomb)

            js = 1 + int(specmult)
            fs = np.mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00[k]
            fac010 = fs1 * fac10[k]
            fac100 = fs * fac00[k]
            fac110 = fs * fac10[k]
            fac001 = fs1 * fac01[k]
            fac011 = fs1 * fac11[k]
            fac101 = fs * fac01[k]
            fac111 = fs * fac11[k]

            ind01 = id0[k, 20] + js
            ind02 = ind01 + 1
            ind03 = ind01 + 5
            ind04 = ind01 + 6
            ind11 = id1[k, 20] + js
            ind12 = ind11 + 1
            ind13 = ind11 + 5
            ind14 = ind11 + 6

            indf = indfor[k] - 1
            indfp = indf + 1

            for j in range(NG21):
                taug[k, NS21 + j] = speccomb * (
                    fac000 * absb[ind01, j]
                    + fac100 * absb[ind02, j]
                    + fac010 * absb[ind03, j]
                    + fac110 * absb[ind04, j]
                    + fac001 * absb[ind11, j]
                    + fac101 * absb[ind12, j]
                    + fac011 * absb[ind13, j]
                    + fac111 * absb[ind14, j]
                ) + colamt[k, 0] * forfac[k] * (
                    forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                )

        return taug, taur

    # The subroutine computes the optical depth in band 22:  7700-8050
    # cm-1 (low - h2o,o2; high - o2)

    def taumol22(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 22:  7700-8050 cm-1 (low - h2o,o2; high - o2)               !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb22_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        rayl = ds["rayl"].data

        #  --- ...  the following factor is the ratio of total o2 band intensity (lines
        #           and mate continuum) to o2 band intensity (line only). it is needed
        #           to adjust the optical depths since the k's include only lines.

        o2adj = 1.6
        o2tem = 4.35e-4 / (350.0 * 2.0)

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            tauray = colmol[k] * rayl

            for j in range(NG22):
                taur[k, NS22 + j] = tauray

        for k in range(laytrop):
            o2cont = o2tem * colamt[k, 5]
            speccomb = colamt[k, 0] + self.strrat[6] * colamt[k, 5]
            specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

            js = 1 + int(specmult)
            fs = np.mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00[k]
            fac010 = fs1 * fac10[k]
            fac100 = fs * fac00[k]
            fac110 = fs * fac10[k]
            fac001 = fs1 * fac01[k]
            fac011 = fs1 * fac11[k]
            fac101 = fs * fac01[k]
            fac111 = fs * fac11[k]

            ind01 = id0[k, 21] + js
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[k, 21] + js
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself[k] - 1
            indf = indfor[k] - 1
            indsp = inds + 1
            indfp = indf + 1

            for j in range(NG22):
                taug[k, NS22 + j] = (
                    speccomb
                    * (
                        fac000 * absa[ind01, j]
                        + fac100 * absa[ind02, j]
                        + fac010 * absa[ind03, j]
                        + fac110 * absa[ind04, j]
                        + fac001 * absa[ind11, j]
                        + fac101 * absa[ind12, j]
                        + fac011 * absa[ind13, j]
                        + fac111 * absa[ind14, j]
                    )
                    + colamt[k, 0]
                    * (
                        selffac[k]
                        * (
                            selfref[inds, j]
                            + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                        )
                        + forfac[k]
                        * (
                            forref[indf, j]
                            + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                        )
                    )
                    + o2cont
                )

        for k in range(laytrop, nlay):
            o2cont = o2tem * colamt[k, 5]

            ind01 = id0[k, 21] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 21] + 1
            ind12 = ind11 + 1

            for j in range(NG22):
                taug[k, NS22 + j] = (
                    colamt[k, 5]
                    * o2adj
                    * (
                        fac00[k] * absb[ind01, j]
                        + fac10[k] * absb[ind02, j]
                        + fac01[k] * absb[ind11, j]
                        + fac11[k] * absb[ind12, j]
                    )
                    + o2cont
                )

        return taug, taur

    # The subroutine computes the optical depth in band 23:  8050-12850
    # cm-1 (low - h2o; high - nothing)

    def taumol23(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 23:  8050-12850 cm-1 (low - h2o; high - nothing)            !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb23_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        rayl = ds["rayl"].data
        givfac = ds["givfac"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            for j in range(NG23):
                taur[k, NS23 + j] = colmol[k] * rayl[j]

        for k in range(laytrop):
            ind01 = id0[k, 22] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 22] + 1
            ind12 = ind11 + 1

            inds = indself[k] - 1
            indf = indfor[k] - 1
            indsp = inds + 1
            indfp = indf + 1

            for j in range(NG23):
                taug[k, NS23 + j] = colamt[k, 0] * (
                    givfac
                    * (
                        fac00[k] * absa[ind01, j]
                        + fac10[k] * absa[ind02, j]
                        + fac01[k] * absa[ind11, j]
                        + fac11[k] * absa[ind12, j]
                    )
                    + selffac[k]
                    * (
                        selfref[inds, j]
                        + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                    )
                    + forfac[k]
                    * (
                        forref[indf, j]
                        + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                    )
                )

        for k in range(laytrop, nlay):
            for j in range(NG23):
                taug[k, NS23 + j] = 0.0

        return taug, taur

    # The subroutine computes the optical depth in band 24:  12850-16000
    # cm-1 (low - h2o,o2; high - o2)

    def taumol24(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 24:  12850-16000 cm-1 (low - h2o,o2; high - o2)             !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb24_data.nc"))
        selfref = ds["selfref"].data
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        abso3a = ds["abso3a"].data
        abso3b = ds["abso3b"].data
        rayla = ds["rayla"].data
        raylb = ds["raylb"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(laytrop):
            speccomb = colamt[k, 0] + self.strrat[8] * colamt[k, 5]
            specmult = 8.0 * min(oneminus, colamt[k, 0] / speccomb)

            js = 1 + int(specmult)
            fs = np.mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00[k]
            fac010 = fs1 * fac10[k]
            fac100 = fs * fac00[k]
            fac110 = fs * fac10[k]
            fac001 = fs1 * fac01[k]
            fac011 = fs1 * fac11[k]
            fac101 = fs * fac01[k]
            fac111 = fs * fac11[k]

            ind01 = id0[k, 23] + js
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[k, 23] + js
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            inds = indself[k] - 1
            indf = indfor[k] - 1
            indsp = inds + 1
            indfp = indf + 1

            for j in range(NG24):
                taug[k, NS24 + j] = (
                    speccomb
                    * (
                        fac000 * absa[ind01, j]
                        + fac100 * absa[ind02, j]
                        + fac010 * absa[ind03, j]
                        + fac110 * absa[ind04, j]
                        + fac001 * absa[ind11, j]
                        + fac101 * absa[ind12, j]
                        + fac011 * absa[ind13, j]
                        + fac111 * absa[ind14, j]
                    )
                    + colamt[k, 2] * abso3a[j]
                    + colamt[k, 0]
                    * (
                        selffac[k]
                        * (
                            selfref[inds, j]
                            + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                        )
                        + forfac[k]
                        * (
                            forref[indf, j]
                            + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                        )
                    )
                )

                taur[k, NS24 + j] = colmol[k] * (
                    rayla[j, js - 1] + fs * (rayla[j, js] - rayla[j, js - 1])
                )

        for k in range(laytrop, nlay):
            ind01 = id0[k, 23] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 23] + 1
            ind12 = ind11 + 1

            for j in range(NG24):
                taug[k, NS24 + j] = (
                    colamt[k, 5]
                    * (
                        fac00[k] * absb[ind01, j]
                        + fac10[k] * absb[ind02, j]
                        + fac01[k] * absb[ind11, j]
                        + fac11[k] * absb[ind12, j]
                    )
                    + colamt[k, 2] * abso3b[j]
                )

                taur[k, NS24 + j] = colmol[k] * raylb[j]

        return taug, taur

    # The subroutine computes the optical depth in band 25:  16000-22650
    # cm-1 (low - h2o; high - nothing)

    def taumol25(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 25:  16000-22650 cm-1 (low - h2o; high - nothing)           !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb25_data.nc"))
        absa = ds["absa"].data
        abso3a = ds["abso3a"].data
        abso3b = ds["abso3b"].data
        rayl = ds["rayl"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            for j in range(NG25):
                taur[k, NS25 + j] = colmol[k] * rayl[j]

        for k in range(laytrop):
            ind01 = id0[k, 24] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 24] + 1
            ind12 = ind11 + 1

            for j in range(NG25):
                taug[k, NS25 + j] = (
                    colamt[k, 0]
                    * (
                        fac00[k] * absa[ind01, j]
                        + fac10[k] * absa[ind02, j]
                        + fac01[k] * absa[ind11, j]
                        + fac11[k] * absa[ind12, j]
                    )
                    + colamt[k, 2] * abso3a[j]
                )

        for k in range(laytrop, nlay):
            for j in range(NG25):
                taug[k, NS25 + j] = colamt[k, 2] * abso3b[j]

        return taug, taur

    # The subroutine computes the optical depth in band 26:  22650-29000
    # cm-1 (low - nothing; high - nothing)

    def taumol26(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 26:  22650-29000 cm-1 (low - nothing; high - nothing)       !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb26_data.nc"))
        rayl = ds["rayl"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            for j in range(NG26):
                taug[k, NS26 + j] = 0.0
                taur[k, NS26 + j] = colmol[k] * rayl[j]

        return taug, taur

    # The subroutine computes the optical depth in band 27:  29000-38000
    # cm-1 (low - o3; high - o3)

    def taumol27(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 27:  29000-38000 cm-1 (low - o3; high - o3)                 !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb27_data.nc"))
        absa = ds["absa"].data
        absb = ds["absb"].data
        rayl = ds["rayl"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            for j in range(NG27):
                taur[k, NS27 + j] = colmol[k] * rayl[j]

        for k in range(laytrop):
            ind01 = id0[k, 26] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 26] + 1
            ind12 = ind11 + 1

            for j in range(NG27):
                taug[k, NS27 + j] = colamt[k, 2] * (
                    fac00[k] * absa[ind01, j]
                    + fac10[k] * absa[ind02, j]
                    + fac01[k] * absa[ind11, j]
                    + fac11[k] * absa[ind12, j]
                )

        for k in range(laytrop, nlay):
            ind01 = id0[k, 26] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 26] + 1
            ind12 = ind11 + 1

            for j in range(NG27):
                taug[k, NS27 + j] = colamt[k, 2] * (
                    fac00[k] * absb[ind01, j]
                    + fac10[k] * absb[ind02, j]
                    + fac01[k] * absb[ind11, j]
                    + fac11[k] * absb[ind12, j]
                )

        return taug, taur

    # The subroutine computes the optical depth in band 28:  38000-50000
    # cm-1 (low - o3,o2; high - o3,o2)

    def taumol28(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 28:  38000-50000 cm-1 (low - o3,o2; high - o3,o2)           !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb28_data.nc"))
        absa = ds["absa"].data
        absb = ds["absb"].data
        rayl = ds["rayl"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            tauray = colmol[k] * rayl

            for j in range(NG28):
                taur[k, NS28 + j] = tauray

        for k in range(laytrop):
            speccomb = colamt[k, 2] + self.strrat[12] * colamt[k, 5]
            specmult = 8.0 * min(oneminus, colamt[k, 2] / speccomb)

            js = 1 + int(specmult)
            fs = np.mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00[k]
            fac010 = fs1 * fac10[k]
            fac100 = fs * fac00[k]
            fac110 = fs * fac10[k]
            fac001 = fs1 * fac01[k]
            fac011 = fs1 * fac11[k]
            fac101 = fs * fac01[k]
            fac111 = fs * fac11[k]

            ind01 = id0[k, 27] + js
            ind02 = ind01 + 1
            ind03 = ind01 + 9
            ind04 = ind01 + 10
            ind11 = id1[k, 27] + js
            ind12 = ind11 + 1
            ind13 = ind11 + 9
            ind14 = ind11 + 10

            for j in range(NG28):
                taug[k, NS28 + j] = speccomb * (
                    fac000 * absa[ind01, j]
                    + fac100 * absa[ind02, j]
                    + fac010 * absa[ind03, j]
                    + fac110 * absa[ind04, j]
                    + fac001 * absa[ind11, j]
                    + fac101 * absa[ind12, j]
                    + fac011 * absa[ind13, j]
                    + fac111 * absa[ind14, j]
                )

        for k in range(laytrop, nlay):
            speccomb = colamt[k, 2] + self.strrat[12] * colamt[k, 5]
            specmult = 4.0 * min(oneminus, colamt[k, 2] / speccomb)

            js = 1 + int(specmult)
            fs = np.mod(specmult, 1.0)
            fs1 = 1.0 - fs
            fac000 = fs1 * fac00[k]
            fac010 = fs1 * fac10[k]
            fac100 = fs * fac00[k]
            fac110 = fs * fac10[k]
            fac001 = fs1 * fac01[k]
            fac011 = fs1 * fac11[k]
            fac101 = fs * fac01[k]
            fac111 = fs * fac11[k]

            ind01 = id0[k, 27] + js
            ind02 = ind01 + 1
            ind03 = ind01 + 5
            ind04 = ind01 + 6
            ind11 = id1[k, 27] + js
            ind12 = ind11 + 1
            ind13 = ind11 + 5
            ind14 = ind11 + 6

            for j in range(NG28):
                taug[k, NS28 + j] = speccomb * (
                    fac000 * absb[ind01, j]
                    + fac100 * absb[ind02, j]
                    + fac010 * absb[ind03, j]
                    + fac110 * absb[ind04, j]
                    + fac001 * absb[ind11, j]
                    + fac101 * absb[ind12, j]
                    + fac011 * absb[ind13, j]
                    + fac111 * absb[ind14, j]
                )

        return taug, taur

    # The subroutine computes the optical depth in band 29:  820-2600
    # cm-1 (low - h2o; high - co2)

    def taumol29(
        self,
        colamt,
        colmol,
        fac00,
        fac01,
        fac10,
        fac11,
        jp,
        jt,
        jt1,
        laytrop,
        forfac,
        forfrac,
        indfor,
        selffac,
        selffrac,
        indself,
        nlay,
        id0,
        id1,
        taug,
        taur,
    ):

        #  ------------------------------------------------------------------  !
        #     band 29:  820-2600 cm-1 (low - h2o; high - co2)                  !
        #  ------------------------------------------------------------------  !
        #

        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb29_data.nc"))
        forref = ds["forref"].data
        absa = ds["absa"].data
        absb = ds["absb"].data
        selfref = ds["selfref"].data
        absh2o = ds["absh2o"].data
        absco2 = ds["absco2"].data
        rayl = ds["rayl"].data

        #  --- ...  compute the optical depth by interpolating in ln(pressure),
        #           temperature, and appropriate species.  below laytrop, the water
        #           vapor self-continuum is interpolated (in temperature) separately.

        for k in range(nlay):
            tauray = colmol[k] * rayl

            for j in range(NG29):
                taur[k, NS29 + j] = tauray

        for k in range(laytrop):
            ind01 = id0[k, 28] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 28] + 1
            ind12 = ind11 + 1

            inds = indself[k] - 1
            indf = indfor[k] - 1
            indsp = inds + 1
            indfp = indf + 1

            for j in range(NG29):
                taug[k, NS29 + j] = (
                    colamt[k, 0]
                    * (
                        (
                            fac00[k] * absa[ind01, j]
                            + fac10[k] * absa[ind02, j]
                            + fac01[k] * absa[ind11, j]
                            + fac11[k] * absa[ind12, j]
                        )
                        + selffac[k]
                        * (
                            selfref[inds, j]
                            + selffrac[k] * (selfref[indsp, j] - selfref[inds, j])
                        )
                        + forfac[k]
                        * (
                            forref[indf, j]
                            + forfrac[k] * (forref[indfp, j] - forref[indf, j])
                        )
                    )
                    + colamt[k, 1] * absco2[j]
                )

        for k in range(laytrop, nlay):
            ind01 = id0[k, 28] + 1
            ind02 = ind01 + 1
            ind11 = id1[k, 28] + 1
            ind12 = ind11 + 1

            for j in range(NG29):
                taug[k, NS29 + j] = (
                    colamt[k, 1]
                    * (
                        fac00[k] * absb[ind01, j]
                        + fac10[k] * absb[ind02, j]
                        + fac01[k] * absb[ind11, j]
                        + fac11[k] * absb[ind12, j]
                    )
                    + colamt[k, 0] * absh2o[j]
                )

        return taug, taur
