import numpy as np
import os
import sys

sys.path.insert(0, "..")
from phys_const import con_ttp, con_pi, con_g, con_rd, con_t0c, con_thgni
from radphysparam import lcrick, lcnorm, lnoprec


class CloudClass:
    VTAGCLD = "NCEP-Radiation_clouds    v5.1  Nov 2012 "
    gfac = 1.0e5 / con_g
    gord = con_g / con_rd
    NF_CLDS = 9
    NK_CLDS = 3
    ptopc = np.array([[1050.0, 650.0, 400.0, 0.0], [1050.0, 750.0, 500.0, 0.0]]).T
    climit = 0.001
    climit2 = 0.05
    ovcst = 1.0 - 1.0e-8
    reliq_def = 10.0
    reice_def = 50.0
    rrain_def = 1000.0
    rsnow_def = 250.0
    cldssa_def = 0.99
    cldasy_def = 0.84

    def __init__(self, si, NLAY, imp_physics, me, ivflip, icldflg, iovrsw, iovrlw):

        self.iovr = max(iovrsw, iovrlw)
        self.ivflip = ivflip

        if me == 0:
            print(self.VTAGCLD)  # print out version tag

        if icldflg == 0:
            print(" - Diagnostic Cloud Method has been discontinued")
        else:
            if me == 0:
                print("- Using Prognostic Cloud Method")
                if imp_physics == 99:
                    print("   --- Zhao/Carr/Sundqvist microphysics")
                elif imp_physics == 98:
                    print("   --- zhao/carr/sundqvist + pdf cloud")
                elif imp_physics == 11:
                    print("   --- GFDL Lin cloud microphysics")
                elif imp_physics == 8:
                    print("   --- Thompson cloud microphysics")
                elif imp_physics == 6:
                    print("   --- WSM6 cloud microphysics")
                elif imp_physics == 10:
                    print("   --- MG cloud microphysics")
                else:
                    raise ValueError(
                        "!!! ERROR in cloud microphysc specification!!!",
                        f"imp_physics (NP3D) = {imp_physics}",
                    )
        # Compute the top of BL cld (llyr), which is the topmost non
        # cld(low) layer for stratiform (at or above lowest 0.1 of the
        # atmosphere).

        if ivflip == 0:  # data from toa to sfc
            for k in range(NLAY - 1, 0, -1):
                kl = k
                if si[k] < 0.9e0:
                    break

            llyr = kl + 1
        else:  # data from sfc to top
            for k in range(1, NLAY):
                kl = k
                if si[k] < 0.9e0:
                    break

            llyr = kl

        self.llyr = llyr

    def return_initdata(self):
        outdict = {"llyr": self.llyr}
        return outdict

    def progcld1(
        self,
        plyr,
        plvl,
        tlyr,
        tvly,
        qlyr,
        qstl,
        rhly,
        clw,
        xlat,
        xlon,
        slmsk,
        dz,
        delp,
        IX,
        NLAY,
        NLP1,
        uni_cld,
        lmfshal,
        lmfdeep2,
        cldcov,
        effrl,
        effri,
        effrr,
        effrs,
        effr_in,
        iovrsw,
        iovrlw,
        ivflip,
        llyr,
    ):

        # =================   subprogram documentation block   ================ !
        #                                                                       !
        # subprogram:    progcld1    computes cloud related quantities using    !
        #   zhao/moorthi's prognostic cloud microphysics scheme.                !
        #                                                                       !
        # abstract:  this program computes cloud fractions from cloud           !
        #   condensates, calculates liquid/ice cloud droplet effective radius,  !
        #   and computes the low, mid, high, total and boundary layer cloud     !
        #   fractions and the vertical indices of low, mid, and high cloud      !
        #   top and base.  the three vertical cloud domains are set up in the   !
        #   initial subroutine "cld_init".                                      !
        #                                                                       !
        # usage:         call progcld1                                          !
        #                                                                       !
        # subprograms called:   gethml                                          !
        #                                                                       !
        # attributes:                                                           !
        #   language:   fortran 90                                              !
        #   machine:    ibm-sp, sgi                                             !
        #                                                                       !
        #                                                                       !
        #  ====================  definition of variables  ====================  !
        #                                                                       !
        # input variables:                                                      !
        #   plyr  (IX,NLAY) : model layer mean pressure in mb (100Pa)           !
        #   plvl  (IX,NLP1) : model level pressure in mb (100Pa)                !
        #   tlyr  (IX,NLAY) : model layer mean temperature in k                 !
        #   tvly  (IX,NLAY) : model layer virtual temperature in k              !
        #   qlyr  (IX,NLAY) : layer specific humidity in gm/gm                  !
        #   qstl  (IX,NLAY) : layer saturate humidity in gm/gm                  !
        #   rhly  (IX,NLAY) : layer relative humidity (=qlyr/qstl)              !
        #   clw   (IX,NLAY) : layer cloud condensate amount                     !
        #   xlat  (IX)      : grid latitude in radians, default to pi/2 -> -pi/2!
        #                     range, otherwise see in-line comment              !
        #   xlon  (IX)      : grid longitude in radians  (not used)             !
        #   slmsk (IX)      : sea/land mask array (sea:0,land:1,sea-ice:2)      !
        #   dz    (ix,nlay) : layer thickness (km)                              !
        #   delp  (ix,nlay) : model layer pressure thickness in mb (100Pa)      !
        #   IX              : horizontal dimention                              !
        #   NLAY,NLP1       : vertical layer/level dimensions                   !
        #   uni_cld         : logical - true for cloud fraction from shoc       !
        #   lmfshal         : logical - true for mass flux shallow convection   !
        #   lmfdeep2        : logical - true for mass flux deep convection      !
        #   cldcov          : layer cloud fraction (used when uni_cld=.true.    !
        #                                                                       !
        # output variables:                                                     !
        #   clouds(IX,NLAY,NF_CLDS) : cloud profiles                            !
        #      clouds(:,:,1) - layer total cloud fraction                       !
        #      clouds(:,:,2) - layer cloud liq water path         (g/m**2)      !
        #      clouds(:,:,3) - mean eff radius for liq cloud      (micron)      !
        #      clouds(:,:,4) - layer cloud ice water path         (g/m**2)      !
        #      clouds(:,:,5) - mean eff radius for ice cloud      (micron)      !
        #      clouds(:,:,6) - layer rain drop water path         not assigned  !
        #      clouds(:,:,7) - mean eff radius for rain drop      (micron)      !
        #  *** clouds(:,:,8) - layer snow flake water path        not assigned  !
        #      clouds(:,:,9) - mean eff radius for snow flake     (micron)      !
        #  *** fu's scheme need to be normalized by snow density (g/m**3/1.0e6) !
        #   clds  (IX,5)    : fraction of clouds for low, mid, hi, tot, bl      !
        #   mtop  (IX,3)    : vertical indices for low, mid, hi cloud tops      !
        #   mbot  (IX,3)    : vertical indices for low, mid, hi cloud bases     !
        #   de_lgth(ix)     : clouds decorrelation length (km)                  !
        #                                                                       !
        # module variables:                                                     !
        #   ivflip          : control flag of vertical index direction          !
        #                     =0: index from toa to surface                     !
        #                     =1: index from surface to toa                     !
        #   lmfshal         : mass-flux shallow conv scheme flag                !
        #   lmfdeep2        : scale-aware mass-flux deep conv scheme flag       !
        #   lcrick          : control flag for eliminating CRICK                !
        #                     =t: apply layer smoothing to eliminate CRICK      !
        #                     =f: do not apply layer smoothing                  !
        #   lcnorm          : control flag for in-cld condensate                !
        #                     =t: normalize cloud condensate                    !
        #                     =f: not normalize cloud condensate                !
        #                                                                       !
        #  ====================    end of description    =====================  !
        #
        #  ---  constant values

        cldtot = np.zeros((IX, NLAY))
        cldcnv = np.zeros((IX, NLAY))
        cwp = np.zeros((IX, NLAY))
        cip = np.zeros((IX, NLAY))
        crp = np.zeros((IX, NLAY))
        csp = np.zeros((IX, NLAY))
        rew = np.zeros((IX, NLAY))
        rei = np.zeros((IX, NLAY))
        res = np.zeros((IX, NLAY))
        rer = np.zeros((IX, NLAY))
        tem2d = np.zeros((IX, NLAY))
        clwf = np.zeros((IX, NLAY))

        ptop1 = np.zeros((IX, self.NK_CLDS + 1))
        rxlat = np.zeros(IX)

        clouds = np.zeros((IX, NLAY, self.NF_CLDS))
        de_lgth = np.zeros(IX)

        if effr_in:
            for k in range(NLAY):
                for i in range(IX):
                    cldtot[i, k] = 0.0
                    cldcnv[i, k] = 0.0
                    cwp[i, k] = 0.0
                    cip[i, k] = 0.0
                    crp[i, k] = 0.0
                    csp[i, k] = 0.0
                    rew[i, k] = effrl[i, k]
                    rei[i, k] = effri[i, k]
                    rer[i, k] = effrr[i, k]
                    res[i, k] = effrs[i, k]
                    tem2d[i, k] = min(1.0, max(0.0, (con_ttp - tlyr[i, k]) * 0.05))
                    clwf[i, k] = 0.0
        else:
            for k in range(NLAY):
                for i in range(IX):
                    cldtot[i, k] = 0.0
                    cldcnv[i, k] = 0.0
                    cwp[i, k] = 0.0
                    cip[i, k] = 0.0
                    crp[i, k] = 0.0
                    csp[i, k] = 0.0
                    rew[i, k] = self.reliq_def  # default liq  radius to 10   micron
                    rei[i, k] = self.reice_def  # default ice  radius to 50   micron
                    rer[i, k] = self.rrain_def  # default rain radius to 1000 micron
                    res[i, k] = self.rsnow_def  # default snow radius to 250  micron
                    tem2d[i, k] = min(1.0, max(0.0, (con_ttp - tlyr[i, k]) * 0.05))
                    clwf[i, k] = 0.0

        if lcrick:
            for i in range(IX):
                clwf[i, 0] = 0.75 * clw[i, 0] + 0.25 * clw[i, 1]
                clwf[i, NLAY] = 0.75 * clw[i, NLAY] + 0.25 * clw[i, NLAY - 2]
            for k in range(1, NLAY - 1):
                for i in range(IX):
                    clwf[i, k] = (
                        0.25 * clw[i, k - 1] + 0.5 * clw[i, k] + 0.25 * clw[i, k + 1]
                    )
        else:
            for k in range(NLAY):
                for i in range(IX):
                    clwf[i, k] = clw[i, k]

        # Find top pressure for each cloud domain for given latitude.
        #     ptopc(k,i): top presure of each cld domain (k=1-4 are sfc,L,m,h;
        #  ---  i=1,2 are low-lat (<45 degree) and pole regions)

        for i in range(IX):
            rxlat[i] = abs(xlat[i] / con_pi)  # if xlat in pi/2 -> -pi/2 range

        for id in range(4):
            tem1 = self.ptopc[id, 1] - self.ptopc[id, 0]

            for i in range(IX):
                ptop1[i, id] = self.ptopc[id, 0] + tem1 * max(0.0, 4.0 * rxlat[i] - 1.0)

        # Compute cloud liquid/ice condensate path in \f$ g/m^2 \f$ .
        for k in range(NLAY):
            for i in range(IX):
                clwt = max(0.0, clwf[i, k]) * self.gfac * delp[i, k]
                cip[i, k] = clwt * tem2d[i, k]
                cwp[i, k] = clwt - cip[i, k]

        # Compute effective liquid cloud droplet radius over land.

        if not effr_in:
            for i in range(IX):
                if round(slmsk[i]) == 1:
                    for k in range(NLAY):
                        rew[i, k] = 5.0 + 5.0 * tem2d[i, k]

        if uni_cld:  # use unified sgs clouds generated outside
            for k in range(NLAY):
                for i in range(IX):
                    cldtot[i, k] = cldcov[i, k]
        else:
            # Calculate layer cloud fraction.
            clwmin = 0.0
            if not lmfshal:
                for k in range(NLAY):
                    for i in range(IX):
                        clwt = 1.0e-6 * (plyr[i, k] * 0.001)
                        if clwf[i, k] > clwt:
                            onemrh = max(1.0e-10, 1.0 - rhly[i, k])
                            clwm = clwmin / max(0.01, plyr[i, k] * 0.001)

                            tem1 = min(
                                max(np.sqrt(np.sqrt(onemrh * qstl[i, k])), 0.0001), 1.0
                            )
                            tem1 = 2000.0 / tem1

                            value = max(min(tem1 * (clwf[i, k] - clwm), 50.0), 0.0)
                            tem2 = np.sqrt(np.sqrt(rhly[i, k]))

                            cldtot[i, k] = max(tem2 * (1.0 - np.exp(-value)), 0.0)
            else:
                for k in range(NLAY):
                    for i in range(IX):
                        clwt = 1.0e-6 * (plyr[i, k] * 0.001)

                        if clwf(i, k) > clwt:
                            onemrh = max(1.0e-10, 1.0 - rhly[i, k])
                            clwm = clwmin / max(0.01, plyr[i, k] * 0.001)

                            tem1 = min(
                                max((onemrh * qstl[i, k]) ** 0.49, 0.0001), 1.0
                            )  # jhan
                            if lmfdeep2:
                                tem1 = self.xrc3 / tem1
                            else:
                                tem1 = 100.0 / tem1

                            value = max(min(tem1 * (clwf[i, k] - clwm), 50.0), 0.0)
                            tem2 = np.sqrt(np.sqrt(rhly[i, k]))

                            cldtot[i, k] = max(tem2 * (1.0 - np.exp(-value)), 0.0)

        for k in range(NLAY):
            for i in range(IX):
                if cldtot[i, k] < self.climit:
                    cldtot[i, k] = 0.0
                    cwp[i, k] = 0.0
                    cip[i, k] = 0.0
                    crp[i, k] = 0.0
                    csp[i, k] = 0.0

        if lcnorm:
            for k in range(NLAY):
                for i in range(IX):
                    if cldtot[i, k] >= self.climit:
                        tem1 = 1.0 / max(self.climit2, cldtot[i, k])
                        cwp[i, k] = cwp[i, k] * tem1
                        cip[i, k] = cip[i, k] * tem1
                        crp[i, k] = crp[i, k] * tem1
                        csp[i, k] = csp[i, k] * tem1

        # Compute effective ice cloud droplet radius following Heymsfield
        #    and McFarquhar (1996) \cite heymsfield_and_mcfarquhar_1996.

        if not effr_in:
            for k in range(NLAY):
                for i in range(IX):
                    tem2 = tlyr[i, k] - con_ttp

                    if cip[i, k] > 0.0:
                        tem3 = (
                            self.gord
                            * cip[i, k]
                            * plyr[i, k]
                            / (delp[i, k] * tvly[i, k])
                        )

                        if tem2 < -50.0:
                            rei[i, k] = (1250.0 / 9.917) * tem3 ** 0.109
                        elif tem2 < -40.0:
                            rei[i, k] = (1250.0 / 9.337) * tem3 ** 0.08
                        elif tem2 < -30.0:
                            rei[i, k] = (1250.0 / 9.208) * tem3 ** 0.055
                        else:
                            rei[i, k] = (1250.0 / 9.387) * tem3 ** 0.031

                        rei[i, k] = max(10.0, min(rei[i, k], 150.0))

        for k in range(NLAY):
            for i in range(IX):
                clouds[i, k, 0] = cldtot[i, k]
                clouds[i, k, 1] = cwp[i, k]
                clouds[i, k, 2] = rew[i, k]
                clouds[i, k, 3] = cip[i, k]
                clouds[i, k, 4] = rei[i, k]
                clouds[i, k, 6] = rer[i, k]
                clouds[i, k, 8] = res[i, k]

        #  --- ...  estimate clouds decorrelation length in km
        #           this is only a tentative test, need to consider change later

        if self.iovr == 3:
            for i in range(IX):
                de_lgth[i] = max(0.6, 2.78 - 4.6 * rxlat[i])

        # Call gethml() to compute low,mid,high,total, and boundary layer
        #    cloud fractions and clouds top/bottom layer indices for low, mid,
        #    and high clouds.
        # ---  compute low, mid, high, total, and boundary layer cloud fractions
        #      and clouds top/bottom layer indices for low, mid, and high clouds.
        #      The three cloud domain boundaries are defined by ptopc.  The cloud
        #      overlapping method is defined by control flag 'iovr', which may
        #      be different for lw and sw radiation programs.

        clds, mtop, mbot = self.gethml(
            plyr, ptop1, cldtot, cldcnv, dz, de_lgth, IX, NLAY
        )

        return clouds, clds, mtop, mbot, de_lgth

    def progcld2(
        self,
        plyr,
        plvl,
        tlyr,
        tvly,
        qlyr,
        qstl,
        rhly,
        clw,
        xlat,
        xlon,
        slmsk,
        dz,
        delp,
        f_ice,
        f_rain,
        r_rime,
        flgmin,
        IX,
        NLAY,
        NLP1,
        lmfshal,
        lmfdeep2,
        ivflip,
        iovrsw,
        iovrlw,
    ):
        # =================   subprogram documentation block   ================ !
        #                                                                       !
        # subprogram:    progcld2    computes cloud related quantities using    !
        #   ferrier's prognostic cloud microphysics scheme.                     !
        #                                                                       !
        # abstract:  this program computes cloud fractions from cloud           !
        #   condensates, calculates liquid/ice cloud droplet effective radius,  !
        #   and computes the low, mid, high, total and boundary layer cloud     !
        #   fractions and the vertical indices of low, mid, and high cloud      !
        #   top and base.  the three vertical cloud domains are set up in the   !
        #   initial subroutine "cld_init".                                      !
        #                                                                       !
        # usage:         call progcld2                                          !
        #                                                                       !
        # subprograms called:   gethml                                          !
        #                                                                       !
        # attributes:                                                           !
        #   language:   fortran 90                                              !
        #   machine:    ibm-sp, sgi                                             !
        #                                                                       !
        #                                                                       !
        #  ====================  definition of variables  ====================  !
        #                                                                       !
        # input variables:                                                      !
        #   plyr  (IX,NLAY) : model layer mean pressure in mb (100Pa)           !
        #   plvl  (IX,NLP1) : model level pressure in mb (100Pa)                !
        #   tlyr  (IX,NLAY) : model layer mean temperature in k                 !
        #   tvly  (IX,NLAY) : model layer virtual temperature in k              !
        #   qlyr  (IX,NLAY) : layer specific humidity in gm/gm                  !
        #   qstl  (IX,NLAY) : layer saturate humidity in gm/gm                  !
        #   rhly  (IX,NLAY) : layer relative humidity (=qlyr/qstl)              !
        #   clw   (IX,NLAY) : layer cloud condensate amount                     !
        #   f_ice (IX,NLAY) : fraction of layer cloud ice  (ferrier micro-phys) !
        #   f_rain(IX,NLAY) : fraction of layer rain water (ferrier micro-phys) !
        #   r_rime(IX,NLAY) : mass ratio of total ice to unrimed ice (>=1)      !
        #   flgmin(IX)      : minimim large ice fraction                        !
        #   xlat  (IX)      : grid latitude in radians, default to pi/2 -> -pi/2!
        #                     range, otherwise see in-line comment              !
        #   xlon  (IX)      : grid longitude in radians  (not used)             !
        #   slmsk (IX)      : sea/land mask array (sea:0,land:1,sea-ice:2)      !
        #   dz    (ix,nlay) : layer thickness (km)                              !
        #   delp  (ix,nlay) : model layer pressure thickness in mb (100Pa)      !
        #   IX              : horizontal dimention                              !
        #   NLAY,NLP1       : vertical layer/level dimensions                   !
        #                                                                       !
        # output variables:                                                     !
        #   clouds(IX,NLAY,NF_CLDS) : cloud profiles                            !
        #      clouds(:,:,1) - layer total cloud fraction                       !
        #      clouds(:,:,2) - layer cloud liq water path         (g/m**2)      !
        #      clouds(:,:,3) - mean eff radius for liq cloud      (micron)      !
        #      clouds(:,:,4) - layer cloud ice water path         (g/m**2)      !
        #      clouds(:,:,5) - mean eff radius for ice cloud      (micron)      !
        #      clouds(:,:,6) - layer rain drop water path         (g/m**2)      !
        #      clouds(:,:,7) - mean eff radius for rain drop      (micron)      !
        #  *** clouds(:,:,8) - layer snow flake water path        (g/m**2)      !
        #      clouds(:,:,9) - mean eff radius for snow flake     (micron)      !
        #  *** fu's scheme need to be normalized by snow density (g/m**3/1.0e6) !
        #   clds  (IX,5)    : fraction of clouds for low, mid, hi, tot, bl      !
        #   mtop  (IX,3)    : vertical indices for low, mid, hi cloud tops      !
        #   mbot  (IX,3)    : vertical indices for low, mid, hi cloud bases     !
        #   de_lgth(ix)     : clouds decorrelation length (km)                  !
        #                                                                       !
        # external module variables:                                            !
        #   ivflip          : control flag of vertical index direction          !
        #                     =0: index from toa to surface                     !
        #                     =1: index from surface to toa                     !
        #   lmfshal         : mass-flux shallow conv scheme flag                !
        #   lmfdeep2        : scale-aware mass-flux deep conv scheme flag       !
        #   lcrick          : control flag for eliminating CRICK                !
        #                     =t: apply layer smoothing to eliminate CRICK      !
        #                     =f: do not apply layer smoothing                  !
        #   lcnorm          : control flag for in-cld condensate                !
        #                     =t: normalize cloud condensate                    !
        #                     =f: not normalize cloud condensate                !
        #   lnoprec         : precip effect in radiation flag (ferrier scheme)  !
        #                     =t: snow/rain has no impact on radiation          !
        #                     =f: snow/rain has impact on radiation             !
        #                                                                       !
        #  ====================    end of description    =====================  !
        #

        cldtot = np.zeros((IX, NLAY))
        cldcnv = np.zeros((IX, NLAY))
        cwp = np.zeros((IX, NLAY))
        cip = np.zeros((IX, NLAY))
        crp = np.zeros((IX, NLAY))
        csp = np.zeros((IX, NLAY))
        rew = np.zeros((IX, NLAY))
        rei = np.zeros((IX, NLAY))
        res = np.zeros((IX, NLAY))
        rer = np.zeros((IX, NLAY))
        tem2d = np.zeros((IX, NLAY))
        clw2 = np.zeros((IX, NLAY))
        qcwat = np.zeros((IX, NLAY))
        qcice = np.zeros((IX, NLAY))
        qrain = np.zeros((IX, NLAY))
        fcice = np.zeros((IX, NLAY))
        frain = np.zeros((IX, NLAY))
        rrime = np.zeros((IX, NLAY))
        rsden = np.zeros((IX, NLAY))
        clwf = np.zeros((IX, NLAY))

        ptop1 = np.zerso((IX, self.NK_CLDS + 1))
        rxlat = np.zeros(IX)

        clouds = np.zeros((IX, NLAY, self.NF_CLDS))
        de_lgth = np.zeros(IX)

        for k in range(NLAY):
            for i in range(IX):
                rew[i, k] = self.reliq_def  # default liq radius to 10 micron
                rei[i, k] = self.reice_def  # default ice radius to 50 micron
                rer[i, k] = self.rrain_def  # default rain radius to 1000 micron
                res[i, k] = self.rsnow_def  # default snow radius to 250 micron
                fcice[i, k] = max(0.0, min(1.0, f_ice[i, k]))
                frain[i, k] = max(0.0, min(1.0, f_rain[i, k]))
                rrime[i, k] = max(1.0, r_rime[i, k])
                tem2d[i, k] = tlyr[i, k] - con_t0c

        if lcrick:
            for i in range(IX):
                clwf[i, 0] = 0.75 * clw[i, 0] + 0.25 * clw[i, 1]
                clwf[i, NLAY] = 0.75 * clw[i, NLAY] + 0.25 * clw[i, NLAY - 1]

            for k in range(1, NLAY - 1):
                for i in range(IX):
                    clwf[i, k] = (
                        0.25 * clw[i, k - 1] + 0.5 * clw[i, k] + 0.25 * clw[i, k + 1]
                    )
        else:
            for k in range(NLAY):
                for i in range(IX):
                    clwf[i, k] = clw[i, k]

        # Find top pressure (ptopc) for each cloud domain for given latitude.
        #   - ptopc(k,i): top pressure of each cld domain (k=1-4 are sfc,l,m,
        #     h; i=1,2 are low-lat (<45 degree) and pole regions)

        for i in range(IX):
            rxlat[i] = abs(xlat[i] / con_pi)  # if xlat in pi/2 -> -pi/2 range

        for id in range(4):
            tem1 = self.ptopc[id, 1] - self.ptopc[id, 0]

            for i in range(IX):
                ptop1[i, id] = self.ptopc[id, 0] + tem1 * max(0.0, 4.0 * rxlat[i] - 1.0)

        # -# Seperate cloud condensate into liquid, ice, and rain types, and
        # save the liquid+ice condensate in array clw2 for later calculation
        #  of cloud fraction.

        for k in range(NLAY):
            for i in range(IX):
                if tem2d[i, k] > -40.0:
                    qcice[i, k] = clwf[i, k] * fcice[i, k]
                    tem1 = clwf[i, k] - qcice[i, k]
                    qrain[i, k] = tem1 * frain[i, k]
                    qcwat[i, k] = tem1 - qrain[i, k]
                    clw2[i, k] = qcwat[i, k] + qcice[i, k]
                else:
                    qcice[i, k] = clwf[i, k]
                    qrain[i, k] = 0.0
                    qcwat[i, k] = 0.0
                    clw2[i, k] = clwf[i, k]

        # -# Call module_microphysics::rsipath2(), in Ferrier's scheme, to
        # compute layer's cloud liquid, ice, rain, and snow water condensate
        # path and the partical effective radius for liquid droplet, rain drop,
        # and snow flake.
        cwp, cip, crp, csp, rew, rer, res, rsden = rsipath2(
            plyr, plvl, tlyr, qlyr, qcwat, qcice, qrain, rrime, IX, NLAY, ivflip, flgmin
        )

        for k in range(NLAY):
            for i in range(IX):
                tem2d[i, k] = (con_g * plyr[i, k]) / (con_rd * delp[i, k])

        # Calculate layer cloud fraction.

        clwmin = 0.0e-6
        if not lmfshal:
            for k in range(NLAY):
                for i in range(IX):
                    clwt = 2.0e-6 * (plyr[i, k] * 0.001)

                    if clw2[i, k] > clwt:
                        onemrh = max(1.0e-10, 1.0 - rhly[i, k])
                        clwm = clwmin / max(0.01, plyr[i, k] * 0.001)

                        tem1 = min(
                            max(np.sqrt(np.sqrt(onemrh * qstl[i, k])), 0.0001), 1.0
                        )
                        tem1 = 2000.0 / tem1

                        value = max(min(tem1 * (clw2[i, k] - clwm), 50.0), 0.0)
                        tem2 = np.sqrt(np.sqrt(rhly[i, k]))

                        cldtot[i, k] = max(tem2 * (1.0 - np.exp(-value)), 0.0)
        else:
            for k in range(NLAY):
                for i in range(IX):
                    clwt = 2.0e-6 * (plyr[i, k] * 0.001)

                    if clw2[i, k] > clwt:
                        onemrh = max(1.0e-10, 1.0 - rhly[i, k])
                        clwm = clwmin / max(0.01, plyr[i, k] * 0.001)

                        tem1 = min(
                            max((onemrh * qstl[i, k]) ** 0.49, 0.0001), 1.0
                        )  # jhan
                        if lmfdeep2:
                            tem1 = self.xrc3 / tem1
                        else:
                            tem1 = 100.0 / tem1

                        value = max(min(tem1 * (clw2[i, k] - clwm), 50.0), 0.0)
                        tem2 = np.sqrt(np.sqrt(rhly[i, k]))

                        cldtot[i, k] = max(tem2 * (1.0 - np.exp(-value)), 0.0)

        for k in range(NLAY):
            for i in range(IX):
                if cldtot[i, k] < self.climit:
                    cldtot[i, k] = 0.0
                    cwp[i, k] = 0.0
                    cip[i, k] = 0.0
                    crp[i, k] = 0.0
                    csp[i, k] = 0.0

        #     When lnoprec = .true. snow/rain has no impact on radiation
        if lnoprec:
            for k in range(NLAY):
                for i in range(IX):
                    crp[i, k] = 0.0
                    csp[i, k] = 0.0

        if lcnorm:
            for k in range(NLAY):
                for i in range(IX):
                    if cldtot[i, k] >= self.climit:
                        tem1 = 1.0 / max(self.climit2, cldtot[i, k])
                        cwp[i, k] = cwp[i, k] * tem1
                        cip[i, k] = cip[i, k] * tem1
                        crp[i, k] = crp[i, k] * tem1
                        csp[i, k] = csp[i, k] * tem1

        # Calculate effective ice cloud droplet radius.
        for k in range(NLAY):
            for i in range(IX):
                tem1 = tlyr[i, k] - con_ttp
                tem2 = cip[i, k]

                if tem2 > 0.0:
                    tem3 = tem2d[i, k] * tem2 / tvly[i, k]

                    if tem1 < -50.0:
                        rei[i, k] = (1250.0 / 9.917) * tem3 ** 0.109
                    elif tem1 < -40.0:
                        rei[i, k] = (1250.0 / 9.337) * tem3 ** 0.08
                    elif tem1 < -30.0:
                        rei[i, k] = (1250.0 / 9.208) * tem3 ** 0.055
                    else:
                        rei[i, k] = (1250.0 / 9.387) * tem3 ** 0.031

                    rei[i, k] = max(10.0, min(rei[i, k], 300.0))

        for k in range(NLAY):
            for i in range(IX):
                clouds[i, k, 0] = cldtot[i, k]
                clouds[i, k, 1] = cwp[i, k]
                clouds[i, k, 2] = rew[i, k]
                clouds[i, k, 3] = cip[i, k]
                clouds[i, k, 4] = rei[i, k]
                clouds[i, k, 5] = crp[i, k]
                clouds[i, k, 6] = rer[i, k]
                clouds[i, k, 7] = csp[i, k] * rsden[i, k]  # fu's scheme
                clouds[i, k, 8] = res[i, k]

        #  --- ...  estimate clouds decorrelation length in km
        #           this is only a tentative test, need to consider change later

        if self.iovr == 3:
            for i in range(IX):
                de_lgth[i] = max(0.6, 2.78 - 4.6 * rxlat[i])

        # -# Call gethml(), to compute low, mid, high, total, and boundary
        # layer cloud fractions and clouds top/bottom layer indices for low,
        # mid, and high clouds.
        #      The three cloud domain boundaries are defined by ptopc.  The cloud
        #      overlapping method is defined by control flag 'iovr', which may
        #      be different for lw and sw radiation programs.

        clds, mtop, mbot = self.gethml(
            plyr, ptop1, cldtot, cldcnv, dz, de_lgth, IX, NLAY
        )

        return clouds, clds, mtop, mbot, de_lgth

    def progcld3(
        self,
        plyr,
        plvl,
        tlyr,
        tvly,
        qlyr,
        qstl,
        rhly,
        clw,
        cnvw,
        cnvc,
        xlat,
        xlon,
        slmsk,
        dz,
        delp,
        ix,
        nlay,
        nlp1,
        deltaq,
        sup,
        kdt,
        me,
        iovrsw,
        iovrlw,
    ):
        # =================   subprogram documentation block   ================ !
        #                                                                       !
        # subprogram:    progcld3    computes cloud related quantities using    !
        #   zhao/moorthi's prognostic cloud microphysics scheme.                !
        #                                                                       !
        # abstract:  this program computes cloud fractions from cloud           !
        #   condensates, calculates liquid/ice cloud droplet effective radius,  !
        #   and computes the low, mid, high, total and boundary layer cloud     !
        #   fractions and the vertical indices of low, mid, and high cloud      !
        #   top and base.  the three vertical cloud domains are set up in the   !
        #   initial subroutine "cld_init".                                      !
        #                                                                       !
        # usage:         call progcld3                                          !
        #                                                                       !
        # subprograms called:   gethml                                          !
        #                                                                       !
        # attributes:                                                           !
        #   language:   fortran 90                                              !
        #   machine:    ibm-sp, sgi                                             !
        #                                                                       !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        # input variables:                                                      !
        #   plyr  (ix,nlay) : model layer mean pressure in mb (100pa)           !
        #   plvl  (ix,nlp1) : model level pressure in mb (100pa)                !
        #   tlyr  (ix,nlay) : model layer mean temperature in k                 !
        #   tvly  (ix,nlay) : model layer virtual temperature in k              !
        #   qlyr  (ix,nlay) : layer specific humidity in gm/gm                  !
        #   qstl  (ix,nlay) : layer saturate humidity in gm/gm                  !
        #   rhly  (ix,nlay) : layer relative humidity (=qlyr/qstl)              !
        #   clw   (ix,nlay) : layer cloud condensate amount                     !
        #   xlat  (ix)      : grid latitude in radians, default to pi/2 -> -pi/2!
        #                     range, otherwise see in-line comment              !
        #   xlon  (ix)      : grid longitude in radians  (not used)             !
        #   slmsk (ix)      : sea/land mask array (sea:0,land:1,sea-ice:2)      !
        #   dz    (ix,nlay) : layer thickness (km)                              !
        #   delp  (ix,nlay) : model layer pressure thickness in mb (100Pa)      !
        #   ix              : horizontal dimention                              !
        #   nlay,nlp1       : vertical layer/level dimensions                   !
        #   cnvw  (ix,nlay) : layer convective cloud condensate                 !
        #   cnvc  (ix,nlay) : layer convective cloud cover                      !
        #   deltaq(ix,nlay) : half total water distribution width               !
        #   sup             : supersaturation                                   !   #
        #                                                                       !
        # output variables:                                                     !
        #   clouds(ix,nlay,nf_clds) : cloud profiles                            !
        #      clouds(:,:,1) - layer total cloud fraction                       !
        #      clouds(:,:,2) - layer cloud liq water path         (g/m**2)      !
        #      clouds(:,:,3) - mean eff radius for liq cloud      (micron)      !
        #      clouds(:,:,4) - layer cloud ice water path         (g/m**2)      !
        #      clouds(:,:,5) - mean eff radius for ice cloud      (micron)      !
        #      clouds(:,:,6) - layer rain drop water path         not assigned  !
        #      clouds(:,:,7) - mean eff radius for rain drop      (micron)      !
        #  *** clouds(:,:,8) - layer snow flake water path        not assigned  !
        #      clouds(:,:,9) - mean eff radius for snow flake     (micron)      !
        #  *** fu's scheme need to be normalized by snow density (g/m**3/1.0e6) !
        #   clds  (ix,5)    : fraction of clouds for low, mid, hi, tot, bl      !
        #   mtop  (ix,3)    : vertical indices for low, mid, hi cloud tops      !
        #   mbot  (ix,3)    : vertical indices for low, mid, hi cloud bases     !
        #   de_lgth(ix)     : clouds decorrelation length (km)                  !
        #                                                                       !
        # module variables:                                                     !
        #   ivflip          : control flag of vertical index direction          !
        #                     =0: index from toa to surface                     !
        #                     =1: index from surface to toa                     !
        #   lcrick          : control flag for eliminating crick                !
        #                     =t: apply layer smoothing to eliminate crick      !
        #                     =f: do not apply layer smoothing                  !
        #   lcnorm          : control flag for in-cld condensate                !
        #                     =t: normalize cloud condensate                    !
        #                     =f: not normalize cloud condensate                !
        #                                                                       !
        #  ====================    end of description    =====================  !
        #

        cldtot = np.zeros((ix, nlay))
        cldcnv = np.zeros((ix, nlay))
        cwp = np.zeros((ix, nlay))
        cip = np.zeros((ix, nlay))
        crp = np.zeros((ix, nlay))
        csp = np.zeros((ix, nlay))
        rew = np.zeros((ix, nlay))
        rei = np.zeros((ix, nlay))
        res = np.zeros((ix, nlay))
        rer = np.zeros((ix, nlay))
        tem2d = np.zeros((ix, nlay))
        clwf = np.zeros((ix, nlay))

        ptop1 = np.zerso((ix, self.NK_CLDS + 1))
        rxlat = np.zeros(ix)

        clouds = np.zeros((ix, nlay, self.NF_CLDS))
        de_lgth = np.zeros(ix)

        for k in range(nlay):
            for i in range(ix):
                rew[i, k] = self.reliq_def  # default liq radius to 10 micron
                rei[i, k] = self.reice_def  # default ice radius to 50 micron
                rer[i, k] = self.rrain_def  # default rain radius to 1000 micron
                res[i, k] = self.rsnow_def  # default snow radius to 250 micron
                tem2d[i, k] = min(1.0, max(0.0, (con_ttp - tlyr(i, k)) * 0.05))
                clwf[i, k] = 0.0

        if lcrick:
            for i in range(ix):
                clwf[i, 0] = 0.75 * clw[i, 0] + 0.25 * clw[i, 1]
                clwf[i, nlay] = 0.75 * clw[i, nlay] + 0.25 * clw[i, nlay - 1]
            for k in range(1, nlay - 1):
                for i in range(ix):
                    clwf[i, k] = (
                        0.25 * clw[i, k - 1] + 0.5 * clw[i, k] + 0.25 * clw[i, k + 1]
                    )
        else:
            for k in range(nlay):
                for i in range(ix):
                    clwf[i, k] = clw[i, k]

        if kdt == 1:
            for k in range(nlay):
                for i in range(ix):
                    deltaq[i, k] = (1.0 - 0.95) * qstl[i, k]

        # -# Find top pressure (ptopc) for each cloud domain for given latitude.
        #    ptopc(k,i): top presure of each cld domain (k=1-4 are sfc,l,m,h;
        # ---  i=1,2 are low-lat (<45 degree) and pole regions)

        for i in range(ix):
            rxlat[i] = abs(xlat[i] / con_pi)  # if xlat in pi/2 -> -pi/2 range

        for id in range(4):
            tem1 = self.ptopc[id, 1] - self.ptopc[id, 0]

            for i in range(ix):
                ptop1[i, id] = self.ptopc[id, 0] + tem1 * max(0.0, 4.0 * rxlat[i] - 1.0)

        # -# Calculate liquid/ice condensate path in \f$ g/m^2 \f$

        for k in range(nlay):
            for i in range(ix):
                clwt = max(0.0, (clwf[i, k] + cnvw[i, k])) * self.gfac * delp[i, k]
                cip[i, k] = clwt * tem2d[i, k]
                cwp[i, k] = clwt - cip[i, k]

        # -# Calculate effective liquid cloud droplet radius over land.

        for i in range(ix):
            if round(slmsk[i]) == 1:
                for k in range(nlay):
                    rew[i, k] = 5.0 + 5.0 * tem2d[i, k]

        # -# Calculate layer cloud fraction.

        for k in range(nlay):
            for i in range(ix):
                tem1 = tlyr[i, k] - 273.16
                if tem1 < con_thgni:  # for pure ice, has to be consistent with gscond
                    qsc = sup * qstl[i, k]
                    rhs = sup
                else:
                    qsc = qstl[i, k]
                    rhs = 1.0

                if rhly[i, k] >= rhs:
                    cldtot[i, k] = 1.0
                else:
                    qtmp = qlyr[i, k] + clwf[i, k] - qsc
                    if deltaq[i, k] > self.epsq:
                        if qtmp <= -deltaq[i, k]:
                            cldtot[i, k] = 0.0
                        elif qtmp >= deltaq[i, k]:
                            cldtot[i, k] = 1.0
                        else:
                            cldtot[i, k] = 0.5 * qtmp / deltaq[i, k] + 0.5
                            cldtot[i, k] = max(cldtot[i, k], 0.0)
                            cldtot[i, k] = min(cldtot[i, k], 1.0)
                    else:
                        if qtmp > 0.0:
                            cldtot[i, k] = 1.0
                        else:
                            cldtot[i, k] = 0.0

                cldtot[i, k] = cnvc[i, k] + (1 - cnvc[i, k]) * cldtot[i, k]
                cldtot[i, k] = max(cldtot[i, k], 0.0)
                cldtot[i, k] = min(cldtot[i, k], 1.0)

        for k in range(nlay):
            for i in range(ix):
                if cldtot[i, k] < self.climit:
                    cldtot[i, k] = 0.0
                    cwp[i, k] = 0.0
                    cip[i, k] = 0.0
                    crp[i, k] = 0.0
                    csp[i, k] = 0.0

        if lcnorm:
            for k in range(nlay):
                for i in range(ix):
                    if cldtot[i, k] >= self.climit:
                        tem1 = 1.0 / max(self.climit2, cldtot[i, k])
                        cwp[i, k] = cwp[i, k] * tem1
                        cip[i, k] = cip[i, k] * tem1
                        crp[i, k] = crp[i, k] * tem1
                        csp[i, k] = csp[i, k] * tem1

        # -# Calculate effective ice cloud droplet radius following Heymsfield
        #    and McFarquhar (1996) \cite heymsfield_and_mcfarquhar_1996.

        for k in range(nlay):
            for i in range(ix):
                tem2 = tlyr[i, k] - con_ttp

                if cip[i, k] > 0.0:
                    tem3 = (
                        self.gord * cip[i, k] * plyr[i, k] / (delp[i, k] * tvly[i, k])
                    )

                    if tem2 < -50.0:
                        rei[i, k] = (1250.0 / 9.917) * tem3 ** 0.109
                    elif tem2 < -40.0:
                        rei[i, k] = (1250.0 / 9.337) * tem3 ** 0.08
                    elif tem2 < -30.0:
                        rei[i, k] = (1250.0 / 9.208) * tem3 ** 0.055
                    else:
                        rei[i, k] = (1250.0 / 9.387) * tem3 ** 0.031

                    rei[i, k] = max(10.0, min(rei[i, k], 150.0))

        for k in range(nlay):
            for i in range(ix):
                clouds[i, k, 0] = cldtot[i, k]
                clouds[i, k, 1] = cwp[i, k]
                clouds[i, k, 2] = rew[i, k]
                clouds[i, k, 3] = cip[i, k]
                clouds[i, k, 4] = rei[i, k]
                clouds[i, k, 6] = rer[i, k]
                clouds[i, k, 8] = res[i, k]

        #  --- ...  estimate clouds decorrelation length in km
        #           this is only a tentative test, need to consider change later

        if self.iovr == 3:
            for i in range(ix):
                de_lgth[i] = max(0.6, 2.78 - 4.6 * rxlat[i])

        # -# Call gethml() to compute low,mid,high,total, and boundary layer
        # cloud fractions and clouds top/bottom layer indices for low, mid,
        # and high clouds.
        #      the three cloud domain boundaries are defined by ptopc.  the cloud
        #      overlapping method is defined by control flag 'iovr', which may
        #      be different for lw and sw radiation programs.

        clds, mtop, mbot = self.gethml(
            plyr, ptop1, cldtot, cldcnv, dz, de_lgth, ix, nlay
        )

        return clouds, clds, mtop, mbot, de_lgth

    def progcld4(
        self,
        plyr,
        plvl,
        tlyr,
        tvly,
        qlyr,
        qstl,
        rhly,
        clw,
        cnvw,
        cnvc,
        xlat,
        xlon,
        slmsk,
        cldtot,
        dz,
        delp,
        IX,
        NLAY,
        NLP1,
    ):
        # =================   subprogram documentation block   ================ !
        #                                                                       !
        # subprogram:    progcld4    computes cloud related quantities using    !
        #   GFDL Lin MP prognostic cloud microphysics scheme.                   !
        #                                                                       !
        # abstract:  this program computes cloud fractions from cloud           !
        #   condensates, calculates liquid/ice cloud droplet effective radius,  !
        #   and computes the low, mid, high, total and boundary layer cloud     !
        #   fractions and the vertical indices of low, mid, and high cloud      !
        #   top and base.  the three vertical cloud domains are set up in the   !
        #   initial subroutine "cld_init".                                      !
        #                                                                       !
        # usage:         call progcld4                                          !
        #                                                                       !
        # subprograms called:   gethml                                          !
        #                                                                       !
        # attributes:                                                           !
        #   language:   fortran 90                                              !
        #   machine:    ibm-sp, sgi                                             !
        #                                                                       !
        #                                                                       !
        #  ====================  definition of variables  ====================  !
        #                                                                       !
        # input variables:                                                      !
        #   plyr  (IX,NLAY) : model layer mean pressure in mb (100Pa)           !
        #   plvl  (IX,NLP1) : model level pressure in mb (100Pa)                !
        #   tlyr  (IX,NLAY) : model layer mean temperature in k                 !
        #   tvly  (IX,NLAY) : model layer virtual temperature in k              !
        #   qlyr  (IX,NLAY) : layer specific humidity in gm/gm                  !
        #   qstl  (IX,NLAY) : layer saturate humidity in gm/gm                  !
        #   rhly  (IX,NLAY) : layer relative humidity (=qlyr/qstl)              !
        #   clw   (IX,NLAY) : layer cloud condensate amount                     !
        #   cnvw  (IX,NLAY) : layer convective cloud condensate                 !
        #   cnvc  (IX,NLAY) : layer convective cloud cover                      !
        #   xlat  (IX)      : grid latitude in radians, default to pi/2 -> -pi/2!
        #                     range, otherwise see in-line comment              !
        #   xlon  (IX)      : grid longitude in radians  (not used)             !
        #   slmsk (IX)      : sea/land mask array (sea:0,land:1,sea-ice:2)      !
        #   dz    (ix,nlay) : layer thickness (km)                              !
        #   delp  (ix,nlay) : model layer pressure thickness in mb (100Pa)      !
        #   IX              : horizontal dimention                              !
        #   NLAY,NLP1       : vertical layer/level dimensions                   !
        #                                                                       !
        # output variables:                                                     !
        #   clouds(IX,NLAY,NF_CLDS) : cloud profiles                            !
        #      clouds(:,:,1) - layer total cloud fraction                       !
        #      clouds(:,:,2) - layer cloud liq water path         (g/m**2)      !
        #      clouds(:,:,3) - mean eff radius for liq cloud      (micron)      !
        #      clouds(:,:,4) - layer cloud ice water path         (g/m**2)      !
        #      clouds(:,:,5) - mean eff radius for ice cloud      (micron)      !
        #      clouds(:,:,6) - layer rain drop water path         not assigned  !
        #      clouds(:,:,7) - mean eff radius for rain drop      (micron)      !
        #  *** clouds(:,:,8) - layer snow flake water path        not assigned  !
        #      clouds(:,:,9) - mean eff radius for snow flake     (micron)      !
        #  *** fu's scheme need to be normalized by snow density (g/m**3/1.0e6) !
        #   clds  (IX,5)    : fraction of clouds for low, mid, hi, tot, bl      !
        #   mtop  (IX,3)    : vertical indices for low, mid, hi cloud tops      !
        #   mbot  (IX,3)    : vertical indices for low, mid, hi cloud bases     !
        #   de_lgth(ix)     : clouds decorrelation length (km)                  !
        #                                                                       !
        # module variables:                                                     !
        #   ivflip          : control flag of vertical index direction          !
        #                     =0: index from toa to surface                     !
        #                     =1: index from surface to toa                     !
        #   lsashal         : control flag for shallow convection               !
        #   lcrick          : control flag for eliminating CRICK                !
        #                     =t: apply layer smoothing to eliminate CRICK      !
        #                     =f: do not apply layer smoothing                  !
        #   lcnorm          : control flag for in-cld condensate                !
        #                     =t: normalize cloud condensate                    !
        #                     =f: not normalize cloud condensate                !
        #                                                                       !
        #  ====================    end of description    =====================  !
        #

        cldcnv = np.zeros((IX, NLAY))
        cwp = np.zeros((IX, NLAY))
        cip = np.zeros((IX, NLAY))
        crp = np.zeros((IX, NLAY))
        csp = np.zeros((IX, NLAY))
        rew = np.zeros((IX, NLAY))
        rei = np.zeros((IX, NLAY))
        res = np.zeros((IX, NLAY))
        rer = np.zeros((IX, NLAY))
        tem2d = np.zeros((IX, NLAY))
        clwf = np.zeros((IX, NLAY))

        ptop1 = np.zeros((IX, self.NK_CLDS + 1))
        rxlat = np.zeros(IX)

        clouds = np.zeros((IX, NLAY, self.NF_CLDS))
        de_lgth = np.zeros(IX)

        for k in range(NLAY):
            for i in range(IX):
                rew[i, k] = self.reliq_def  # default liq radius to 10 micron
                rei[i, k] = self.reice_def  # default ice radius to 50 micron
                rer[i, k] = self.rrain_def  # default rain radius to 1000 micron
                res[i, k] = self.rsnow_def  # default snow radius to 250 micron
                tem2d[i, k] = min(1.0, max(0.0, (con_ttp - tlyr[i, k]) * 0.05))
                clwf[i, k] = 0.0

        if lcrick:
            for i in range(IX):
                clwf[i, 0] = 0.75 * clw[i, 0] + 0.25 * clw[i, 1]
                clwf[i, NLAY - 1] = 0.75 * clw[i, NLAY - 1] + 0.25 * clw[i, NLAY - 2]
            for k in range(1, NLAY - 1):
                for i in range(IX):
                    clwf[i, k] = (
                        0.25 * clw[i, k - 1] + 0.5 * clw[i, k] + 0.25 * clw[i, k + 1]
                    )
        else:
            for k in range(NLAY):
                for i in range(IX):
                    clwf[i, k] = clw[i, k]

        #  ---  find top pressure for each cloud domain for given latitude
        #       ptopc(k,i): top presure of each cld domain (k=1-4 are sfc,L,m,h;
        #  ---  i=1,2 are low-lat (<45 degree) and pole regions)

        for i in range(IX):
            rxlat[i] = abs(xlat[i] / con_pi)  # if xlat in pi/2 -> -pi/2 range

        for id in range(4):
            tem1 = self.ptopc[id, 1] - self.ptopc[id, 0]

            for i in range(IX):
                ptop1[i, id] = self.ptopc[id, 0] + tem1 * max(0.0, 4.0 * rxlat[i] - 1.0)

        #  ---  compute liquid/ice condensate path in g/m**2

        for k in range(NLAY):
            for i in range(IX):
                clwt = max(0.0, (clwf[i, k] + cnvw[i, k])) * self.gfac * delp[i, k]
                cip[i, k] = clwt * tem2d[i, k]
                cwp[i, k] = clwt - cip[i, k]

        #  ---  effective liquid cloud droplet radius over land

        for i in range(IX):
            if round(slmsk[i]) == 1:
                for k in range(NLAY):
                    rew[i, k] = 5.0 + 5.0 * tem2d[i, k]

        for k in range(NLAY):
            for i in range(IX):
                if cldtot[i, k] < self.climit:
                    cwp[i, k] = 0.0
                    cip[i, k] = 0.0
                    crp[i, k] = 0.0
                    csp[i, k] = 0.0

        if lcnorm:
            for k in range(NLAY):
                for i in range(IX):
                    if cldtot[i, k] >= self.climit:
                        tem1 = 1.0 / max(self.climit2, cldtot[i, k])
                        cwp[i, k] = cwp[i, k] * tem1
                        cip[i, k] = cip[i, k] * tem1
                        crp[i, k] = crp[i, k] * tem1
                        csp[i, k] = csp[i, k] * tem1

        # ---  effective ice cloud droplet radius

        for k in range(NLAY):
            for i in range(IX):
                tem2 = tlyr[i, k] - con_ttp

                if cip[i, k] > 0.0:
                    tem3 = (
                        self.gord * cip[i, k] * plyr[i, k] / (delp[i, k] * tvly[i, k])
                    )

                    if tem2 < -50.0:
                        rei[i, k] = (1250.0 / 9.917) * tem3 ** 0.109
                    elif tem2 < -40.0:
                        rei[i, k] = (1250.0 / 9.337) * tem3 ** 0.08
                    elif tem2 < -30.0:
                        rei[i, k] = (1250.0 / 9.208) * tem3 ** 0.055
                    else:
                        rei[i, k] = (1250.0 / 9.387) * tem3 ** 0.031

                    rei[i, k] = max(10.0, min(rei[i, k], 150.0))

        for k in range(NLAY):
            for i in range(IX):
                clouds[i, k, 0] = cldtot[i, k]
                clouds[i, k, 1] = cwp[i, k]
                clouds[i, k, 2] = rew[i, k]
                clouds[i, k, 3] = cip[i, k]
                clouds[i, k, 4] = rei[i, k]
                clouds[i, k, 6] = rer[i, k]
                clouds[i, k, 8] = res[i, k]

        #  --- ...  estimate clouds decorrelation length in km
        #           this is only a tentative test, need to consider change later

        if self.iovr == 3:
            for i in range(IX):
                de_lgth[i] = max(0.6, 2.78 - 4.6 * rxlat[i])

        #  ---  compute low, mid, high, total, and boundary layer cloud fractions
        #       and clouds top/bottom layer indices for low, mid, and high clouds.
        #       The three cloud domain boundaries are defined by ptopc.  The cloud
        #       overlapping method is defined by control flag 'iovr', which may
        #       be different for lw and sw radiation programs.

        clds, mtop, mbot = self.gethml(
            plyr, ptop1, cldtot, cldcnv, dz, de_lgth, IX, NLAY
        )

        return clouds, clds, mtop, mbot, de_lgth

    def progcld5(
        self,
        plyr,
        plvl,
        tlyr,
        qlyr,
        qstl,
        rhly,
        clw,
        xlat,
        xlon,
        slmsk,
        dz,
        delp,
        ntrac,
        ntcw,
        ntiw,
        ntrw,
        ntsw,
        ntgl,
        IX,
        NLAY,
        NLP1,
        uni_cld,
        lmfshal,
        lmfdeep2,
        cldcov,
        re_cloud,
        re_ice,
        re_snow,
        iovrsw,
        iovrlw,
    ):
        # =================   subprogram documentation block   ================ !
        #                                                                       !
        # subprogram:    progcld5    computes cloud related quantities using    !
        #   Thompson/WSM6 cloud microphysics scheme.                !
        #                                                                       !
        # abstract:  this program computes cloud fractions from cloud           !
        #   condensates,                                                        !
        #   and computes the low, mid, high, total and boundary layer cloud     !
        #   fractions and the vertical indices of low, mid, and high cloud      !
        #   top and base.  the three vertical cloud domains are set up in the   !
        #   initial subroutine "cld_init".                                      !
        #                                                                       !
        # usage:         call progcld5                                          !
        #                                                                       !
        # subprograms called:   gethml                                          !
        #                                                                       !
        # attributes:                                                           !
        #   language:   fortran 90                                              !
        #   machine:    ibm-sp, sgi                                             !
        #                                                                       !
        #                                                                       !
        #  ====================  definition of variables  ====================  !
        #                                                                       !
        # input variables:                                                      !
        #   plyr  (IX,NLAY) : model layer mean pressure in mb (100Pa)           !
        #   plvl  (IX,NLP1) : model level pressure in mb (100Pa)                !
        #   tlyr  (IX,NLAY) : model layer mean temperature in k                 !
        #   tvly  (IX,NLAY) : model layer virtual temperature in k              !
        #   qlyr  (IX,NLAY) : layer specific humidity in gm/gm                  !
        #   qstl  (IX,NLAY) : layer saturate humidity in gm/gm                  !
        #   rhly  (IX,NLAY) : layer relative humidity (=qlyr/qstl)              !
        #   clw   (IX,NLAY,ntrac) : layer cloud condensate amount               !
        #   xlat  (IX)      : grid latitude in radians, default to pi/2 -> -pi/2!
        #                     range, otherwise see in-line comment              !
        #   xlon  (IX)      : grid longitude in radians  (not used)             !
        #   slmsk (IX)      : sea/land mask array (sea:0,land:1,sea-ice:2)      !
        #   dz    (ix,nlay) : layer thickness (km)                              !
        #   delp  (ix,nlay) : model layer pressure thickness in mb (100Pa)      !
        #   IX              : horizontal dimention                              !
        #   NLAY,NLP1       : vertical layer/level dimensions                   !
        #   uni_cld         : logical - true for cloud fraction from shoc       !
        #   lmfshal         : logical - true for mass flux shallow convection   !
        #   lmfdeep2        : logical - true for mass flux deep convection      !
        #   cldcov          : layer cloud fraction (used when uni_cld=.true.    !
        #                                                                       !
        # output variables:                                                     !
        #   clouds(IX,NLAY,NF_CLDS) : cloud profiles                            !
        #      clouds(:,:,1) - layer total cloud fraction                       !
        #      clouds(:,:,2) - layer cloud liq water path         (g/m**2)      !
        #      clouds(:,:,3) - mean eff radius for liq cloud      (micron)      !
        #      clouds(:,:,4) - layer cloud ice water path         (g/m**2)      !
        #      clouds(:,:,5) - mean eff radius for ice cloud      (micron)      !
        #      clouds(:,:,6) - layer rain drop water path         not assigned  !
        #      clouds(:,:,7) - mean eff radius for rain drop      (micron)      !
        #  *** clouds(:,:,8) - layer snow flake water path        not assigned  !
        #      clouds(:,:,9) - mean eff radius for snow flake     (micron)      !
        #  *** fu's scheme need to be normalized by snow density (g/m**3/1.0e6) !
        #   clds  (IX,5)    : fraction of clouds for low, mid, hi, tot, bl      !
        #   mtop  (IX,3)    : vertical indices for low, mid, hi cloud tops      !
        #   mbot  (IX,3)    : vertical indices for low, mid, hi cloud bases     !
        #   de_lgth(ix)     : clouds decorrelation length (km)                  !
        #                                                                       !
        # module variables:                                                     !
        #   ivflip          : control flag of vertical index direction          !
        #                     =0: index from toa to surface                     !
        #                     =1: index from surface to toa                     !
        #   lmfshal         : mass-flux shallow conv scheme flag                !
        #   lmfdeep2        : scale-aware mass-flux deep conv scheme flag       !
        #   lcrick          : control flag for eliminating CRICK                !
        #                     =t: apply layer smoothing to eliminate CRICK      !
        #                     =f: do not apply layer smoothing                  !
        #   lcnorm          : control flag for in-cld condensate                !
        #                     =t: normalize cloud condensate                    !
        #                     =f: not normalize cloud condensate                !
        #                                                                       !
        #  ====================    end of description    =====================  !
        #

        cldtot = np.zeros((IX, NLAY))
        cldcnv = np.zeros((IX, NLAY))
        cwp = np.zeros((IX, NLAY))
        cip = np.zeros((IX, NLAY))
        crp = np.zeros((IX, NLAY))
        csp = np.zeros((IX, NLAY))
        rew = np.zeros((IX, NLAY))
        rei = np.zeros((IX, NLAY))
        res = np.zeros((IX, NLAY))
        rer = np.zeros((IX, NLAY))
        clwf = np.zeros((IX, NLAY))

        ptop1 = np.zerso((IX, self.NK_CLDS + 1))
        rxlat = np.zeros(IX)

        clouds = np.zeros((IX, NLAY, self.NF_CLDS))
        de_lgth = np.zeros(IX)

        for k in range(NLAY):
            for i in range(IX):
                rew[i, k] = re_cloud[i, k]
                rei[i, k] = re_ice[i, k]
                rer[i, k] = self.rrain_def  # default rain radius to 1000 micron
                res[i, k] = re_snow[i, k]

        for k in range(NLAY):
            for i in range(IX):
                clwf[i, k] = clw[i, k, ntcw] + clw[i, k, ntiw] + clw[i, k, ntsw]

        # -# Find top pressure for each cloud domain for given latitude.
        #    ptopc(k,i): top presure of each cld domain (k=1-4 are sfc,L,m,h;
        # ---  i=1,2 are low-lat (<45 degree) and pole regions)

        for i in range(IX):
            rxlat[i] = abs(xlat[i] / con_pi)  # if xlat in pi/2 -> -pi/2 range

        for id in range(4):
            tem1 = self.ptopc[id, 1] - self.ptopc[id, 0]

            for i in range(IX):
                ptop1[i, id] = self.ptopc[id, 0] + tem1 * max(0.0, 4.0 * rxlat[i] - 1.0)

        # -# Compute cloud liquid/ice condensate path in \f$ g/m^2 \f$ .

        for k in range(NLAY):
            for i in range(IX):
                cwp[i, k] = max(0.0, clw[i, k, ntcw] * self.gfac * delp[i, k])
                cip[i, k] = max(0.0, clw[i, k, ntiw] * self.gfac * delp[i, k])
                crp[i, k] = max(0.0, clw[i, k, ntrw] * self.gfac * delp[i, k])
                csp[i, k] = max(
                    0.0, (clw[i, k, ntsw] + clw[i, k, ntgl]) * self.gfac * delp[i, k]
                )

        if uni_cld:  # use unified sgs clouds generated outside
            for k in range(NLAY):
                for i in range(IX):
                    cldtot[i, k] = cldcov[i, k]
        else:
            # -# Calculate layer cloud fraction.
            clwmin = 0.0
            if not lmfshal:
                for k in range(NLAY):
                    for i in range(IX):
                        clwt = 1.0e-6 * (plyr[i, k] * 0.001)

                        if clwf[i, k] > clwt:
                            onemrh = max(1.0e-10, 1.0 - rhly[i, k])
                            clwm = clwmin / max(0.01, plyr[i, k] * 0.001)

                            tem1 = min(
                                max(np.sqrt(np.sqrt(onemrh * qstl[i, k])), 0.0001), 1.0
                            )
                            tem1 = 2000.0 / tem1

                            value = max(min(tem1 * (clwf[i, k] - clwm), 50.0), 0.0)
                            tem2 = np.sqrt(np.sqrt(rhly[i, k]))

                            cldtot[i, k] = max(tem2 * (1.0 - np.exp(-value)), 0.0)
            else:
                for k in range(NLAY):
                    for i in range(IX):
                        clwt = 1.0e-6 * (plyr[i, k] * 0.001)

                        if clwf[i, k] > clwt:
                            onemrh = max(1.0e-10, 1.0 - rhly[i, k])
                            clwm = clwmin / max(0.01, plyr[i, k] * 0.001)

                            tem1 = min(
                                max((onemrh * qstl[i, k]) ** 0.49, 0.0001), 1.0
                            )  # jhan
                            if lmfdeep2:
                                tem1 = self.xrc3 / tem1
                            else:
                                tem1 = 100.0 / tem1

                            value = max(min(tem1 * (clwf[i, k] - clwm), 50.0), 0.0)
                            tem2 = np.sqrt(np.sqrt(rhly[i, k]))

                            cldtot[i, k] = max(tem2 * (1.0 - np.exp(-value)), 0.0)

        for k in range(NLAY):
            for i in range(IX):
                if cldtot[i, k] < self.climit:
                    cldtot[i, k] = 0.0
                    cwp[i, k] = 0.0
                    cip[i, k] = 0.0
                    crp[i, k] = 0.0
                    csp[i, k] = 0.0

        if lcnorm:
            for k in range(NLAY):
                for i in range(IX):
                    if cldtot[i, k] >= self.climit:
                        tem1 = 1.0 / max(self.climit2, cldtot[i, k])
                        cwp[i, k] = cwp[i, k] * tem1
                        cip[i, k] = cip[i, k] * tem1
                        crp[i, k] = crp[i, k] * tem1
                        csp[i, k] = csp[i, k] * tem1

        for k in range(NLAY):
            for i in range(IX):
                clouds[i, k, 1] = cldtot[i, k]
                clouds[i, k, 2] = cwp[i, k]
                clouds[i, k, 3] = rew[i, k]
                clouds[i, k, 4] = cip[i, k]
                clouds[i, k, 5] = rei[i, k]
                clouds[i, k, 6] = crp[i, k]  # added for Thompson
                clouds[i, k, 7] = rer[i, k]
                clouds[i, k, 8] = csp[i, k]  # added for Thompson
                clouds[i, k, 9] = res[i, k]

        #  --- ...  estimate clouds decorrelation length in km
        #           this is only a tentative test, need to consider change later

        if self.iovr == 3:
            for i in range(IX):
                de_lgth[i] = max(0.6, 2.78 - 4.6 * rxlat[i])

        # -# Call gethml() to compute low,mid,high,total, and boundary layer
        #    cloud fractions and clouds top/bottom layer indices for low, mid,
        #    and high clouds.
        # ---  compute low, mid, high, total, and boundary layer cloud fractions
        #      and clouds top/bottom layer indices for low, mid, and high clouds.
        #      The three cloud domain boundaries are defined by ptopc.  The cloud
        #      overlapping method is defined by control flag 'iovr', which may
        #      be different for lw and sw radiation programs.

        clds, mtop, mbot = self.gethml(
            plyr, ptop1, cldtot, cldcnv, dz, de_lgth, IX, NLAY
        )

        return clouds, clds, mtop, mbot, de_lgth

    def progclduni(
        self,
        plyr,
        plvl,
        tlyr,
        tvly,
        ccnd,
        ncnd,
        xlat,
        xlon,
        slmsk,
        dz,
        delp,
        IX,
        NLAY,
        NLP1,
        cldtot,
        effrl,
        effri,
        effrr,
        effrs,
        effr_in,
        iovrsw,
        iovrlw,
    ):
        # =================   subprogram documentation block   ================ !
        #                                                                       !
        # subprogram:    progclduni    computes cloud related quantities using    !
        #   for unified cloud microphysics scheme.                !
        #                                                                       !
        # abstract:  this program computes cloud fractions from cloud           !
        #   condensates, calculates liquid/ice cloud droplet effective radius,  !
        #   and computes the low, mid, high, total and boundary layer cloud     !
        #   fractions and the vertical indices of low, mid, and high cloud      !
        #   top and base.  the three vertical cloud domains are set up in the   !
        #   initial subroutine "cld_init".                                      !
        #                                                                       !
        # usage:         call progclduni                                          !
        #                                                                       !
        # subprograms called:   gethml                                          !
        #                                                                       !
        # attributes:                                                           !
        #   language:   fortran 90                                              !
        #   machine:    ibm-sp, sgi                                             !
        #                                                                       !
        #                                                                       !
        #  ====================  definition of variables  ====================  !
        #                                                                       !
        # input variables:                                                      !
        #   plyr  (IX,NLAY)      : model layer mean pressure in mb (100Pa)           !
        #   plvl  (IX,NLP1)      : model level pressure in mb (100Pa)                !
        #   tlyr  (IX,NLAY)      : model layer mean temperature in k                 !
        #   tvly  (IX,NLAY)      : model layer virtual temperature in k              !
        #   ccnd  (IX,NLAY,ncnd) : layer cloud condensate amount                     !
        #                          water, ice, rain, snow (+ graupel)                !
        #   ncnd                 : number of layer cloud condensate types (max of 4) !
        #   xlat  (IX)           : grid latitude in radians, default to pi/2 -> -pi/2!
        #                          range, otherwise see in-line comment              !
        #   xlon  (IX)           : grid longitude in radians  (not used)             !
        #   slmsk (IX)           : sea/land mask array (sea:0,land:1,sea-ice:2)      !
        #   IX                   : horizontal dimention                              !
        #   NLAY,NLP1            : vertical layer/level dimensions                   !
        #   cldtot               : unified cloud fracrion from moist physics         !
        #   effrl (ix,nlay)      : effective radius for liquid water                 !
        #   effri (ix,nlay)      : effective radius for ice water                    !
        #   effrr (ix,nlay)      : effective radius for rain water                   !
        #   effrs (ix,nlay)      : effective radius for snow water                   !
        #   effr_in              : logical - if .true. use input effective radii     !
        #   dz    (ix,nlay)      : layer thickness (km)                              !
        #   delp  (ix,nlay)      : model layer pressure thickness in mb (100Pa)      !
        #                                                                       !
        # output variables:                                                     !
        #   clouds(IX,NLAY,NF_CLDS) : cloud profiles                            !
        #      clouds(:,:,1) - layer total cloud fraction                       !
        #      clouds(:,:,2) - layer cloud liq water path         (g/m**2)      !
        #      clouds(:,:,3) - mean eff radius for liq cloud      (micron)      !
        #      clouds(:,:,4) - layer cloud ice water path         (g/m**2)      !
        #      clouds(:,:,5) - mean eff radius for ice cloud      (micron)      !
        #      clouds(:,:,6) - layer rain drop water path         not assigned  !
        #      clouds(:,:,7) - mean eff radius for rain drop      (micron)      !
        #  *** clouds(:,:,8) - layer snow flake water path        not assigned  !
        #      clouds(:,:,9) - mean eff radius for snow flake     (micron)      !
        #  *** fu's scheme need to be normalized by snow density (g/m**3/1.0e6) !
        #   clds  (IX,5)    : fraction of clouds for low, mid, hi, tot, bl      !
        #   mtop  (IX,3)    : vertical indices for low, mid, hi cloud tops      !
        #   mbot  (IX,3)    : vertical indices for low, mid, hi cloud bases     !
        #   de_lgth(ix)     : clouds decorrelation length (km)                  !
        #                                                                       !
        # module variables:                                                     !
        #   ivflip          : control flag of vertical index direction          !
        #                     =0: index from toa to surface                     !
        #                     =1: index from surface to toa                     !
        #   lmfshal         : mass-flux shallow conv scheme flag                !
        #   lmfdeep2        : scale-aware mass-flux deep conv scheme flag       !
        #   lcrick          : control flag for eliminating CRICK                !
        #                     =t: apply layer smoothing to eliminate CRICK      !
        #                     =f: do not apply layer smoothing                  !
        #   lcnorm          : control flag for in-cld condensate                !
        #                     =t: normalize cloud condensate                    !
        #                     =f: not normalize cloud condensate                !
        #                                                                       !
        #  ====================    end of description    =====================  !
        #

        cldcnv = np.zeros((IX, NLAY))
        cwp = np.zeros((IX, NLAY))
        cip = np.zeros((IX, NLAY))
        crp = np.zeros((IX, NLAY))
        csp = np.zeros((IX, NLAY))
        rew = np.zeros((IX, NLAY))
        rei = np.zeros((IX, NLAY))
        res = np.zeros((IX, NLAY))
        rer = np.zeros((IX, NLAY))

        cndf = np.zeros((IX, NLAY, ncnd))
        rxlat = np.zeros((IX, self.NK_CLDS + 1))
        ptop1 = np.zerso((IX, self.NK_CLDS + 1))

        clouds = np.zeros((IX, NLAY, self.NF_CLDS))
        de_lgth = np.zeros(IX)

        for n in range(ncnd):
            for k in range(NLAY):
                for i in range(IX):
                    cndf[i, k, n] = ccnd[i, k, n]

        if lcrick:
            for n in range(ncnd):
                for i in range(IX):
                    cndf[i, 0, n] = 0.75 * ccnd[i, 0, n] + 0.25 * ccnd[i, 1, n]
                    cndf[i, NLAY, n] = (
                        0.75 * ccnd[i, NLAY, n] + 0.25 * ccnd[i, NLAY - 1, n]
                    )
            for k in range(1, NLAY - 1):
                for i in range(IX):
                    cndf[i, k, n] = (
                        0.25 * (ccnd[i, k - 1, n] + ccnd[i, k + 1, n])
                        + 0.5 * ccnd[i, k, n]
                    )

        # -# Compute cloud liquid/ice condensate path in \f$ g/m^2 \f$ .

        if ncnd == 2:
            for k in range(NLAY):
                for i in range(IX):
                    tem1 = self.gfac * delp[i, k]
                    cwp[i, k] = cndf[i, k, 0] * tem1
                    cip[i, k] = cndf[i, k, 1] * tem1
        elif ncnd == 4:
            for k in range(NLAY):
                for i in range(IX):
                    tem1 = self.gfac * delp[i, k]
                    cwp[i, k] = cndf[i, k, 0] * tem1
                    cip[i, k] = cndf[i, k, 1] * tem1
                    crp[i, k] = cndf[i, k, 2] * tem1
                    csp[i, k] = cndf[i, k, 3] * tem1

        for k in range(NLAY):
            for i in range(IX):
                if cldtot[i, k] < self.climit:
                    cwp[i, k] = 0.0
                    cip[i, k] = 0.0
                    crp[i, k] = 0.0
                    csp[i, k] = 0.0

        if lcnorm:
            for k in range(NLAY):
                for i in range(IX):
                    if cldtot[i, k] >= self.climit:
                        tem1 = 1.0 / max(self.climit2, cldtot[i, k])
                        cwp[i, k] = cwp[i, k] * tem1
                        cip[i, k] = cip[i, k] * tem1
                        crp[i, k] = crp[i, k] * tem1
                        csp[i, k] = csp[i, k] * tem1

        #     assign/calculate efective radii for cloud water, ice, rain, snow

        if effr_in:
            for k in range(NLAY):
                for i in range(IX):
                    rew[i, k] = effrl[i, k]
                    rei[i, k] = max(10.0, min(150.0, effri[i, k]))
                    rer[i, k] = effrr[i, k]
                    res[i, k] = effrs[i, k]
        else:
            for k in range(NLAY):
                for i in range(IX):
                    rew[i, k] = self.reliq_def  # default liq  radius to 10   micron
                    rei[i, k] = self.reice_def  # default ice  radius to 50   micron
                    rer[i, k] = self.rrain_def  # default rain radius to 1000 micron
                    res[i, k] = self.rsnow_def  # default snow radius to 250  micron

            # -# Compute effective liquid cloud droplet radius over land.
            for i in range(IX):
                if round(slmsk[i]) == 1:
                    for k in range(NLAY):
                        tem1 = min(1.0, max(0.0, (con_ttp - tlyr[i, k]) * 0.05))
                        rew[i, k] = 5.0 + 5.0 * tem1

            # -# Compute effective ice cloud droplet radius following Heymsfield
            #    and McFarquhar (1996) \cite heymsfield_and_mcfarquhar_1996.

            for k in range(NLAY):
                for i in range(IX):
                    tem2 = tlyr[i, k] - con_ttp

                    if cip[i, k] > 0.0:
                        tem3 = (
                            self.gord
                            * cip[i, k]
                            * plyr[i, k]
                            / (delp[i, k] * tvly[i, k])
                        )

                    if tem2 < -50.0:
                        rei[i, k] = (1250.0 / 9.917) * tem3 ** 0.109
                    elif tem2 < -40.0:
                        rei[i, k] = (1250.0 / 9.337) * tem3 ** 0.08
                    elif tem2 < -30.0:
                        rei[i, k] = (1250.0 / 9.208) * tem3 ** 0.055
                    else:
                        rei[i, k] = (1250.0 / 9.387) * tem3 ** 0.031

                    rei[i, k] = max(10.0, min(rei[i, k], 150.0))

        for k in range(NLAY):
            for i in range(IX):
                clouds[i, k, 0] = cldtot[i, k]
                clouds[i, k, 1] = cwp[i, k]
                clouds[i, k, 2] = rew[i, k]
                clouds[i, k, 3] = cip[i, k]
                clouds[i, k, 4] = rei[i, k]
                clouds[i, k, 5] = crp[i, k]
                clouds[i, k, 6] = rer[i, k]
                clouds[i, k, 7] = csp[i, k]
                clouds[i, k, 8] = res[i, k]

        # -# Find top pressure for each cloud domain for given latitude.
        #    ptopc(k,i): top presure of each cld domain (k=1-4 are sfc,L,m,h;
        # ---  i=1,2 are low-lat (<45 degree) and pole regions)

        for i in range(IX):
            rxlat[i] = abs(xlat[i] / con_pi)  # if xlat in pi/2 -> -pi/2 range

        for id in range(4):
            tem1 = self.ptopc[id, 1] - self.ptopc[id, 0]
            for i in range(IX):
                ptop1[i, id] = self.ptopc[id, 0] + tem1 * max(0.0, 4.0 * rxlat[i] - 1.0)

        #  --- ...  estimate clouds decorrelation length in km
        #           this is only a tentative test, need to consider change later

        if self.iovr == 3:
            for i in range(IX):
                de_lgth[i] = max(0.6, 2.78 - 4.6 * rxlat[i])

        # -# Call gethml() to compute low,mid,high,total, and boundary layer
        #    cloud fractions and clouds top/bottom layer indices for low, mid,
        #    and high clouds.
        # ---  compute low, mid, high, total, and boundary layer cloud fractions
        #      and clouds top/bottom layer indices for low, mid, and high clouds.
        #      The three cloud domain boundaries are defined by ptopc.  The cloud
        #      overlapping method is defined by control flag 'iovr', which may
        #      be different for lw and sw radiation programs.

        clds, mtop, mbot = self.gethml(
            plyr, ptop1, cldtot, cldcnv, dz, de_lgth, IX, NLAY
        )

        return clouds, clds, mtop, mbot, de_lgth

    def gethml(self, plyr, ptop1, cldtot, cldcnv, dz, de_lgth, IX, NLAY):
        #  ===================================================================  !
        #                                                                       !
        # abstract: compute high, mid, low, total, and boundary cloud fractions !
        #   and cloud top/bottom layer indices for model diagnostic output.     !
        #   the three cloud domain boundaries are defined by ptopc.  the cloud  !
        #   overlapping method is defined by control flag 'iovr', which is also !
        #   used by lw and sw radiation programs.                               !
        #                                                                       !
        # usage:         call gethml                                            !
        #                                                                       !
        # subprograms called:  none                                             !
        #                                                                       !
        # attributes:                                                           !
        #   language:   fortran 90                                              !
        #   machine:    ibm-sp, sgi                                             !
        #                                                                       !
        #                                                                       !
        #  ====================  definition of variables  ====================  !
        #                                                                       !
        # input variables:                                                      !
        #   plyr  (IX,NLAY) : model layer mean pressure in mb (100Pa)           !
        #   ptop1 (IX,4)    : pressure limits of cloud domain interfaces        !
        #                     (sfc,low,mid,high) in mb (100Pa)                  !
        #   cldtot(IX,NLAY) : total or straiform cloud profile in fraction      !
        #   cldcnv(IX,NLAY) : convective cloud (for diagnostic scheme only)     !
        #   dz    (ix,nlay) : layer thickness (km)                              !
        #   de_lgth(ix)     : clouds vertical de-correlation length (km)        !
        #   IX              : horizontal dimention                              !
        #   NLAY            : vertical layer dimensions                         !
        #                                                                       !
        # output variables:                                                     !
        #   clds  (IX,5)    : fraction of clouds for low, mid, hi, tot, bl      !
        #   mtop  (IX,3)    : vertical indices for low, mid, hi cloud tops      !
        #   mbot  (IX,3)    : vertical indices for low, mid, hi cloud bases     !
        #                                                                       !
        # external module variables:  (in physparam)                            !
        #   ivflip          : control flag of vertical index direction          !
        #                     =0: index from toa to surface                     !
        #                     =1: index from surface to toa                     !
        #                                                                       !
        # internal module variables:                                            !
        #   iovr            : control flag for cloud overlap                    !
        #                     =0 random overlapping clouds                      !
        #                     =1 max/ran overlapping clouds                     !
        #                     =2 maximum overlapping  ( for mcica only )        !
        #                     =3 decorr-length ovlp   ( for mcica only )        !
        #                                                                       !
        #  ====================    end of description    =====================  !
        #

        cl1 = np.ones(IX)
        cl2 = np.ones(IX)

        dz1 = np.zeros(IX)

        idom = np.zeros(IX, dtype=np.int32)
        kbt1 = np.zeros(IX)
        kth1 = np.zeros(IX)
        kbt2 = np.zeros(IX)
        kth2 = np.zeros(IX)

        clds = np.zeros((IX, 5))
        mtop = np.zeros((IX, 3))
        mbot = np.zeros((IX, 3))

        #  ---  total and bl clouds, where cl1, cl2 are fractions of clear-sky view
        #       layer processed from surface and up

        # Calculate total and BL cloud fractions (maximum-random cloud
        # overlapping is operational).

        if self.ivflip == 0:  # input data from toa to sfc
            kstr = NLAY
            kend = 1
            kinc = -1
        else:  # input data from sfc to toa
            kstr = 1
            kend = NLAY
            kinc = 1

        if self.iovr == 0:  # random overlap

            for k in range(kstr - 1, kend, kinc):
                for i in range(IX):
                    ccur = min(self.ovcst, max(cldtot[i, k], cldcnv[i, k]))
                    if ccur >= self.climit:
                        cl1[i] = cl1[i] * (1.0 - ccur)

                if k == self.llyr - 1:
                    for i in range(IX):
                        clds[i, 4] = 1.0 - cl1[i]  # save bl cloud

            for i in range(IX):
                clds[i, 3] = 1.0 - cl1[i]  # save total cloud

        elif self.iovr == 1:  # max/ran overlap

            for k in range(kstr - 1, kend, kinc):
                for i in range(IX):
                    ccur = min(self.ovcst, max(cldtot[i, k], cldcnv[i, k]))
                    if ccur >= self.climit:  # cloudy layer
                        cl2[i] = min(cl2[i], (1.0 - ccur))
                    else:  # clear layer
                        cl1[i] = cl1[i] * cl2[i]
                        cl2[i] = 1.0

                if k == self.llyr - 1:
                    for i in range(IX):
                        clds[i, 4] = 1.0 - cl1[i] * cl2[i]  # save bl cloud

            for i in range(IX):
                clds[i, 3] = 1.0 - cl1[i] * cl2[i]  # save total cloud

        elif self.iovr == 2:  # maximum overlap all levels

            cl1[:] = 0.0

            for k in range(kstr - 1, kend, kinc):
                for i in range(IX):
                    ccur = min(self.ovcst, max(cldtot[i, k], cldcnv[i, k]))
                    if ccur >= self.climit:
                        cl1[i] = max(cl1[i], ccur)

                if k == self.llyr - 1:
                    for i in range(IX):
                        clds[i, 4] = cl1[i]  # save bl cloud

            for i in range(IX):
                clds[i, 3] = cl1[i]  # save total cloud

        elif self.iovr == 3:  # random if clear-layer divided,
            # otherwise de-corrlength method
            for i in range(IX):
                dz1[i] = -dz[i, kstr]

            for k in range(kstr - 1, kend, kinc):
                for i in range(IX):
                    ccur = min(self.ovcst, max(cldtot[i, k], cldcnv[i, k]))
                    if ccur >= self.climit:  # cloudy layer
                        alfa = np.exp(-0.5 * (dz1[i] + dz[i, k]) / de_lgth[i])
                        dz1[i] = dz[i, k]
                        cl2[i] = alfa * min(cl2[i], (1.0 - ccur)) + (1.0 - alfa) * (
                            cl2(i) * (1.0 - ccur)
                        )  # random part
                    else:  # clear layer
                        cl1[i] = cl1[i] * cl2[i]
                        cl2[i] = 1.0
                        if k != kend - 1:
                            dz1[i] = -dz[i, k + kinc]

                if k == self.llyr - 1:
                    for i in range(IX):
                        clds[i, 4] = 1.0 - cl1[i] * cl2[i]  # save bl cloud

            for i in range(IX):
                clds[i, 3] = 1.0 - cl1[i] * cl2[i]  # save total cloud

        #  ---  high, mid, low clouds, where cl1, cl2 are cloud fractions
        #       layer processed from one layer below llyr and up
        #  ---  change! layer processed from surface to top, so low clouds will
        #       contains both bl and low clouds.

        # Calculte high, mid, low cloud fractions and vertical indices of
        #    cloud tops/bases.
        if self.ivflip == 0:  # input data from toa to sfc
            for i in range(IX):
                cl1[i] = 0.0
                cl2[i] = 0.0
                kbt1[i] = NLAY
                kbt2[i] = NLAY
                kth1[i] = 0
                kth2[i] = 0
                idom[i] = 1
                mbot[i, 0] = NLAY
                mtop[i, 0] = NLAY
                mbot[i, 1] = NLAY - 1
                mtop[i, 1] = NLAY - 1
                mbot[i, 2] = NLAY - 1
                mtop[i, 2] = NLAY - 1

            for k in range(NLAY, -1, -1):
                for i in range(IX):
                    id = idom[i] - 1
                    id1 = id + 1

                    pcur = plyr[i, k]
                    ccur = min(self.ovcst, max(cldtot[i, k], cldcnv[i, k]))

                    if k > 1:
                        pnxt = plyr[i, k - 1]
                        cnxt = min(self.ovcst, max(cldtot[i, k - 1], cldcnv[i, k - 1]))
                    else:
                        pnxt = -1.0
                        cnxt = 0.0

                    if pcur < ptop1[i, id1]:
                        id = id + 1
                        id1 = id1 + 1
                        idom[i] = id

                    if ccur >= self.climit:
                        if kth2[i] == 0:
                            kbt2[i] = k

                        kth2[i] = kth2[i] + 1

                        if self.iovr == 0:
                            cl2[i] = cl2[i] + ccur - cl2[i] * ccur
                        else:
                            cl2[i] = max(cl2[i], ccur)

                        if cnxt < self.climit or pnxt < ptop1[i, id1]:
                            kbt1[i] = round(
                                (cl1[i] * kbt1[i] + cl2[i] * kbt2[i])
                                / (cl1[i] + cl2[i])
                            )
                            kth1[i] = round(
                                (cl1[i] * kth1[i] + cl2[i] * kth2[i])
                                / (cl1[i] + cl2[i])
                            )
                            cl1[i] = cl1[i] + cl2[i] - cl1[i] * cl2[i]

                            kbt2[i] = k
                            kth2[i] = 0
                            cl2[i] = 0.0

                    if pnxt < ptop1[i, id1]:
                        clds[i, id] = cl1[i]
                        mtop[i, id] = min(kbt1[i], kbt1[i] - kth1[i] + 1)
                        mbot[i, id] = kbt1[i]

                        cl1[i] = 0.0
                        kbt1[i] = k
                        kth1[i] = 0

                        if id1 <= self.NK_CLDS:
                            mbot[i, id1] = kbt1[i]
                            mtop[i, id1] = kbt1[i]

        else:  # input data from sfc to toa

            for i in range(IX):
                cl1[i] = 0.0
                cl2[i] = 0.0
                kbt1[i] = 1
                kbt2[i] = 1
                kth1[i] = 0
                kth2[i] = 0
                idom[i] = 1
                mbot[i, 0] = 1
                mtop[i, 0] = 1
                mbot[i, 1] = 2
                mtop[i, 1] = 2
                mbot[i, 2] = 2
                mtop[i, 2] = 2

            for k in range(NLAY):
                for i in range(IX):
                    id = idom[i]
                    id1 = id + 1

                    pcur = plyr[i, k]
                    ccur = min(self.ovcst, max(cldtot[i, k], cldcnv[i, k]))

                    if k < NLAY - 1:
                        pnxt = plyr[i, k + 1]
                        cnxt = min(self.ovcst, max(cldtot[i, k + 1], cldcnv[i, k + 1]))
                    else:
                        pnxt = -1.0
                        cnxt = 0.0

                    if pcur < ptop1[i, id1 - 1]:
                        id += 1
                        id1 += 1
                        idom[i] = id

                    if ccur >= self.climit:
                        if kth2[i] == 0:
                            kbt2[i] = k + 1
                        kth2[i] = kth2[i] + 1

                        if self.iovr == 0:
                            cl2[i] = cl2[i] + ccur - cl2[i] * ccur
                        else:
                            cl2[i] = max(cl2[i], ccur)

                        if cnxt < self.climit or pnxt < ptop1[i, id1 - 1]:
                            kbt1[i] = round(
                                (cl1[i] * kbt1[i] + cl2[i] * kbt2[i])
                                / (cl1[i] + cl2[i])
                            )
                            kth1[i] = round(
                                (cl1[i] * kth1[i] + cl2[i] * kth2[i])
                                / (cl1[i] + cl2[i])
                            )
                            cl1[i] = cl1[i] + cl2[i] - cl1[i] * cl2[i]

                            kbt2[i] = k + 2
                            kth2[i] = 0
                            cl2[i] = 0.0

                    if pnxt < ptop1[i, id1 - 1]:
                        clds[i, id - 1] = cl1[i]
                        mtop[i, id - 1] = max(kbt1[i], kbt1[i] + kth1[i] - 1)
                        mbot[i, id - 1] = kbt1[i]

                        cl1[i] = 0.0
                        kbt1[i] = min(k + 2, NLAY)
                        kth1[i] = 0

                        if id1 <= self.NK_CLDS:
                            mbot[i, id1 - 1] = kbt1[i]
                            mtop[i, id1 - 1] = kbt1[i]

        return clds, mtop, mbot
