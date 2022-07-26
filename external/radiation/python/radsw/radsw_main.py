import numpy as np
import xarray as xr
import os
import sys
import warnings
from numba import jit
import radsw.radsw_bands as bands
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

ngs = np.array(ngs)
ng = np.array(ng)
nspa = np.array(nspa)
nspb = np.array(nspb)

@jit(nopython=True, cache=True)
def vrtqdr(zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1):

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


@jit(nopython=True, cache=True)
def spcvrtm(
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
        idxsfc,
        ftiny,
        eps,
        nuvb,
        exp_tbl,
        bpade,
        flimit,
        oneminus,
        NGB,
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
            ibd = idxsfc[jb - 15] - 1  # spectral band index

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

                ztau0 = max(ftiny, taur[k, jg] + taug[k, jg] + tauae[k, ib])
                zssa0 = taur[k, jg] + tauae[k, ib] * ssaae[k, ib]
                zasy0 = asyae[k, ib] * ssaae[k, ib] * tauae[k, ib]
                zssaw = min(oneminus, zssa0 / ztau0)
                zasyw = zasy0 / max(ftiny, zssa0)

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
                        ftind = zb1 / (bpade + zb1)
                        itind = int(ftind * ntbmx + 0.5)
                        zb2 = exp_tbl[itind]

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
                        max(flimit, abs(zrpp1)), zrpp1
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
                        ftind = zb1 / (bpade + zb1)
                        itind = int(ftind * ntbmx + 0.5)
                        zexm1 = exp_tbl[itind]

                    zexp1 = 1.0 / zexm1

                    zb2 = min(sntz * ztau1, 500.0)
                    if zb2 <= od_lo:
                        zexm2 = 1.0 - zb2 + 0.5 * zb2 * zb2
                    else:
                        ftind = zb2 / (bpade + zb2)
                        itind = int(ftind * ntbmx + 0.5)
                        zexm2 = exp_tbl[itind]

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
                    ftind = zr1 / (bpade + zr1)
                    itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                    zexp3 = exp_tbl[itind]

                ztdbt[k] = zexp3 * ztdbt[kp]
                zldbt[kp] = zexp3

                #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                #           (must use 'orig', unscaled cloud optical depth)

                zr1 = ztau0 * sntz
                if zr1 <= od_lo:
                    zexp4 = 1.0 - zr1 + 0.5 * zr1 * zr1
                else:
                    ftind = zr1 / (bpade + zr1)
                    itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                    zexp4 = exp_tbl[itind]

                zldbt0[k] = zexp4
                ztdbt0 = zexp4 * ztdbt0

            zfu, zfd = vrtqdr(zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1)

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

            if cf1 > eps:

                #  --- ...  set up toa direct beam and surface values (beam and diff)
                ztdbt0 = 1.0
                zldbt[0] = 0.0

                for k in range(nlay - 1, -1, -1):
                    kp = k + 1
                    if cldfmc[k, jg] > ftiny:  # it is a cloudy-layer

                        ztau0 = ztaus[k] + taucw[k, ib]
                        zssa0 = zssas[k] + ssacw[k, ib]
                        zasy0 = zasys[k] + asycw[k, ib]
                        zssaw = min(oneminus, zssa0 / ztau0)
                        zasyw = zasy0 / max(ftiny, zssa0)

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
                                ftind = zb1 / (bpade + zb1)
                                itind = int(ftind * ntbmx + 0.5)
                                zb2 = exp_tbl[itind]

                            #      ...  collimated beam
                            zrefb[kp] = max(
                                0.0, min(1.0, (za2 - za1 * (1.0 - zb2)) / (1.0 + za2))
                            )
                            ztrab[kp] = max(0.0, min(1.0, 1.0 - zrefb[kp]))

                            #      ...  isotropic incidence
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
                                max(flimit, abs(zrpp1)), zrpp1
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
                                ftind = zb1 / (bpade + zb1)
                                itind = int(ftind * ntbmx + 0.5)
                                zexm1 = exp_tbl[itind]

                            zexp1 = 1.0 / zexm1

                            zb2 = min(ztau1 * sntz, 500.0)
                            if zb2 <= od_lo:
                                zexm2 = 1.0 - zb2 + 0.5 * zb2 * zb2
                            else:
                                ftind = zb2 / (bpade + zb2)
                                itind = int(ftind * ntbmx + 0.5)
                                zexm2 = exp_tbl[itind]

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
                            ftind = zr1 / (bpade + zr1)
                            itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                            zexp3 = exp_tbl[itind]

                        zldbt[kp] = zexp3
                        ztdbt[k] = zexp3 * ztdbt[kp]

                        #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                        #           (must use 'orig', unscaled cloud optical depth)

                        zr1 = ztau0 * sntz
                        if zr1 <= od_lo:
                            zexp4 = 1.0 - zr1 + 0.5 * zr1 * zr1
                        else:
                            ftind = zr1 / (bpade + zr1)
                            itind = int(max(0, min(ntbmx, int(0.5 + ntbmx * ftind))))
                            zexp4 = exp_tbl[itind]

                        ztdbt0 = zexp4 * ztdbt0

                    else:  # if_cldfmc_block  ---  it is a clear layer

                        #  --- ...  direct beam transmittance
                        ztdbt[k] = zldbt[kp] * ztdbt[kp]

                        #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                        ztdbt0 = zldbt0[k] * ztdbt0

                #  --- ...  perform vertical quadrature

                zfu, zfd = vrtqdr(
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
        ibd = nuvb - nblow
        suvbf0 = fxdn0[0, ibd]

        if cf1 <= eps:  # clear column, set total-sky=clear-sky fluxes
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

@jit(nopython=True, cache=True)
def mcica_subcol(iovrsw, cldf, nlay, ipseed, dz, de_lgth, ipt,rand2d):
        rand2d = rand2d[ipt, :] 

        #  ---  outputs:
        lcloudy = np.zeros((nlay, ngptsw))

        #  ---  locals:
        cdfunc = np.zeros((nlay, ngptsw))
        #  --- ...  sub-column set up according to overlapping assumption

        if iovrsw == 1:  # max-ran overlap

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
    
@jit(nopython=True, cache=True)
def cldprop(
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
        rand2d,
        extliq1, 
        extliq2, 
        ssaliq1, 
        ssaliq2, 
        asyliq1, 
        asyliq2, 
        extice2, 
        ssaice2, 
        asyice2, 
        extice3, 
        ssaice3, 
        asyice3, 
        abari,  
        bbari, 
        cbari, 
        dbari,  
        ebari,  
        fbari, 
        b0s, 
        b1s, 
        b0r, 
        c0s,  
        c0r,  
        a0r, 
        a1r, 
        a0s, 
        a1s, 
        ftiny,
        idxebc,
        isubcsw,
        iovrsw,
    ):
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

        lcloudy = np.zeros((nlay, ngptsw), dtype=np.bool_)

        # Compute cloud radiative properties for a cloudy column.

        if iswcliq > 0:

            for k in range(nlay):
                if cfrac[k] > ftiny:

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
                        tauice[:] = 0.0
                        ssaice[:] = 0.0
                        asyice[:] = 0.0
                    else:

                        #  --- ...  ebert and curry approach for all particle sizes though somewhat
                        #           unjustified for large ice particles

                        if iswcice == 1:
                            refice = min(130.0, max(13.0, refice))

                            for ib in range(nbandssw):
                                ia = (
                                    idxebc[ib] - 1
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
                if cfrac[k] > ftiny:
                    for ib in range(nbdsw):
                        taucw[k, ib] = cdat1[k]
                        ssacw[k, ib] = cdat1[k] * cdat2[k]
                        asycw[k, ib] = ssacw[k, ib] * cdat3[k]

        # -# if physparam::isubcsw > 0, call mcica_subcol() to distribute
        #    cloud properties to each g-point.

        if isubcsw > 0:  # mcica sub-col clouds approx
            cldf = cfrac
            cldf = np.where(cldf < ftiny, 0.0, cldf)

            #  --- ...  call sub-column cloud generator

            lcloudy = mcica_subcol(iovrsw,cldf, nlay, ipseed, dz, delgth, ipt, rand2d)

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

@jit(nopython=True, cache=True)
def taumol(
        nspa,
        nspb,
        ng,
        ngs,
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
        strrat,
        specwt,
        layreffr,
        ix1,
        ix2 ,
        ibx,
        sfluxref01,
        sfluxref02,
        sfluxref03,
        scalekur,
        selfref_16,
        forref_16,
        absa_16,
        absb_16,
        rayl_16,
        selfref_17,
        forref_17,
        absa_17,
        absb_17,
        rayl_17,
        selfref_18,
        forref_18,
        absa_18,
        absb_18,
        rayl_18,
        selfref_19,
        forref_19,
        absa_19,
        absb_19,
        rayl_19,
        selfref_20,
        forref_20,
        absa_20,
        absb_20,
        absch4_20,
        rayl_20,
        selfref_21,
        forref_21,
        absa_21,
        absb_21,
        rayl_21,
        selfref_22,
        forref_22,
        absa_22,
        absb_22,
        rayl_22,
        selfref_23,
        forref_23,
        absa_23,
        rayl_23,
        givfac_23,
        selfref_24,
        forref_24,
        absa_24,
        absb_24,
        abso3a_24,
        abso3b_24,
        rayla_24,
        raylb_24,
        absa_25,
        abso3a_25,
        abso3b_25,
        rayl_25,
        rayl_26,
        absa_27,
        absb_27,
        rayl_27,
        absa_28,
        absb_28,
        rayl_28,
        forref_29,
        absa_29,
        absb_29,
        selfref_29,
        absh2o_29,
        absco2_29,
        rayl_29,
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
                    speccomb = colm1 + strrat[b] * colm2
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
                    speccomb = colm1 + strrat[b] * colm2
                    specmult = specwt[b] * min(oneminus, colm1 / speccomb)
                    js = 1 + int(specmult) - 1
                    fs = np.mod(specmult, 1.0)

                    for j in range(njb):
                        sfluxzen[ns + j] = sfluxref03[j, js, ibd] + fs * (
                            sfluxref03[j, js + 1, ibd] - sfluxref03[j, js, ibd]
                        )
        taug, taur = bands.taumol16(
            strrat,
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
            selfref_16,
            forref_16,
            absa_16,
            absb_16,
            rayl_16,
        )
        taug, taur = bands.taumol17(
            strrat,
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
            selfref_17,
            forref_17,
            absa_17,
            absb_17,
            rayl_17,
        )
        taug, taur = bands.taumol18(
            strrat,
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
            selfref_18,
            forref_18,
            absa_18,
            absb_18,
            rayl_18,
        )
        taug, taur = bands.taumol19(
            strrat,
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
            selfref_19,
            forref_19,
            absa_19,
            absb_19,
            rayl_19,
        )
        taug, taur = bands.taumol20(
            strrat,
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
            selfref_20,
            forref_20,
            absa_20,
            absb_20,
            absch4_20,
            rayl_20,
        )
        taug, taur = bands.taumol21(
            strrat,
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
            selfref_21,
            forref_21,
            absa_21,
            absb_21,
            rayl_21,
        )
        taug, taur = bands.taumol22(
            strrat,
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
            selfref_22,
            forref_22,
            absa_22,
            absb_22,
            rayl_22,
        )
        taug, taur = bands.taumol23(
            strrat,
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
            selfref_23,
            forref_23,
            absa_23,
            rayl_23,
            givfac_23,
        )
        taug, taur = bands.taumol24(
            strrat,
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
            selfref_24,
            forref_24,
            absa_24,
            absb_24,
            abso3a_24,
            abso3b_24,
            rayla_24,
            raylb_24,
        )
        taug, taur = bands.taumol25(
            strrat,
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
            absa_25,
            abso3a_25,
            abso3b_25,
            rayl_25,
        )
        taug, taur = bands.taumol26(
            strrat,
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
            rayl_26,
        )
        taug, taur = bands.taumol27(
            strrat,
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
            absa_27,
            absb_27,
            rayl_27,
        )
        taug, taur = bands.taumol28(
            strrat,
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
            absa_28,
            absb_28,
            rayl_28,
        )
        taug, taur = bands.taumol29(
            strrat,
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
            forref_29,
            absa_29,
            absb_29,
            selfref_29,
            absh2o_29,
            absco2_29,
            rayl_29,
        )

        return sfluxzen, taug, taur

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

        ## data loading 
        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_sflux_data.nc"))
        strrat = ds["strrat"].values
        specwt = ds["specwt"].values
        layreffr = ds["layreffr"].values
        ix1 = ds["ix1"].values
        ix2 = ds["ix2"].values
        ibx = ds["ibx"].values
        sfluxref01 = ds["sfluxref01"].values
        sfluxref02 = ds["sfluxref02"].values
        sfluxref03 = ds["sfluxref03"].values
        scalekur = ds["scalekur"].values

        ## loading data for taumol
        ds_bands = {}
        for nband in range(16,30):
            ds_bands['radsw_kgb' + str(nband)] = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_kgb" + str(nband) + "_data.nc"))
        ## data loading for setcoef
        ds = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_ref_data.nc"))
        preflog = ds["preflog"].values
        tref = ds["tref"].values

        ## load data for cldprop
        ds_cldprtb = xr.open_dataset(os.path.join(LOOKUP_DIR, "radsw_cldprtb_data.nc"))
        extliq1 = ds_cldprtb["extliq1"].values
        extliq2 = ds_cldprtb["extliq2"].values
        ssaliq1 = ds_cldprtb["ssaliq1"].values
        ssaliq2 = ds_cldprtb["ssaliq2"].values
        asyliq1 = ds_cldprtb["asyliq1"].values
        asyliq2 = ds_cldprtb["asyliq2"].values
        extice2 = ds_cldprtb["extice2"].values
        ssaice2 = ds_cldprtb["ssaice2"].values
        asyice2 = ds_cldprtb["asyice2"].values
        extice3 = ds_cldprtb["extice3"].values
        ssaice3 = ds_cldprtb["ssaice3"].values
        asyice3 = ds_cldprtb["asyice3"].values
        abari = ds_cldprtb["abari"].values
        bbari = ds_cldprtb["bbari"].values
        cbari = ds_cldprtb["cbari"].values
        dbari = ds_cldprtb["dbari"].values
        ebari = ds_cldprtb["ebari"].values
        fbari = ds_cldprtb["fbari"].values
        b0s = ds_cldprtb["b0s"].values
        b1s = ds_cldprtb["b1s"].values
        b0r = ds_cldprtb["b0r"].values
        c0s = ds_cldprtb["c0s"].values
        c0r = ds_cldprtb["c0r"].values
        a0r = ds_cldprtb["a0r"].values
        a1r = ds_cldprtb["a1r"].values
        a0s = ds_cldprtb["a0s"].values
        a1s = ds_cldprtb["a1s"].values

        ## 
        ds = xr.open_dataset(self.rand_file)
        rand2d_data = ds["rand2d"].values

        ## data loading for taumol 
        selfref_16= ds_bands['radsw_kgb16']["selfref"].values
        forref_16 = ds_bands['radsw_kgb16']["forref"].values
        absa_16 = ds_bands['radsw_kgb16']["absa"].values
        absb_16 = ds_bands['radsw_kgb16']["absb"].values
        rayl_16 = ds_bands['radsw_kgb16']["rayl"].values

        selfref_17 = ds_bands['radsw_kgb17']["selfref"].values
        forref_17 = ds_bands['radsw_kgb17']["forref"].values
        absa_17 = ds_bands['radsw_kgb17']["absa"].values
        absb_17 = ds_bands['radsw_kgb17']["absb"].values
        rayl_17 = ds_bands['radsw_kgb17']["rayl"].values

        selfref_18 = ds_bands['radsw_kgb18']["selfref"].values
        forref_18 = ds_bands['radsw_kgb18']["forref"].values
        absa_18 = ds_bands['radsw_kgb18']["absa"].values
        absb_18 = ds_bands['radsw_kgb18']["absb"].values
        rayl_18 = ds_bands['radsw_kgb18']["rayl"].values

        selfref_19 = ds_bands['radsw_kgb19']["selfref"].values
        forref_19 = ds_bands['radsw_kgb19']["forref"].values
        absa_19 = ds_bands['radsw_kgb19']["absa"].values
        absb_19 = ds_bands['radsw_kgb19']["absb"].values
        rayl_19 = ds_bands['radsw_kgb19']["rayl"].values

        selfref_20 = ds_bands['radsw_kgb20']["selfref"].values
        forref_20  = ds_bands['radsw_kgb20']["forref"].values
        absa_20  = ds_bands['radsw_kgb20']["absa"].values
        absb_20  = ds_bands['radsw_kgb20']["absb"].values
        absch4_20  = ds_bands['radsw_kgb20']["absch4"].values
        rayl_20  = ds_bands['radsw_kgb20']["rayl"].values

        selfref_21 = ds_bands['radsw_kgb21']["selfref"].values
        forref_21 = ds_bands['radsw_kgb21']["forref"].values
        absa_21 = ds_bands['radsw_kgb21']["absa"].values
        absb_21 = ds_bands['radsw_kgb21']["absb"].values
        rayl_21 = ds_bands['radsw_kgb21']["rayl"].values

        selfref_22 = ds_bands['radsw_kgb22']["selfref"].values
        forref_22 = ds_bands['radsw_kgb22']["forref"].values
        absa_22 = ds_bands['radsw_kgb22']["absa"].values
        absb_22 = ds_bands['radsw_kgb22']["absb"].values
        rayl_22 = ds_bands['radsw_kgb22']["rayl"].values
            
        selfref_23 = ds_bands['radsw_kgb23']["selfref"].values
        forref_23 = ds_bands['radsw_kgb23']["forref"].values
        absa_23 = ds_bands['radsw_kgb23']["absa"].values
        rayl_23 = ds_bands['radsw_kgb23']["rayl"].values
        givfac_23 = ds_bands['radsw_kgb23']["givfac"].values

        selfref_24 = ds_bands['radsw_kgb24']["selfref"].values
        forref_24 = ds_bands['radsw_kgb24']["forref"].values
        absa_24 = ds_bands['radsw_kgb24']["absa"].values
        absb_24 = ds_bands['radsw_kgb24']["absb"].values
        abso3a_24 = ds_bands['radsw_kgb24']["abso3a"].values
        abso3b_24 = ds_bands['radsw_kgb24']["abso3b"].values
        rayla_24 = ds_bands['radsw_kgb24']["rayla"].values
        raylb_24 = ds_bands['radsw_kgb24']["raylb"].values
    
        absa_25 = ds_bands['radsw_kgb25']["absa"].values
        abso3a_25 = ds_bands['radsw_kgb25']["abso3a"].values
        abso3b_25 = ds_bands['radsw_kgb25']["abso3b"].values
        rayl_25 = ds_bands['radsw_kgb25']["rayl"].values

        rayl_26 = ds_bands['radsw_kgb26']["rayl"].values
        
        absa_27 = ds_bands['radsw_kgb27']["absa"].values
        absb_27 = ds_bands['radsw_kgb27']["absb"].values
        rayl_27 = ds_bands['radsw_kgb27']["rayl"].values
        
        absa_28 = ds_bands['radsw_kgb28']["absa"].values
        absb_28 = ds_bands['radsw_kgb28']["absb"].values
        rayl_28 = ds_bands['radsw_kgb28']["rayl"].values

        forref_29 = ds_bands['radsw_kgb29']["forref"].values
        absa_29 = ds_bands['radsw_kgb29']["absa"].values
        absb_29 = ds_bands['radsw_kgb29']["absb"].values
        selfref_29 = ds_bands['radsw_kgb29']["selfref"].values
        absh2o_29 = ds_bands['radsw_kgb29']["absh2o"].values
        absco2_29 = ds_bands['radsw_kgb29']["absco2"].values
        rayl_29 = ds_bands['radsw_kgb29']["rayl"].values
        
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

            pavel = plyr[j1, :]
            tavel = tlyr[j1, :]
            delp = delpin[j1, :]
            dz = dzlyr[j1, :]

            #  --- ...  set absorber amount
            # ncep model use
            h2ovmr = np.maximum(0.0, qlyr[j1, :] * self.amdw / (1.0 - qlyr[j1, :]))  # input specific humidity
            o3vmr  = np.maximum(0.0, olyr[j1, :] * self.amdo3)  # input mass mixing ratio

            tem0 = (1.0 - h2ovmr) * con_amd + h2ovmr * con_amw
            coldry = tem2 * delp / (tem1 * tem0 * (1.0 + h2ovmr))
            temcol = 1.0e-12 * coldry

            colamt[:, 0] = np.maximum(0.0, coldry * h2ovmr)  # h2o
            colamt[:, 1] = np.maximum(temcol, coldry * gasvmr[j1, :, 0])  # co2
            colamt[:, 2] = np.maximum(0.0, coldry * o3vmr)  # o3
            colmol = coldry + colamt[:, 0]

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
                colamt[:, 3] = np.maximum(temcol, coldry * gasvmr[j1, :, 1])  # n2o
                colamt[:, 4] = np.maximum(temcol, coldry * gasvmr[j1, :, 2])  # ch4
                colamt[:, 5] = np.maximum(temcol, coldry * gasvmr[j1, :, 3])  # o2
            else:
                    colamt[:, 3] = temcol  # n2o
                    colamt[:, 4] = temcol  # ch4
                    colamt[:, 5] = temcol  # o2
            #  --- ...  set aerosol optical properties

            for ib in range(nbdsw):
                tauae[:, ib] = aerosols[j1, :, ib, 0]
                ssaae[:, ib] = aerosols[j1, :, ib, 1]
                asyae[:, ib] = aerosols[j1, :, ib, 2]

            if iswcliq > 0:  # use prognostic cloud method
                cfrac = clouds[j1, :, 0]  # cloud fraction
                cliqp = clouds[j1, :, 1]  # cloud liq path
                reliq = clouds[j1, :, 2]  # liq partical effctive radius
                cicep = clouds[j1, :, 3]  # cloud ice path
                reice = clouds[j1, :, 4]  # ice partical effctive radius
                cdat1 = clouds[j1, :, 5]  # cloud rain drop path
                cdat2 = clouds[j1, :, 6]  # rain partical effctive radius
                cdat3 = clouds[j1, :, 7]  # cloud snow path
                cdat4 = clouds[j1, :, 8]  # snow partical effctive radius
            else:  # use diagnostic cloud method
                cfrac = clouds[j1, :, 0]  # cloud fraction
                cdat1 = clouds[j1, :, 1]  # cloud optical depth
                cdat2 = clouds[j1, :, 2]  # cloud single scattering albedo
                cdat3 = clouds[j1, :, 3]  # cloud asymmetry factor

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
                taucw, ssacw, asycw, cldfrc, cldfmc = cldprop(
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
                    rand2d_data, 
                    extliq1, 
                    extliq2, 
                    ssaliq1, 
                    ssaliq2, 
                    asyliq1, 
                    asyliq2, 
                    extice2, 
                    ssaice2, 
                    asyice2, 
                    extice3, 
                    ssaice3, 
                    asyice3, 
                    abari,  
                    bbari, 
                    cbari, 
                    dbari,  
                    ebari,  
                    fbari, 
                    b0s, 
                    b1s, 
                    b0r, 
                    c0s,  
                    c0r,  
                    a0r, 
                    a1r, 
                    a0s, 
                    a1s, 
                    self.ftiny,
                    np.array(self.idxebc),
                    self.isubcsw,
                    self.iovrsw,
                )

                #  --- ...  save computed layer cloud optical depth for output
                #           rrtm band 10 is approx to the 0.55 mu spectrum

                cldtau[j1, :] = taucw[:, 9]
            else:
                cldfrc[:] = 0.0
                cldfmc[:, :] = 0.0
                taucw[:, :] = 0.0
                ssacw[:, :] = 0.0
                asycw[:, :] = 0.0

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
            ) = self.setcoef(pavel, tavel, h2ovmr, nlay, nlp1, preflog, tref)

            # -# Call taumol() to calculate optical depths for gaseous absorption
            #    and rayleigh scattering
            sfluxzen, taug, taur = taumol(
                nspa,
                nspb,
                ng,
                ngs,
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
                strrat,
                specwt,
                layreffr,
                ix1,
                ix2 ,
                ibx,
                sfluxref01,
                sfluxref02,
                sfluxref03,
                scalekur,
                selfref_16,
                forref_16,
                absa_16,
                absb_16,
                rayl_16,
                selfref_17,
                forref_17,
                absa_17,
                absb_17,
                rayl_17,
                selfref_18,
                forref_18,
                absa_18,
                absb_18,
                rayl_18,
                selfref_19,
                forref_19,
                absa_19,
                absb_19,
                rayl_19,
                selfref_20,
                forref_20,
                absa_20,
                absb_20,
                absch4_20,
                rayl_20,
                selfref_21,
                forref_21,
                absa_21,
                absb_21,
                rayl_21,
                selfref_22,
                forref_22,
                absa_22,
                absb_22,
                rayl_22,
                selfref_23,
                forref_23,
                absa_23,
                rayl_23,
                givfac_23,
                selfref_24,
                forref_24,
                absa_24,
                absb_24,
                abso3a_24,
                abso3b_24,
                rayla_24,
                raylb_24,
                absa_25,
                abso3a_25,
                abso3b_25,
                rayl_25,
                rayl_26,
                absa_27,
                absb_27,
                rayl_27,
                absa_28,
                absb_28,
                rayl_28,
                forref_29,
                absa_29,
                absb_29,
                selfref_29,
                absh2o_29,
                absco2_29,
                rayl_29,
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
            ) = spcvrtm(
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
                np.array(self.idxsfc),
                self.ftiny,
                self.eps,
                self.nuvb,
                self.exp_tbl,
                self.bpade,
                self.flimit,
                self.oneminus,
                np.array(NGB),
            )

            # -# Save outputs.
            #  --- ...  sum up total spectral fluxes for total-sky

            flxuc = np.nansum(fxupc , axis = 1)
            flxdc = np.nansum(fxdnc , axis = 1)

            # --- ...  optional clear sky fluxes
            if self.lhsw0 or self.lflxprf:
                flxu0 = np.nansum(fxup0 , axis = 1)
                flxd0 = np.nansum(fxdn0, axis = 1)
            #  --- ...  prepare for final outputs
            #for k in range(nlay):
            rfdelp = self.heatfac / delp

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
            fnet[1:] = flxdc[1:] - flxuc[1:]
            hswc[j1, :] = (fnet[1:] - fnet[:-1]) * rfdelp

            # --- ...  optional flux profiles

            if self.lflxprf:
                upfxc_f[j1, :] = flxuc
                dnfxc_f[j1, :] = flxdc
                upfx0_f[j1, :] = flxu0
                dnfx0_f[j1, :] = flxd0

            # --- ...  optional clear sky heating rates

            if self.lhsw0:
                fnet[0] = flxd0[0] - flxu0[0]
                fnet[1:] = flxd0[1:] - flxu0[1:]
                hsw0[j1, :] = (fnet[1:] - fnet[:-1]) * rfdelp

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

    def setcoef(self, pavel, tavel, h2ovmr, nlay, nlp1, preflog, tref):
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
