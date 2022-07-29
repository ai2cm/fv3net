import sys
import os
import numpy as np
import xarray as xr

sys.path.insert(0, "..")
from radphysparam import semis_file
from phys_const import con_tice, con_ttp, con_t0c, con_pi
from config import *


class SurfaceClass:
    VTAGSFC = "NCEP-Radiation_surface   v5.1  Nov 2012"

    NF_ALBD = 4
    rad2dg = 180.0 / con_pi
    IMXEMS = 360
    JMXEMS = 180

    def __init__(self, me, ialb, iems):

        self.ialbflg = ialb
        self.iemsflg = iems
        self.semis_file = os.path.join(FORCING_DIR, semis_file)

        if me == 0:
            print(self.VTAGSFC)  # print out version tag

        # - Initialization of surface albedo section
        # physparam::ialbflg
        # = 0: using climatology surface albedo scheme for SW
        # = 1: using MODIS based land surface albedo for SW

        if self.ialbflg == 0:
            if me == 0:
                print("- Using climatology surface albedo scheme for sw")

        elif self.ialbflg == 1:
            if me == 0:
                print("- Using MODIS based land surface albedo for sw")
        else:
            raise ValueError(f"!! ERROR in Albedo Scheme Setting, IALB={self.ialbflg}")

        # - Initialization of surface emissivity section
        # physparam::iemsflg
        # = 0: fixed SFC emissivity at 1.0
        # = 1: input SFC emissivity type map from "semis_file"

        self.iemslw = self.iemsflg % 10  # emissivity control
        if self.iemslw == 0:  # fixed sfc emis at 1.0
            if me == 0:
                print("- Using Fixed Surface Emissivity = 1.0 for lw")

        elif self.iemslw == 1:  # input sfc emiss type map
            if "idxems" not in vars():
                idxems = np.zeros((self.IMXEMS, self.JMXEMS))

                file_exist = os.path.isfile(self.semis_file)

                if not file_exist:
                    if me == 0:
                        print("- Using Varying Surface Emissivity for lw")
                        print(f'Requested data file "{semis_file}" not found!')
                        print("Change to fixed surface emissivity = 1.0 !")

                    self.iemslw = 0
                else:
                    ds = xr.open_dataset(self.semis_file)

                    cline = ds["cline"].data
                    idxems = ds["idxems"].data

                    if me == 0:
                        print("- Using Varying Surface Emissivity for lw")
                        print(f"Opened data file: {semis_file}")
                        print(cline)
            else:
                raise ValueError(
                    f"!! ERROR in Emissivity Scheme Setting, IEMS={self.iemsflg}"
                )

        self.cline = cline
        self.idxems = idxems

    def return_initdata(self):
        outdict = {"idxems": self.idxems}
        return outdict

    def setalb(
        self,
        slmsk,
        snowf,
        sncovr,
        snoalb,
        zorlf,
        coszf,
        tsknf,
        tairf,
        hprif,
        alvsf,
        alnsf,
        alvwf,
        alnwf,
        facsf,
        facwf,
        fice,
        tisfc,
        IMAX,
        albPpert,
        pertalb,
    ):
        #  ===================================================================  !
        #                                                                       !
        #  this program computes four components of surface albedos (i.e.       !
        #  vis-nir, direct-diffused) according to controflag ialbflg.           !
        #   1) climatological surface albedo scheme (briegleb 1992)             !
        #   2) modis retrieval based scheme from boston univ.                   !
        #                                                                       !
        #                                                                       !
        # usage:         call setalb                                            !
        #                                                                       !
        # subprograms called:  none                                             !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                              !
        #     slmsk (IMAX)  - sea(0),land(1),ice(2) mask on fcst model grid     !
        #     snowf (IMAX)  - snow depth water equivalent in mm                 !
        #     sncovr(IMAX)  - ialgflg=0: not used                               !
        #                     ialgflg=1: snow cover over land in fraction       !
        #     snoalb(IMAX)  - ialbflg=0: not used                               !
        #                     ialgflg=1: max snow albedo over land in fraction  !
        #     zorlf (IMAX)  - surface roughness in cm                           !
        #     coszf (IMAX)  - cosin of solar zenith angle                       !
        #     tsknf (IMAX)  - ground surface temperature in k                   !
        #     tairf (IMAX)  - lowest model layer air temperature in k           !
        #     hprif (IMAX)  - topographic sdv in m                              !
        #           ---  for ialbflg=0 climtological albedo scheme  ---         !
        #     alvsf (IMAX)  - 60 degree vis albedo with strong cosz dependency  !
        #     alnsf (IMAX)  - 60 degree nir albedo with strong cosz dependency  !
        #     alvwf (IMAX)  - 60 degree vis albedo with weak cosz dependency    !
        #     alnwf (IMAX)  - 60 degree nir albedo with weak cosz dependency    !
        #           ---  for ialbflg=1 modis based land albedo scheme ---       !
        #     alvsf (IMAX)  - visible black sky albedo at zenith 60 degree      !
        #     alnsf (IMAX)  - near-ir black sky albedo at zenith 60 degree      !
        #     alvwf (IMAX)  - visible white sky albedo                          !
        #     alnwf (IMAX)  - near-ir white sky albedo                          !
        #                                                                       !
        #     facsf (IMAX)  - fractional coverage with strong cosz dependency   !
        #     facwf (IMAX)  - fractional coverage with weak cosz dependency     !
        #     fice  (IMAX)  - sea-ice fraction                                  !
        #     tisfc (IMAX)  - sea-ice surface temperature                       !
        #     IMAX          - array horizontal dimension                        !
        #                                                                       !
        #  outputs:                                                             !
        #     sfcalb(IMAX,NF_ALBD)                                              !
        #           ( :, 1) -     near ir direct beam albedo                    !
        #           ( :, 2) -     near ir diffused albedo                       !
        #           ( :, 3) -     uv+vis direct beam albedo                     !
        #           ( :, 4) -     uv+vis diffused albedo                        !
        #                                                                       !
        #  module internal control variables:                                   !
        #     ialbflg       - =0 use the default climatology surface albedo     !
        #                     =1 use modis retrieved albedo and input snow cover!
        #                        for land areas                                 !
        #                                                                       !
        #  ====================    end of description    =====================  !
        #

        #  ---  outputs
        sfcalb = np.zeros((IMAX, self.NF_ALBD))

        # If use climatological albedo scheme:
        if self.ialbflg == 0:  # use climatological albedo scheme

            for i in range(IMAX):

                #    - Modified snow albedo scheme - units convert to m (originally
                #      snowf in mm; zorlf in cm)

                asnow = 0.02 * snowf[i]
                argh = min(0.50, max(0.025, 0.01 * zorlf[i]))
                hrgh = min(1.0, max(0.20, 1.0577 - 1.1538e-3 * hprif[i]))
                fsno0 = asnow / (argh + asnow) * hrgh
                if round(slmsk[i]) == 0 and tsknf[i] > con_tice:
                    fsno0 = 0.0
                fsno1 = 1.0 - fsno0
                flnd0 = min(1.0, facsf[i] + facwf[i])
                fsea0 = max(0.0, 1.0 - flnd0)
                fsno = fsno0
                fsea = fsea0 * fsno1
                flnd = flnd0 * fsno1

                #    - Calculate diffused sea surface albedo

                if tsknf[i] >= 271.5:
                    asevd = 0.06
                    asend = 0.06
                elif tsknf[i] < 271.1:
                    asevd = 0.70
                    asend = 0.65
                else:
                    a1 = (tsknf[i] - 271.1) ** 2
                    asevd = 0.7 - 4.0 * a1
                    asend = 0.65 - 3.6875 * a1

                #    - Calculate diffused snow albedo.

                if round(slmsk[i]) == 2:
                    ffw = 1.0 - fice[i]
                    if ffw < 1.0:
                        dtgd = max(0.0, min(5.0, (con_ttp - tisfc[i])))
                        b1 = 0.03 * dtgd
                    else:
                        b1 = 0.0

                    b3 = 0.06 * ffw
                    asnvd = (0.70 + b1) * fice[i] + b3
                    asnnd = (0.60 + b1) * fice[i] + b3
                    asevd = 0.70 * fice[i] + b3
                    asend = 0.60 * fice[i] + b3
                else:
                    asnvd = 0.90
                    asnnd = 0.75

                #    - Calculate direct snow albedo.

                if coszf[i] < 0.5:
                    csnow = 0.5 * (3.0 / (1.0 + 4.0 * coszf[i]) - 1.0)
                    asnvb = min(0.98, asnvd + (1.0 - asnvd) * csnow)
                    asnnb = min(0.98, asnnd + (1.0 - asnnd) * csnow)
                else:
                    asnvb = asnvd
                    asnnb = asnnd

                #    - Calculate direct sea surface albedo.

                if coszf[i] > 0.0001:
                    rfcs = 1.4 / (1.0 + 0.8 * coszf[i])
                    rfcw = 1.1 / (1.0 + 0.2 * coszf[i])

                    if tsknf[i] >= con_t0c:
                        asevb = max(
                            asevd,
                            0.026 / (coszf[i] ** 1.7 + 0.065)
                            + 0.15
                            * (coszf[i] - 0.1)
                            * (coszf[i] - 0.5)
                            * (coszf[i] - 1.0),
                        )
                        asenb = asevb
                    else:
                        asevb = asevd
                        asenb = asend
                else:
                    rfcs = 1.0
                    rfcw = 1.0
                    asevb = asevd
                    asenb = asend

                a1 = alvsf[i] * facsf[i]
                b1 = alvwf[i] * facwf[i]
                a2 = alnsf[i] * facsf[i]
                b2 = alnwf[i] * facwf[i]
                ab1bm = a1 * rfcs + b1 * rfcw
                ab2bm = a2 * rfcs + b2 * rfcw
                sfcalb[i, 0] = min(0.99, ab2bm) * flnd + asenb * fsea + asnnb * fsno
                sfcalb[i, 1] = (a2 + b2) * 0.96 * flnd + asend * fsea + asnnd * fsno
                sfcalb[i, 2] = min(0.99, ab1bm) * flnd + asevb * fsea + asnvb * fsno
                sfcalb[i, 3] = (a1 + b1) * 0.96 * flnd + asevd * fsea + asnvd * fsno

        # If use modis based albedo for land area:
        else:

            for i in range(IMAX):

                #    - Calculate snow cover input directly for land model, no
                #      conversion needed.

                fsno0 = sncovr[i]

                if round(slmsk[i]) == 0 and tsknf[i] > con_tice:
                    fsno0 = 0.0

                if round(slmsk[i]) == 2:
                    asnow = 0.02 * snowf[i]
                    argh = min(0.50, max(0.025, 0.01 * zorlf[i]))
                    hrgh = min(1.0, max(0.20, 1.0577 - 1.1538e-3 * hprif[i]))
                    fsno0 = asnow / (argh + asnow) * hrgh

                fsno1 = 1.0 - fsno0
                flnd0 = min(1.0, facsf[i] + facwf[i])
                fsea0 = max(0.0, 1.0 - flnd0)
                fsno = fsno0
                fsea = fsea0 * fsno1
                flnd = flnd0 * fsno1

                #    - Calculate diffused sea surface albedo.

                if tsknf[i] >= 271.5:
                    asevd = 0.06
                    asend = 0.06
                elif tsknf[i] < 271.1:
                    asevd = 0.70
                    asend = 0.65
                else:
                    a1 = (tsknf[i] - 271.1) ** 2
                    asevd = 0.7 - 4.0 * a1
                    asend = 0.65 - 3.6875 * a1

                #    - Calculate diffused snow albedo, land area use input max snow
                #      albedo.

                if round(slmsk[i]) == 2:
                    ffw = 1.0 - fice[i]
                    if ffw < 1.0:
                        dtgd = max(0.0, min(5.0, (con_ttp - tisfc[i])))
                        b1 = 0.03 * dtgd
                    else:
                        b1 = 0.0

                    b3 = 0.06 * ffw
                    asnvd = (0.70 + b1) * fice[i] + b3
                    asnnd = (0.60 + b1) * fice[i] + b3
                    asevd = 0.70 * fice[i] + b3
                    asend = 0.60 * fice[i] + b3
                else:
                    asnvd = snoalb[i]
                    asnnd = snoalb[i]

                #    - Calculate direct snow albedo.

                if round(slmsk[i]) == 2:
                    if coszf[i] < 0.5:
                        csnow = 0.5 * (3.0 / (1.0 + 4.0 * coszf[i]) - 1.0)
                        asnvb = min(0.98, asnvd + (1.0 - asnvd) * csnow)
                        asnnb = min(0.98, asnnd + (1.0 - asnnd) * csnow)
                    else:
                        asnvb = asnvd
                        asnnb = asnnd
                else:
                    asnvb = snoalb[i]
                    asnnb = snoalb[i]

                #    - Calculate direct sea surface albedo, use fanglin's zenith angle
                #      treatment.

                if coszf[i] > 0.0001:
                    rfcs = 1.775 / (1.0 + 1.55 * coszf[i])

                    if tsknf[i] >= con_t0c:
                        asevb = max(
                            asevd,
                            0.026 / (coszf[i] ** 1.7 + 0.065)
                            + 0.15
                            * (coszf[i] - 0.1)
                            * (coszf[i] - 0.5)
                            * (coszf[i] - 1.0),
                        )
                        asenb = asevb
                    else:
                        asevb = asevd
                        asenb = asend
                else:
                    rfcs = 1.0
                    asevb = asevd
                    asenb = asend

                ab1bm = min(0.99, alnsf[i] * rfcs)
                ab2bm = min(0.99, alvsf[i] * rfcs)
                sfcalb[i, 0] = ab1bm * flnd + asenb * fsea + asnnb * fsno
                sfcalb[i, 1] = alnwf[i] * flnd + asend * fsea + asnnd * fsno
                sfcalb[i, 2] = ab2bm * flnd + asevb * fsea + asnvb * fsno
                sfcalb[i, 3] = alvwf[i] * flnd + asevd * fsea + asnvd * fsno

        # sfc-perts, mgehne ***
        # perturb all 4 kinds of surface albedo, sfcalb(:,1:4)
        if pertalb[0] > 0.0:
            for i in range(IMAX):
                for kk in range(4):
                    # compute beta distribution parameters for all 4 albedos
                    m = sfcalb[i, kk]
                    s = pertalb[0] * m * (1.0 - m)
                    alpha = m * m * (1.0 - m) / (s * s) - m
                    beta = alpha * (1.0 - m) / m
                    # compute beta distribution value corresponding
                    # to the given percentile albPpert to use as new albedo
                    albtmp = ppfbet(albPpert[i], alpha, beta, iflag)
                    sfcalb[i, kk] = albtmp

        return sfcalb

    def setemis(
        self, xlon, xlat, slmsk, snowf, sncovr, zorlf, tsknf, tairf, hprif, IMAX
    ):
        #  ===================================================================  !
        #                                                                       !
        #  this program computes surface emissivity for lw radiation.           !
        #                                                                       !
        #  usage:         call setemis                                          !
        #                                                                       !
        #  subprograms called:  none                                            !
        #                                                                       !
        #  ====================  defination of variables  ====================  !
        #                                                                       !
        #  inputs:                                                              !
        #     xlon  (IMAX)  - longitude in radiance, ok for both 0->2pi or      !
        #                     -pi -> +pi ranges                                 !
        #     xlat  (IMAX)  - latitude  in radiance, default to pi/2 -> -pi/2   !
        #                     range, otherwise see in-line comment              !
        #     slmsk (IMAX)  - sea(0),land(1),ice(2) mask on fcst model grid     !
        #     snowf (IMAX)  - snow depth water equivalent in mm                 !
        #     sncovr(IMAX)  - ialbflg=1: snow cover over land in fraction       !
        #     zorlf (IMAX)  - surface roughness in cm                           !
        #     tsknf (IMAX)  - ground surface temperature in k                   !
        #     tairf (IMAX)  - lowest model layer air temperature in k           !
        #     hprif (IMAX)  - topographic sdv in m                              !
        #     IMAX          - array horizontal dimension                        !
        #                                                                       !
        #  outputs:                                                             !
        #     sfcemis(IMAX) - surface emissivity                                !
        #                                                                       !
        #  -------------------------------------------------------------------  !
        #                                                                       !
        #  surface type definations:                                            !
        #     1. open water                   2. grass/wood/shrub land          !
        #     3. tundra/bare soil             4. sandy desert                   !
        #     5. rocky desert                 6. forest                         !
        #     7. ice                          8. snow                           !
        #                                                                       !
        #  input index data lon from 0 towards east, lat from n to s            !
        #                                                                       !
        #  ====================    end of description    =====================  !
        #

        sfcemis = np.zeros(IMAX)

        #  ---  reference emiss value for diff surface emiss index
        #       1-open water, 2-grass/shrub land, 3-bare soil, tundra,
        #       4-sandy desert, 5-rocky desert, 6-forest, 7-ice, 8-snow

        emsref = [0.97, 0.95, 0.94, 0.90, 0.93, 0.96, 0.96, 0.99]

        # Set sfcemis default to 1.0 or by surface type and condition.
        if self.iemslw == 0:  # sfc emiss default to 1.0
            sfcemis[:] = 1.0
            return

        else:  # emiss set by sfc type and condition

            dltg = 360.0 / self.IMXEMS
            hdlt = 0.5 * dltg

            #  --- ...  mapping input data onto model grid
            #           note: this is a simple mapping method, an upgrade is needed if
            #           the model grid is much corcer than the 1-deg data resolution

            for i in range(IMAX):
                if round(slmsk[i]) == 0:  # sea point
                    sfcemis[i] = emsref[0]
                elif round(slmsk[i]) == 2:  # sea-ice
                    sfcemis[i] = emsref[6]
                else:  # land

                    #  ---  map grid in longitude direction
                    i2 = 1
                    j2 = 1

                    tmp1 = xlon[i] * self.rad2dg
                    if tmp1 < 0.0:
                        tmp1 += 360.0

                    for i1 in range(self.IMXEMS):
                        tmp2 = dltg * i1 + hdlt

                        if abs(tmp1 - tmp2) <= hdlt:
                            i2 = i1
                            break

                    #   ---  map grid in latitude direction
                    tmp1 = xlat[i] * self.rad2dg  # if xlat in pi/2 -> -pi/2 range

                    for j1 in range(self.JMXEMS):
                        tmp2 = 90.0 - dltg * j1

                        if abs(tmp1 - tmp2) <= hdlt:
                            j2 = j1
                            break

                    idx = max(2, self.idxems[i2, j2]) - 1
                    if idx >= 6:
                        idx = 1
                    sfcemis[i] = emsref[idx]

                # Check for snow covered area.

                if (
                    self.ialbflg == 1 and round(slmsk[i]) == 1
                ):  # input land area snow cover

                    fsno0 = sncovr[i]
                    fsno1 = 1.0 - fsno0
                    sfcemis[i] = sfcemis[i] * fsno1 + emsref[7] * fsno0

                else:  # compute snow cover from snow depth
                    if snowf[i] > 0.0:
                        asnow = 0.02 * snowf[i]
                        argh = min(0.50, max(0.025, 0.01 * zorlf[i]))
                        hrgh = min(1.0, max(0.20, 1.0577 - 1.1538e-3 * hprif[i]))
                        fsno0 = asnow / (argh + asnow) * hrgh
                        if round(slmsk[i]) == 0 and tsknf[i] > 271.2:
                            fsno0 = 0.0

                        fsno1 = 1.0 - fsno0
                        sfcemis[i] = sfcemis[i] * fsno1 + emsref[7] * fsno0

        return sfcemis
