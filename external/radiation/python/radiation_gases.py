import numpy as np
import xarray as xr
import os
import sys

sys.path.insert(0, "..")
from radphysparam import co2cyc_file, co2gbl_file, co2dat_file
from phys_const import con_pi
from config import *


class GasClass:
    VTAGGAS = "NCEP-Radiation_gases     v5.1  Nov 2012"
    NF_VGAS = 10
    IMXCO2 = 24
    JMXCO2 = 12
    co2vmr_def = 350.0e-6
    MINYEAR = 1957

    resco2 = 15.0
    raddeg = 180.0 / con_pi
    prsco2 = 788.0
    hfpi = 0.5 * con_pi

    n2ovmr_def = 0.31e-6
    ch4vmr_def = 1.50e-6
    o2vmr_def = 0.209
    covmr_def = 1.50e-8
    f11vmr_def = 3.520e-10
    f12vmr_def = 6.358e-10
    f22vmr_def = 1.500e-10
    cl4vmr_def = 1.397e-10
    f113vmr_def = 8.2000e-11

    def __init__(self, me, iozn, ico2, ictm):
        self.kyrsav = 0
        self.kmonsav = 1

        self.me = me
        self.ioznflg = iozn
        self.ico2flg = ico2
        self.ictmflg = ictm

        self.co2cyc_file = os.path.join(FORCING_DIR, co2cyc_file)
        self.co2gbl_file = os.path.join(FORCING_DIR, co2gbl_file)
        self.co2dat_file = os.path.join(FORCING_DIR, co2dat_file)

        if self.ioznflg > 0:
            if self.me == 0:
                print(" - Using interactive ozone distribution")
        else:
            print("Climatological ozone data not implemented")

        #  --- ...  co2 data section

        self.co2_glb = self.co2vmr_def
        self.gco2cyc = np.zeros(12)

        if self.ico2flg == 0:
            if me == 0:
                print(f"- Using prescribed co2 global mean value={self.co2vmr_def}")

        else:
            if self.ictmflg == -1:  # input user provided data
                print("ictmflg = -1 is not implemented")

            else:  # input from observed data
                if self.ico2flg == 1:
                    print("Using observed co2 global annual mean value")

                elif self.ico2flg == 2:
                    if me == 0:
                        print("Using observed co2 monthly 2-d data")
                        self.co2vmr_sav = np.zeros((self.IMXCO2, self.JMXCO2, 12))

                else:
                    raise ValueError(
                        f" ICO2={self.ico2flg}, is not a valid selection",
                        " - Stoped in subroutine gas_init!!!",
                    )

            if self.ictmflg == -2:
                file_exist = os.path.isfile(self.co2cyc_file)
                if not file_exist:
                    raise FileNotFoundError(
                        "Can not find seasonal cycle CO2 data: ",
                        f"{co2cyc_file} - Stopped in subroutine gas_init !!",
                    )
                else:
                    co2cyc_sav = np.zeros((self.IMXCO2, self.JMXCO2, 12))
                    ds = xr.open_dataset(self.co2cyc_file)
                    #  --- ...  read in co2 2-d seasonal cycle data
                    cline = ds["cline"].data
                    co2g1 = ds["co2g1"].data
                    co2g2 = ds["co2g2"].data
                    co2dat = ds["co2dat"]

                    if me == 0:
                        print(" - Superimpose seasonal cycle to mean CO2 data")
                        print(
                            "Opened CO2 climatology seasonal cycle data",
                            f" file: {co2cyc_file}",
                        )

                    gco2cyc = ds["gco2cyc"]
                    gco2cyc = gco2cyc * 1.0e-6

                    self.co2cyc_sav = co2dat

    def return_initdata(self):
        outdict = {"co2cyc_sav": 0}
        return outdict

    def return_updatedata(self):
        outdict = {"co2vmr_sav": self.co2vmr_sav, "gco2cyc": self.gco2cyc}
        return outdict

    def gas_update(self, iyear, imon, iday, ihour, loz1st, ldoco2, me):
        #  ===================================================================  !
        #                                                                       !
        #  gas_update reads in 2-d monthly co2 data set for a specified year.   !
        #  data are in a 15 degree lat/lon horizontal resolution.               !
        #                                                                       !
        #  inputs:                                               dimemsion      !
        #     iyear   - year of the requested data for fcst         1           !
        #     imon    - month of the year                           1           !
        #     iday    - day of the month                            1           !
        #     ihour   - hour of the day                             1           !
        #     loz1st  - clim ozone 1st time update control flag     1           !
        #     ldoco2  - co2 update control flag                     1           !
        #     me      - print message control flag                  1           !
        #                                                                       !
        #  outputs: (to the module variables)                                   !
        #    ( none )                                                           !
        #                                                                       !
        #  external module variables:  (in physparam)                           !
        #     ico2flg    - co2 data source control flag                         !
        #                   =0: use prescribed co2 global mean value            !
        #                   =1: use input global mean co2 value (co2_glb)       !
        #                   =2: use input 2-d monthly co2 value (co2vmr_sav)    !
        #     ictmflg    - =yyyy#, data ic time/date control flag               !
        #                  =   -2: same as 0, but superimpose seasonal cycle    !
        #                          from climatology data set.                   !
        #                  =   -1: use user provided external data for the fcst !
        #                          time, no extrapolation.                      !
        #                  =    0: use data at initial cond time, if not existed!
        #                          then use latest, without extrapolation.      !
        #                  =    1: use data at the forecast time, if not existed!
        #                          then use latest and extrapolate to fcst time.!
        #                  =yyyy0: use yyyy data for the forecast time, no      !
        #                          further data extrapolation.                  !
        #                  =yyyy1: use yyyy data for the fcst. if needed, do    !
        #                          extrapolation to match the fcst time.        !
        #     ioznflg    - ozone data control flag                              !
        #                   =0: use climatological ozone profile                !
        #                   >0: use interactive ozone profile                   !
        #     ivflip     - vertical profile indexing flag                       !
        #     co2dat_file- external co2 2d monthly obsv data table              !
        #     co2gbl_file- external co2 global annual mean data table           !
        #                                                                       !
        #  internal module variables:                                           !
        #     co2vmr_sav - monthly co2 volume mixing ratio     IMXCO2*JMXCO2*12 !
        #     co2cyc_sav - monthly cycle co2 vol mixing ratio  IMXCO2*JMXCO2*12 !
        #     co2_glb    - global annual mean co2 mixing ratio                  !
        #     gco2cyc    - global monthly mean co2 variation       12           !
        #     k1oz,k2oz,facoz                                                   !
        #                - climatology ozone parameters             1           !
        #                                                                       !
        #  usage:    call gas_update                                            !
        #                                                                       !
        #  subprograms called:  none                                            !
        #                                                                       !
        #  ===================================================================  !
        #

        co2dat = np.zeros((self.IMXCO2, self.JMXCO2))
        co2ann = np.zeros((self.IMXCO2, self.JMXCO2))
        co2vmr_sav = np.zeros((self.IMXCO2, self.JMXCO2, 12))

        midmon = 15
        midm = 15
        midp = 45
        #  ---  number of days in a month
        mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 30]

        #
        # ===>  ...  begin here
        #
        # - Ozone data section

        if self.ioznflg == 0:
            midmon = mdays[imon] / 2 + 1
            change = loz1st or ((iday == midmon) and (ihour == 0))

            if change:
                if iday < midmon:
                    k1oz = (imon + 10 % 12) + 1
                    midm = mdays[k1oz] / 2 + 1
                    k2oz = imon
                    midp = mdays[k1oz] + midmon
                else:
                    k1oz = imon
                    midm = midmon
                    k2oz = (imon % 12) + 1
                    midp = mdays[k2oz] / 2 + 1 + mdays[k1oz]

            if iday < midmon:
                id = iday + mdays[k1oz]
            else:
                id = iday

            facoz = float(id - midm) / float(midp - midm)

        # - co2 data section

        if self.ico2flg == 0:
            return  # use prescribed global mean co2 data
        if self.ictmflg == -1:
            return  # use user provided co2 data
        if not ldoco2:
            return  # no need to update co2 data

        if self.ictmflg < 0:  # use user provided external data
            lextpl = False  # no time extrapolation
            idyr = iyear  # use the model year
        else:  # use historically observed data
            lextpl = (self.ictmflg % 10) == 1  # flag for data extrapolation
            idyr = self.ictmflg // 10  # year of data source used
            if idyr == 0:
                idyr = iyear  # not specified, use model year

        #  --- ...  auto select co2 2-d data table for required year
        self.kmonsav = imon
        if self.kyrsav == iyear:
            return
        self.kyrsav = iyear
        iyr = iyear

        #  --- ...  for data earlier than MINYEAR (1957), the data are in
        #           the form of semi-yearly global mean values.  otherwise,
        #           data are monthly mean in horizontal 2-d map.

        if idyr < self.MINYEAR and self.ictmflg > 0:
            if me == 0:
                print(f"Requested CO2 data year {iyear} earlier than {self.MINYEAR}")
                print("Which is the earliest monthly observation", " data available.")
                print("Thus, historical global mean data is used")

            #  --- ... check to see if requested co2 data file existed

            file_exist = os.path.isfile(self.co2gbl_file)

            if not file_exist:
                raise FileNotFoundError(
                    f'Requested co2 data file "{co2gbl_file}" not found',
                    " - Stopped in subroutine gas_update!!",
                )
            else:
                ds = xr.open_dataset(self.co2gbl_file)
                iyr1 = ds["iyr1"]
                iyr2 = ds["iyr2"]
                cline = ds["cline"]

                if me == 0:
                    print(f"Opened co2 data file: {co2gbl_file}")

                if idyr < iyr1:
                    iyr = iyr1

                i = iyr2
                while i >= iyr1:
                    jyr = ds["jyr"]
                    co2g1 = ds["co2g1"]
                    co2g2 = ds["co2g2"]

                    if i == iyr and iyr == jyr:
                        co2_glb = (co2g1 + co2g2) * 0.5e-6
                        if self.ico2flg == 2:
                            for j in range(self.JMXCO2):
                                for i in range(self.IMXCO2):
                                    co2vmr_sav[i, j, :6] = co2g1 * 1.0e-6
                                    co2vmr_sav[i, j, 6:] = co2g2 * 1.0e-6

                        if me == 0:
                            print(f"Co2 data for year {iyear} = {co2_glb}")
                        break
                    else:
                        i -= 1

        else:  # Lab_if_idyr

            #  --- ...  set up input data file name

            cfile1 = co2dat_file
            cfile1 = co2dat_file[:18] + str(idyr) + co2dat_file[22:]
            cfile1 = os.path.join(FORCING_DIR, cfile1)

            #  --- ... check to see if requested co2 data file existed

            file_exist = os.path.isfile(cfile1)
            if not file_exist:
                if self.ictmflg > 10:  # specified year of data not found
                    raise FileNotFoundError(
                        f"Specified co2 data for year {idyr} not found !!",
                        "Need to change namelist ICTM !!",
                        "   *** Stopped in subroutine gas_update !!",
                    )
                else:  # looking for latest available data
                    print(
                        f"Requested co2 data for year {idyr}",
                        " not found, check for other available data set",
                    )

                    while iyr >= self.MINYEAR:
                        iyr -= 1
                        cfile1 = co2dat_file[:18] + str(iyr) + co2dat_file[22:]
                        cfile1 = os.path.join(FORCING_DIR, cfile1)

                        file_exist = os.path.isfile(cfile1)
                        if me == 0:
                            print(f"Looking for CO2 file {cfile1}")

                        if file_exist:
                            break

                    if not file_exist:
                        raise FileNotFoundError(
                            "   Can not find co2 data source file",
                            "   *** Stopped in subroutine gas_update !!",
                        )

            #  --- ...  read in co2 2-d data for the requested month
            ds = xr.open_dataset(cfile1)
            iyr = ds["iyr"].data
            cline = ds["cline"].data
            co2g1 = ds["co2g1"].data
            co2g2 = ds["co2g2"].data

            if me == 0:
                print(f"Opened co2 data file: {cfile1}")
                print(f"{iyr}, {cline} {co2g1},  GROWTH RATE = {co2g2}")

            #  --- ...  add growth rate if needed
            if lextpl:
                rate = 2.00 * (iyear - iyr)  # avg rate for recent period
            else:
                rate = 0.0

            self.co2_glb = (co2g1 + rate) * 1.0e-6
            if me == 0:
                print(f"Global annual mean CO2 data for year {iyear} = {self.co2_glb}")

            if self.ictmflg == -2:  # need to calc ic time annual mean first
                raise NotImplementedError(f"ictmflg = {self.ictmflg} Not implemented!")
            else:  # no need to calc ic time annual mean first
                if self.ico2flg == 2:
                    co2dat = ds["co2dat"].data

                    co2vmr_sav = (co2dat + rate) * 1.0e-6

                    if me == 0:
                        print(
                            "CHECK: Sample of selected months of CO2 ",
                            f"data used for year: {iyear}",
                        )
                        for imo in range(0, 12, 3):
                            print(f"Month = {imo+1}")
                            print(co2vmr_sav[0, :, imo])

                gco2cyc = np.zeros(12)

            self.co2vmr_sav = co2vmr_sav
            self.gco2cyc = gco2cyc

    def getgases(self, plvl, xlon, xlat, IMAX, LMAX):
        #  ===================================================================  !
        #                                                                       !
        #  getgases set up global distribution of radiation absorbing  gases    !
        #  in volume mixing ratio.  currently only co2 has the options from     !
        #  observed values, all other gases are asigned to the climatological   !
        #  values.                                                              !
        #                                                                       !
        #  inputs:                                                              !
        #     plvl(IMAX,LMAX+1)- pressure at model layer interfaces (mb)        !
        #     xlon(IMAX)       - grid longitude in radians, ok both 0->2pi or   !
        #                        -pi -> +pi arrangements                        !
        #     xlat(IMAX)       - grid latitude in radians, default range to     !
        #                        pi/2 -> -pi/2, otherwise see in-line comment   !
        #     IMAX, LMAX       - horiz, vert dimensions for output data         !
        #                                                                       !
        #  outputs:                                                             !
        #     gasdat(IMAX,LMAX,NF_VGAS) - gases volume mixing ratioes           !
        #               (:,:,1)           - co2                                 !
        #               (:,:,2)           - n2o                                 !
        #               (:,:,3)           - ch4                                 !
        #               (:,:,4)           - o2                                  !
        #               (:,:,5)           - co                                  !
        #               (:,:,6)           - cfc11                               !
        #               (:,:,7)           - cfc12                               !
        #               (:,:,8)           - cfc22                               !
        #               (:,:,9)           - ccl4                                !
        #               (:,:,10)          - cfc113                              !
        #                                                                       !
        #  external module variables:  (in physparam)                           !
        #     ico2flg    - co2 data source control flag                         !
        #                   =0: use prescribed co2 global mean value            !
        #                   =1: use input global mean co2 value (co2_glb)       !
        #                   =2: use input 2-d monthly co2 value (co2vmr_sav)    !
        #     ivflip     - vertical profile indexing flag                       !
        #                                                                       !
        #  internal module variables used:                                      !
        #     co2vmr_sav - saved monthly co2 concentration from sub gas_update  !
        #     co2_glb    - saved global annual mean co2 value from  gas_update  !
        #     gco2cyc    - saved global seasonal variation of co2 climatology   !
        #                  in 12-month form                                     !
        #  ** note: for lower atmos co2vmr_sav may have clim monthly deviations !
        #           superimposed on init-cond co2 value, while co2_glb only     !
        #           contains the global mean value, thus needs to add the       !
        #           monthly dglobal mean deviation gco2cyc at upper atmos. for  !
        #           ictmflg/=-2, this value will be zero.                       !
        #                                                                       !
        #  usage:    call getgases                                              !
        #                                                                       !
        #  subprograms called:  none                                            !
        #                                                                       !
        #  ===================================================================  !
        #

        gasdat = np.zeros((IMAX, LMAX, 10))

        #  --- ...  assign default values

        for k in range(LMAX):
            for i in range(IMAX):
                gasdat[i, k, 0] = self.co2vmr_def
                gasdat[i, k, 1] = self.n2ovmr_def
                gasdat[i, k, 2] = self.ch4vmr_def
                gasdat[i, k, 3] = self.o2vmr_def
                gasdat[i, k, 4] = self.covmr_def
                gasdat[i, k, 5] = self.f11vmr_def
                gasdat[i, k, 6] = self.f12vmr_def
                gasdat[i, k, 7] = self.f22vmr_def
                gasdat[i, k, 8] = self.cl4vmr_def
                gasdat[i, k, 9] = self.f113vmr_def

        #  --- ...  co2 section

        if self.ico2flg == 1:
            #  ---  use obs co2 global annual mean value only

            for k in range(LMAX):
                for i in range(IMAX):
                    gasdat[i, k, 0] = self.co2_glb + self.gco2cyc[self.kmonsav - 1]

        elif self.ico2flg == 2:
            #  ---  use obs co2 monthly data with 2-d variation at lower atmos
            #       otherwise use global mean value

            tmp = self.raddeg / self.resco2
            for i in range(IMAX):
                xlon1 = xlon[i]
                if xlon1 < 0.0:
                    xlon1 = xlon1 + con_pi  # if xlon in -pi->pi, convert to 0->2pi

                xlat1 = self.hfpi - xlat[i]  # if xlat in pi/2 -> -pi/2 range

                ilon = min(self.IMXCO2, int(xlon1 * tmp + 1)) - 1
                ilat = min(self.JMXCO2, int(xlat1 * tmp + 1)) - 1

                for k in range(LMAX):
                    if plvl[i, k + 1] >= self.prsco2:
                        gasdat[i, k, 0] = self.co2vmr_sav[ilon, ilat, self.kmonsav - 1]
                    else:
                        gasdat[i, k, 0] = self.co2_glb + self.gco2cyc[self.kmonsav - 1]

        return gasdat
