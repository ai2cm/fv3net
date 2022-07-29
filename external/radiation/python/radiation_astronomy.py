import numpy as np
import xarray as xr
import os
import sys
import warnings

sys.path.insert(0, "..")
from phys_const import con_pi, con_solr, con_solr_old
from radphysparam import solar_file
from config import *


class AstronomyClass:
    VTAGAST = "NCEP-Radiation_astronomy v5.2  Jan 2013 "

    degrad = 180.0 / con_pi
    tpi = 2.0 * con_pi
    hpi = 0.5 * con_pi
    f12 = 12.0
    f3600 = 3600.0
    czlimt = 0.0001  # ~ cos(89.99427)
    pid12 = con_pi / f12  # angle per hour

    def __init__(self, me, isolar):
        self.sollag = 0.0
        self.sindec = 0.0
        self.cosdec = 0.0
        self.anginc = 0.0
        self.smon_sav = con_solr * np.ones(12)
        self.iyr_sav = 0
        self.nstp = 6

        if me == 0:
            print(self.VTAGAST)  # print out version tag

        #  ---  initialization
        self.isolflg = isolar
        self.solc0 = con_solr
        self.solar_fname = solar_file
        self.iyr_sav = 0
        self.nstp = 6

        if isolar == 0:
            self.solc0 = con_solr_old
            if me == 0:
                print(f"- Using old fixed solar constant = {self.solc0}")
        elif isolar == 10:
            if me == 0:
                print(f"- Using new fixed solar constant = {self.solc0}")
        elif isolar == 1:  # noaa ann-mean tsi in absolute scale
            self.solar_fname = solar_file[:14] + "noaa_a0.nc" + solar_file[26:]

            if me == 0:
                print(
                    "- Using NOAA annual mean TSI table in ABS scale",
                    " with cycle approximation (old values)!",
                )

            file_exist = os.path.isfile(os.path.join(FORCING_DIR, self.solar_fname))
            if not file_exist:
                self.isolflg = 10

                if me == 0:
                    warnings.warn(
                        f'Requested solar data file "{self.solar_fname}" not found!',
                        f"Using the default solar constant value = {self.solc0}",
                        f" reset control flag isolflg={self.isolflg}",
                    )

        elif isolar == 2:  # noaa ann-mean tsi in tim scale
            self.solar_fname = solar_file[:14] + "noaa_an.nc" + solar_file[26:]

            if me == 0:
                print(
                    " - Using NOAA annual mean TSI table in TIM scale",
                    " with cycle approximation (new values)!",
                )

            file_exist = os.path.isfile(os.path.join(FORCING_DIR, self.solar_fname))
            if not file_exist:
                self.isolflg = 10

                if me == 0:
                    warnings.warn(
                        f'Requested solar data file "{self.solar_fname}" not found!',
                        f"Using the default solar constant value = {self.solc0}",
                        f" reset control flag isolflg={self.isolflg}",
                    )

        elif isolar == 3:  # cmip5 ann-mean tsi in tim scale
            self.solar_fname = solar_file[:14] + "cmip_an.nc" + solar_file[26:]

            if me == 0:
                print(
                    "- Using CMIP5 annual mean TSI table in TIM scale",
                    " with cycle approximation",
                )

            file_exist = os.path.isfile(os.path.join(FORCING_DIR, self.solar_fname))
            if not file_exist:
                self.isolflg = 10

                if me == 0:
                    warnings.warn(
                        f'Requested solar data file "{self.solar_fname}" not found!',
                        f"Using the default solar constant value = {self.solc0}",
                        f" reset control flag isolflg={self.isolflg}",
                    )

        elif isolar == 4:  # cmip5 mon-mean tsi in tim scale
            self.solar_fname = solar_file[:14] + "cmip_mn.nc" + solar_file[26:]

            if me == 0:
                print(
                    "- Using CMIP5 monthly mean TSI table in TIM scale",
                    " with cycle approximation",
                )

            file_exist = os.path.isfile(os.path.join(FORCING_DIR, self.solar_fname))
            if not file_exist:
                self.isolflg = 10

                if me == 0:
                    warnings.warn(
                        f'Requested solar data file "{self.solar_fname}" not found!',
                        f"Using the default solar constant value = {self.solc0}",
                        f" reset control flag isolflg={self.isolflg}",
                    )
        else:  # selection error
            self.isolflg = 10

            if me == 0:
                warnings.warn(
                    "- !!! ERROR in selection of solar constant data",
                    f" source, ISOL = {isolar}",
                )
                warnings.warn(
                    f"Using the default solar constant value = {self.solc0}",
                    f" reset control flag isolflg={self.isolflg}",
                )

    def return_initdata(self):
        outdict = {"solar_fname": self.solar_fname}
        return outdict

    def sol_update(self, jdate, kyear, deltsw, deltim, lsol_chg, me):
        #  ===================================================================  !
        #                                                                       !
        #  sol_update computes solar parameters at forecast time                !
        #                                                                       !
        #  inputs:                                                              !
        #     jdate(8)- ncep absolute date and time at fcst time                !
        #                (yr, mon, day, t-zone, hr, min, sec, mil-sec)          !
        #     kyear   - usually kyear=jdate(1). if not, it is for hindcast mode,!
        #               and it is usually the init cond time and serves as the  !
        #               upper limit of data can be used.                        !
        #     deltsw  - time duration in seconds per sw calculation             !
        #     deltim  - timestep in seconds                                     !
        #     lsol_chg- logical flags for change solar constant                 !
        #     me      - print message control flag                              !
        #                                                                       !
        #  outputs:                                                             !
        #    slag          - equation of time in radians                        !
        #    sdec, cdec    - sin and cos of the solar declination angle         !
        #    solcon        - sun-earth distance adjusted solar constant (w/m2)  !
        #                                                                       !
        #                                                                       !
        #  module variable:                                                     !
        #   solc0   - solar constant  (w/m**2) not adjusted by earth-sun dist   !
        #   isolflg - solar constant control flag                               !
        #             = 0: use the old fixed solar constant                     !
        #             =10: use the new fixed solar constant                     !
        #             = 1: use noaa ann-mean tsi tbl abs-scale with cycle apprx !
        #             = 2: use noaa ann-mean tsi tbl tim-scale with cycle apprx !
        #             = 3: use cmip5 ann-mean tsi tbl tim-scale with cycle apprx!
        #             = 4: use cmip5 mon-mean tsi tbl tim-scale with cycle apprx!
        #   solar_fname-external solar constant data table                      !
        #   sindec  - sine of the solar declination angle                       !
        #   cosdec  - cosine of the solar declination angle                     !
        #   anginc  - solar angle increment per iteration for cosz calc         !
        #   nstp    - total number of zenith angle iterations                   !
        #   smon_sav- saved monthly solar constants (isolflg=4 only)            !
        #   iyr_sav - saved year  of data previously used                       !
        #                                                                       !
        #  usage:    call sol_update                                            !
        #                                                                       !
        #  subprograms called:  solar, prtime                                   !
        #                                                                       !
        #  external functions called: iw3jdn                                    !
        #                                                                       !
        #  ===================================================================  !
        #

        #  ---  locals:
        hrday = 1.0 / 24.0  # frc day/hour
        minday = 1.0 / 1440.0  # frc day/minute
        secday = 1.0 / 86400.0  # frc day/second

        f3600 = 3600.0
        pid12 = con_pi / 12.0
        #
        # ===>  ...  begin here
        #
        #  --- ...  forecast time
        iyear = jdate[0]
        imon = jdate[1]
        iday = jdate[2]
        ihr = jdate[4]
        imin = jdate[5]
        isec = jdate[6]

        if lsol_chg:  # get solar constant from data table
            if self.iyr_sav == iyear:  # same year, no new reading necessary
                if self.isolflg == 4:
                    self.solc0 = self.smon_sav[imon]
            else:  # need to read in new data
                self.iyr_sav = iyear
                #  --- ...  check to see if the solar constant data file existed
                file_exist = os.path.isfile(os.path.join(FORCING_DIR, self.solar_fname))
                if not file_exist:
                    raise FileNotFoundError(
                        " !!! ERROR! Can not find solar constant file!!!"
                    )
                else:
                    iyr = iyear
                    ds = xr.open_dataset(os.path.join(FORCING_DIR, self.solar_fname))
                    iyr1 = ds["yr_start"].data
                    iyr2 = ds["yr_end"].data
                    icy1 = ds["yr_cyc1"].data
                    icy2 = ds["yr_cyc2"].data
                    smean = ds["smean"].data
                    if me == 0:
                        print("Updating solar constant with cycle approx")
                        print(f"Opened solar constant data file: {self.solar_fname}")
                    #  --- ...  check if there is a upper year limit put on the data table
                    if iyr < iyr1:
                        icy = (
                            icy1 - iyr1 + 1
                        )  # range of the earlest cycle in data table
                        while iyr < iyr1:
                            iyr += icy
                        if me == 0:
                            warnings.warn(
                                f"*** Year {iyear} out of table range!",
                                f"{iyr1}, {iyr2}",
                                f"Using the closest-cycle year ('{iyr}')",
                            )
                    elif iyr > iyr2:
                        icy = iyr2 - icy2 + 1  # range of the latest cycle in data table
                        while iyr > iyr2:
                            iyr -= icy
                        if me == 0:
                            warnings.warn(
                                f"*** Year {iyear} out of table range!",
                                f"{iyr1}, {iyr2}",
                                f"Using the closest-cycle year ('{iyr}')",
                            )
                    #  --- ...  locate the right record for the year of data
                    if self.isolflg < 4:  # use annual mean data tables
                        solc1 = ds["solc1"].sel(year=iyr).data
                        self.solc0 = smean + solc1
                        if me == 0:
                            print(
                                "CHECK: Solar constant data used for year",
                                f"{iyr}, {solc1}, {self.solc0}",
                            )
                    elif self.isolflg == 4:  # use monthly mean data tables
                        i = iyr2
                        while i >= iyr1:
                            jyr = ds["jyr"]
                            smon = ds["smon"]
                            if i == iyr and iyr == jyr:
                                for nn in range(12):
                                    self.smon_sav[nn] = smean + smon[nn]
                                self.solc0 = smean + smon[imon]
                                if me == 0:
                                    print("CHECK: Solar constant data used for year")
                                    print(f"{iyr} and month {imon}")

                                else:
                                    i -= 1
        else:
            self.solc0 = con_solr

        #  --- ...  calculate forecast julian day and fraction of julian day
        jd1 = self.iw3jdn(iyear, imon, iday)

        #  --- ...  unlike in normal applications, where day starts from 0 hr,
        #           in astronomy applications, day stats from noon.

        if ihr < 12:
            jd1 -= 1
            fjd1 = (
                0.5 + float(ihr) * hrday + float(imin) * minday + float(isec) * secday
            )
        else:
            fjd1 = float(ihr - 12) * hrday + float(imin) * minday + float(isec) * secday

        fjd1 += jd1

        jd = int(fjd1)
        fjd = fjd1 - jd

        # -# Call solar()
        r1, dlt, alp, sollag, sindec, cosdec = self.solar(jd, fjd)

        #  --- ...  calculate sun-earth distance adjustment factor appropriate to date
        self.solcon = self.solc0 / (r1 * r1)

        self.slag = sollag
        self.sdec = sindec
        self.cdec = cosdec
        self.sollag = sollag

        #  --- ...  diagnostic print out

        if me == 0:
            self.prtime(jd, fjd, dlt, alp, r1, self.solcon, sollag)

        #  --- ...  setting up calculation parameters used by subr coszmn

        nswr = max(1, np.round(deltsw / deltim))  # number of mdl t-step per sw call
        dtswh = deltsw / f3600  # time length in hours

        self.nstp = max(6, nswr)
        self.anginc = pid12 * dtswh / float(self.nstp)

        if me == 0:
            print(
                "for cosz calculations: nswr,deltim,deltsw,dtswh =",
                f"{nswr[0]}, {deltim[0]}, {deltsw[0]}, {dtswh[0]}, anginc, nstp =",
                f"{self.anginc[0]}, {self.nstp}",
            )

        return self.slag, self.sdec, self.cdec, self.solcon

    def prtime(self, jd, fjd, dlt, alp, r1, solc, sollag):
        #  ===================================================================  !
        #                                                                       !
        #  prtime prints out forecast date, time, and astronomy quantities.     !
        #                                                                       !
        #  inputs:                                                              !
        #    jd       - forecast julian day                                     !
        #    fjd      - forecast fraction of julian day                         !
        #    dlt      - declination angle of sun in radians                     !
        #    alp      - right ascension of sun in radians                       !
        #    r1       - earth-sun radius vector in meter                        !
        #    solc     - solar constant in w/m^2                                 !
        #                                                                       !
        #  outputs:   ( none )                                                  !
        #                                                                       !
        #  module variables:                                                    !
        #    sollag   - equation of time in radians                             !
        #                                                                       !
        #  usage:    call prtime                                                !
        #                                                                       !
        #  external subroutines called: w3fs26                                  !
        #                                                                       !
        #  ===================================================================  !
        #

        #  ---  locals:
        sixty = 60.0

        hpi = 0.5 * con_pi

        sign = "-"
        sigb = " "

        month = [
            "JAN.",
            "FEB.",
            "MAR.",
            "APR.",
            "MAY ",
            "JUNE",
            "JULY",
            "AUG.",
            "SEP.",
            "OCT.",
            "NOV ",
            "DEC.",
        ]

        # ===>  ...  begin here

        #  --- ...  get forecast hour and minute from fraction of julian day

        if fjd >= 0.5:
            jda = jd + 1
            mfjd = round(fjd * 1440.0)
            ihr = mfjd // 60 - 12
            xmin = float(mfjd) - (ihr + 12) * sixty
        else:
            jda = jd
            mfjd = round(fjd * 1440.0)
            ihr = mfjd // 60 + 12
            xmin = float(mfjd) - (ihr - 12) * sixty

        #  --- ...  get forecast year, month, and day from julian day

        iyear, imon, iday, idaywk, idayyr = self.w3fs26(jda)

        #  -- ...  compute solar parameters

        dltd = np.rad2deg(dlt)
        ltd = int(dltd)
        dltm = sixty * (np.abs(dltd) - abs(float(ltd)))
        ltm = int(dltm)
        dlts = sixty * (dltm - float(ltm))

        if (dltd < 0.0) and (ltd == 0.0):
            dsig = sign
        else:
            dsig = sigb

        halp = 6.0 * alp / hpi
        ihalp = int(halp)
        ymin = np.abs(halp - float(ihalp)) * sixty
        iyy = int(ymin)
        asec = (ymin - float(iyy)) * sixty

        eqt = 228.55735 * sollag
        eqsec = sixty * eqt

        print(
            f"0 FORECAST DATE {iday},{month[imon-1]},{iyear} AT {ihr} HRS, {xmin} MINS",
            f"  JULIAN DAY {jd} PLUS {fjd}",
        )

        print(f"  RADIUS VECTOR {r1}")
        print(
            f"  RIGHT ASCENSION OF SUN {halp} HRS, OR {ihalp} HRS {iyy} MINS {asec} SECS"
        )

        print(
            f"  DECLINATION OF THE SUN {dltd} DEGS, OR {dsig}",
            f"  {ltd} DEGS {ltm} MINS {dlts} SECS",
        )
        print(f"  EQUATION OF TIME {eqt} MINS, OR {eqsec} SECS, OR {sollag} RADIANS")
        print(f"  SOLAR CONSTANT {solc} (DISTANCE AJUSTED)")
        print(" ")
        print(" ")

    def solar(self, jd, fjd):
        #  ===================================================================  !
        #                                                                       !
        #  solar computes radius vector, declination and right ascension of     !
        #  sun, and equation of time.                                           !
        #                                                                       !
        #  inputs:                                                              !
        #    jd       - julian day                                              !
        #    fjd      - fraction of the julian day                              !
        #                                                                       !
        #  outputs:                                                             !
        #    r1       - earth-sun radius vector                                 !
        #    dlt      - declination of sun in radians                           !
        #    alp      - right ascension of sun in radians                       !
        #                                                                       !
        #  module variables:                                                    !
        #    sollag   - equation of time in radians                             !
        #    sindec   - sine of declination angle                               !
        #    cosdec   - cosine of declination angle                             !
        #                                                                       !
        #  usage:    call solar                                                 !
        #                                                                       !
        #  external subroutines called: none                                    !
        #                                                                       !
        #  ===================================================================  !
        #

        #  ---  locals:
        cyear = 365.25  # days of year
        ccr = 1.3e-6  # iteration limit
        tpp = 1.55  # days between epoch and
        svt6 = 78.035  # days between perihelion passage
        jdor = 2415020  # jd of epoch which is january

        tpi = 2.0 * con_pi

        # ===>  ...  begin here

        # --- ...  computes time in julian centuries after epoch

        t1 = float(jd - jdor) / 36525.0

        # --- ...  computes length of anomalistic and tropical years (minus 365 days)

        year = 0.25964134 + 0.304e-5 * t1
        tyear = 0.24219879 - 0.614e-5 * t1

        # --- ...  computes orbit eccentricity and angle of earth's inclination from t

        ec = 0.01675104 - (0.418e-4 + 0.126e-6 * t1) * t1
        angin = 23.452294 - (0.0130125 + 0.164e-5 * t1) * t1

        ador = jdor
        jdoe = int(ador + (svt6 * cyear) / (year - tyear))

        # --- ...  deleqn is updated svt6 for current date

        deleqn = float(jdoe - jd) * (year - tyear) / cyear
        year = year + 365.0
        sni = np.sin(np.deg2rad(angin))
        tini = 1.0 / np.tan(np.deg2rad(angin))
        er = np.sqrt((1.0 + ec) / (1.0 - ec))
        qq = deleqn * tpi / year

        # --- ...  determine true anomaly at equinox

        e1 = 1.0
        cd = 1.0
        iter = 0

        while cd > ccr:
            ep = e1 - (e1 - ec * np.sin(e1) - qq) / (1.0 - ec * np.cos(e1))
            cd = np.abs(e1 - ep)
            e1 = ep
            iter += 1

            if iter > 10:
                print(f"ITERATION COUNT FOR LOOP 32 = {iter}")
                print(f"E, EP, CD = {e1}, {ep}, {cd}")
                break

        eq = 2.0 * np.arctan(er * np.tan(0.5 * e1))

        # --- ...  date is days since last perihelion passage

        dat = float(jd - jdor) - tpp + fjd
        date = dat % year

        # --- ...  solve orbit equations by newton's method

        em = tpi * date / year
        e1 = 1.0
        cr = 1.0
        iter = 0

        while cr > ccr:
            ep = e1 - (e1 - ec * np.sin(e1) - em) / (1.0 - ec * np.cos(e1))
            cr = np.abs(e1 - ep)
            e1 = ep
            iter += 1

            if iter > 10:
                print(f"ITERATION COUNT FOR LOOP 31 = {iter}")
                break

        w1 = 2.0 * np.arctan(er * np.tan(0.5 * e1))

        r1 = 1.0 - ec * np.cos(e1)

        sindec = sni * np.sin(w1 - eq)
        cosdec = np.sqrt(1.0 - sindec * sindec)

        dlt = np.arcsin(sindec)
        alp = np.arcsin(np.tan(dlt) * tini)

        tst = np.cos(w1 - eq)
        if tst < 0.0:
            alp = con_pi - alp
        if alp < 0.0:
            alp = alp + tpi

        sun = tpi * (date - deleqn) / year
        if sun < 0.0:
            sun = sun + tpi
        sollag = sun - alp - 0.03255

        return r1, dlt, alp, sollag, sindec, cosdec

    def integer_divide_towards_zero(self, a, b):
        return -(-a // b) if a < 0 else a // b

    def iw3jdn(self, iyear, month, iday):

        iw3jdn = (
            iday
            - 32075
            + self.integer_divide_towards_zero(
                1461
                * (iyear + 4800 + self.integer_divide_towards_zero((month - 14), 12)),
                4,
            )
            + self.integer_divide_towards_zero(
                367
                * (month - 2 - self.integer_divide_towards_zero((month - 14), 12) * 12),
                12,
            )
            - self.integer_divide_towards_zero(
                3
                * (
                    self.integer_divide_towards_zero(
                        (
                            iyear
                            + 4900
                            + self.integer_divide_towards_zero((month - 14), 12)
                        ),
                        100,
                    )
                ),
                4,
            )
        )

        return int(iw3jdn)

    def w3fs26(self, JLDAYN):
        L = JLDAYN + 68569
        N = int(4 * L // 146097)
        L = int(L - (146097 * N + 3) // 4)
        I = int(4000 * (L + 1) // 1461001)
        L = int(L - 1461 * I // 4 + 31)
        J = int(80 * L // 2447)
        IDAY = int(L - 2447 * J // 80)
        L = int(J // 11)
        MONTH = int(J + 2 - 12 * L)
        IYEAR = int(100 * (N - 49) + I + L)
        IDAYWK = int(((JLDAYN + 1) % 7) + 1)
        IDAYYR = int(
            JLDAYN
            - (-31739 + 1461 * (IYEAR + 4799) // 4 - 3 * ((IYEAR + 4899) // 100) // 4)
        )

        return IYEAR, MONTH, IDAY, IDAYWK, IDAYYR

    def coszmn(self, xlon, sinlat, coslat, solhr, IM, me):
        #  ===================================================================  !
        #                                                                       !
        #  coszmn computes mean cos solar zenith angle over sw calling interval !
        #                                                                       !
        #  inputs:                                                              !
        #    xlon  (IM)    - grids' longitudes in radians, work both on zonal   !
        #                    0->2pi and -pi->+pi arrangements                   !
        #    sinlat(IM)    - sine of the corresponding latitudes                !
        #    coslat(IM)    - cosine of the corresponding latitudes              !
        #    solhr         - time after 00z in hours                            !
        #    IM            - num of grids in horizontal dimension               !
        #    me            - print message control flag                         !
        #                                                                       !
        #  outputs:                                                             !
        #    coszen(IM)    - average of cosz for daytime only in sw call interval
        #    coszdg(IM)    - average of cosz over entire sw call interval       !
        #                                                                       !
        #  module variables:                                                    !
        #    sollag        - equation of time                                   !
        #    sindec        - sine of the solar declination angle                !
        #    cosdec        - cosine of the solar declination angle              !
        #    anginc        - solar angle increment per iteration for cosz calc  !
        #    nstp          - total number of zenith angle iterations            !
        #                                                                       !
        #  usage:    call comzmn                                                !
        #                                                                       !
        #  external subroutines called: none                                    !
        #                                                                       !
        #  ===================================================================  !
        #

        coszen = np.zeros(IM)
        coszdg = np.zeros(IM)
        istsun = np.zeros(IM)

        solang = self.pid12 * (solhr - 12.0)  # solar angle at present time
        rstp = 1.0 / float(self.nstp)

        for it in range(self.nstp):
            cns = solang + (it + 0.5) * self.anginc + self.sollag
            for i in range(IM):
                coszn = self.sdec * sinlat[i] + self.cdec * coslat[i] * np.cos(
                    cns + xlon[i]
                )
                coszen[i] = coszen[i] + max(0.0, coszn)
                if coszn > self.czlimt:
                    istsun[i] += 1

        #  --- ...  compute time averages

        for i in range(IM):
            coszdg[i] = coszen[i] * rstp
            if istsun[i] > 0:
                coszen[i] = coszen[i] / istsun[i]

        return coszen, coszdg
