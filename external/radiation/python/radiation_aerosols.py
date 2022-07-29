from multiprocessing import Value
import numpy as np
import xarray as xr
import os
import sys

sys.path.insert(0, "..")
from radlw.radlw_param import NBDLW, wvnlw1, wvnlw2
from radsw.radsw_param import nbdsw, wvnum1, wvnum2, NSWSTR
from phys_const import con_pi, con_plnk, con_c, con_boltz, con_t0c, con_rd, con_g
from radphysparam import aeros_file, lalw1bd

from config import *


class AerosolClass:
    VTAGAER = "NCEP-Radiation_aerosols  v5.2  Jan 2013 "

    NF_AESW = 3
    NF_AELW = 3
    NLWSTR = 1
    NSPC = 5
    NSPC1 = NSPC + 1

    NWVSOL = 151
    NWVTOT = 57600
    NWVTIR = 4000

    nwvns0 = [
        100,
        11,
        14,
        18,
        24,
        33,
        50,
        83,
        12,
        12,
        13,
        15,
        15,
        17,
        18,
        20,
        21,
        24,
        26,
        30,
        32,
        37,
        42,
        47,
        55,
        64,
        76,
        91,
        111,
        139,
        179,
        238,
        333,
        41,
        42,
        45,
        46,
        48,
        51,
        53,
        55,
        58,
        61,
        64,
        68,
        71,
        75,
        79,
        84,
        89,
        95,
        101,
        107,
        115,
        123,
        133,
        142,
        154,
        167,
        181,
        197,
        217,
        238,
        263,
        293,
        326,
        368,
        417,
        476,
        549,
        641,
        758,
        909,
        101,
        103,
        105,
        108,
        109,
        112,
        115,
        117,
        119,
        122,
        125,
        128,
        130,
        134,
        137,
        140,
        143,
        147,
        151,
        154,
        158,
        163,
        166,
        171,
        175,
        181,
        185,
        190,
        196,
        201,
        207,
        213,
        219,
        227,
        233,
        240,
        248,
        256,
        264,
        274,
        282,
        292,
        303,
        313,
        325,
        337,
        349,
        363,
        377,
        392,
        408,
        425,
        444,
        462,
        483,
        505,
        529,
        554,
        580,
        610,
        641,
        675,
        711,
        751,
        793,
        841,
        891,
        947,
        1008,
        1075,
        1150,
        1231,
        1323,
        1425,
        1538,
        1667,
        1633,
        14300,
    ]

    s0intv = [
        1.60000e-6,
        2.88000e-5,
        3.60000e-5,
        4.59200e-5,
        6.13200e-5,
        8.55000e-5,
        1.28600e-4,
        2.16000e-4,
        2.90580e-4,
        3.10184e-4,
        3.34152e-4,
        3.58722e-4,
        3.88050e-4,
        4.20000e-4,
        4.57056e-4,
        4.96892e-4,
        5.45160e-4,
        6.00600e-4,
        6.53600e-4,
        7.25040e-4,
        7.98660e-4,
        9.11200e-4,
        1.03680e-3,
        1.18440e-3,
        1.36682e-3,
        1.57560e-3,
        1.87440e-3,
        2.25500e-3,
        2.74500e-3,
        3.39840e-3,
        4.34000e-3,
        5.75400e-3,
        7.74000e-3,
        9.53050e-3,
        9.90192e-3,
        1.02874e-2,
        1.06803e-2,
        1.11366e-2,
        1.15830e-2,
        1.21088e-2,
        1.26420e-2,
        1.32250e-2,
        1.38088e-2,
        1.44612e-2,
        1.51164e-2,
        1.58878e-2,
        1.66500e-2,
        1.75140e-2,
        1.84450e-2,
        1.94106e-2,
        2.04864e-2,
        2.17248e-2,
        2.30640e-2,
        2.44470e-2,
        2.59840e-2,
        2.75940e-2,
        2.94138e-2,
        3.13950e-2,
        3.34800e-2,
        3.57696e-2,
        3.84054e-2,
        4.13490e-2,
        4.46880e-2,
        4.82220e-2,
        5.22918e-2,
        5.70078e-2,
        6.19888e-2,
        6.54720e-2,
        6.69060e-2,
        6.81226e-2,
        6.97788e-2,
        7.12668e-2,
        7.27100e-2,
        7.31610e-2,
        7.33471e-2,
        7.34814e-2,
        7.34717e-2,
        7.35072e-2,
        7.34939e-2,
        7.35202e-2,
        7.33249e-2,
        7.31713e-2,
        7.35462e-2,
        7.36920e-2,
        7.23677e-2,
        7.25023e-2,
        7.24258e-2,
        7.20766e-2,
        7.18284e-2,
        7.32757e-2,
        7.31645e-2,
        7.33277e-2,
        7.36128e-2,
        7.33752e-2,
        7.28965e-2,
        7.24924e-2,
        7.23307e-2,
        7.21050e-2,
        7.12620e-2,
        7.10903e-2,
        7.12714e-2,
        7.08012e-2,
        7.03752e-2,
        7.00350e-2,
        6.98639e-2,
        6.90690e-2,
        6.87621e-2,
        6.52080e-2,
        6.65184e-2,
        6.60038e-2,
        6.47615e-2,
        6.44831e-2,
        6.37206e-2,
        6.24102e-2,
        6.18698e-2,
        6.06320e-2,
        5.83498e-2,
        5.67028e-2,
        5.51232e-2,
        5.48645e-2,
        5.12340e-2,
        4.85581e-2,
        4.85010e-2,
        4.79220e-2,
        4.44058e-2,
        4.48718e-2,
        4.29373e-2,
        4.15242e-2,
        3.81744e-2,
        3.16342e-2,
        2.99615e-2,
        2.92740e-2,
        2.67484e-2,
        1.76904e-2,
        1.40049e-2,
        1.46224e-2,
        1.39993e-2,
        1.19574e-2,
        1.06386e-2,
        1.00980e-2,
        8.63808e-3,
        6.52736e-3,
        4.99410e-3,
        4.39350e-3,
        2.21676e-3,
        1.33812e-3,
        1.12320e-3,
        5.59000e-4,
        3.60000e-4,
        2.98080e-4,
        7.46294e-5,
    ]

    MINVYR = 1850
    MAXVYR = 1999
    NXC = 5
    NAE = 7
    NDM = 5
    IMXAE = 72
    JMXAE = 37
    NAERBND = 61
    NRHLEV = 8
    NCM1 = 6
    NCM2 = 4
    NCM = NCM1 + NCM2

    rhlev = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

    idxspc = [1, 2, 1, 1, 1, 1, 3, 5, 5, 4]

    KAERBND = 61
    KRHLEV = 36

    wvn550 = 1.0e4 / 0.55

    def __init__(self, NLAY, me, iaerflg, ivflip):
        self.NSWBND = nbdsw
        self.NLWBND = NBDLW
        self.NSWLWBD = nbdsw * NBDLW
        self.lalwflg = True
        self.laswflg = True
        self.lavoflg = True
        self.lmap_new = True
        self.NLAY = NLAY
        self.ivflip = ivflip

        self.kyrstr = 1
        self.kyrend = 1
        self.kyrsav = 1
        self.kmonsav = 1

        self.haer = np.zeros((self.NDM, self.NAE))
        self.prsref = np.zeros((self.NDM, self.NAE))
        self.sigref = np.zeros((self.NDM, self.NAE))

        self.cmixg = np.zeros((self.NXC, self.IMXAE, self.JMXAE))
        self.denng = np.zeros((2, self.IMXAE, self.JMXAE))
        self.idxcg = np.zeros((self.NXC, self.IMXAE, self.JMXAE))
        self.kprfg = np.zeros((self.IMXAE, self.JMXAE))

        self.aeros_file = os.path.join(FORCING_DIR, aeros_file)

        self.iaerflg = iaerflg
        self.iaermdl = int(self.iaerflg / 1000)
        if self.iaermdl < 0 or self.iaermdl > 2 and self.iaermdl != 5:
            raise ValueError("Error -- IAER flag is incorrect, Abort")

        self.laswflg = self.iaerflg % 10 > 0  # control flag for sw tropospheric aerosol
        self.lalwflg = (
            self.iaerflg / 10 % 10 > 0
        )  # control flag for lw tropospheric aerosol
        self.lavoflg = (
            self.iaerflg >= 100
        )  # control flag for stratospheric volcanic aeros

        # -# Call wrt_aerlog() to write aerosol parameter configuration to output logs.

        if me == 0:
            self.wrt_aerlog()  # write aerosol param info to log file

        if self.iaerflg == 0:
            return  # return without any aerosol calculations

            #  --- ...  in sw, aerosols optical properties are computed for each radiation
            #           spectral band; while in lw, optical properties can be calculated
            #           for either only one broad band or for each of the lw radiation bands

        if self.laswflg:
            self.NSWBND = nbdsw
        else:
            self.NSWBND = 0

        if self.lalwflg:
            if lalw1bd:
                self.NLWBND = 1
            else:
                self.NLWBND = NBDLW
        else:
            self.NLWBND = 0

        self.NSWLWBD = self.NSWBND + self.NLWBND

        self.wvn_sw1 = wvnum1
        self.wvn_sw2 = wvnum2
        self.wvn_lw1 = wvnlw1
        self.wvn_lw2 = wvnlw2

        # note: for result consistency, the defalt opac-clim aeros setting still use
        #       old spectral band mapping. use iaermdl=5 to use new mapping method

        if self.iaermdl == 0:  # opac-climatology scheme
            self.lmap_new = False

            self.wvn_sw1[1 : nbdsw - 1] = self.wvn_sw1[1 : nbdsw - 1] + 1
            self.wvn_lw1[1:NBDLW] = self.wvn_lw1[1:NBDLW] + 1
        else:
            self.lmap_new = True

        if self.iaerflg != 100:

            # -# Call set_spectrum() to set up spectral one wavenumber solar/IR
            # fluxes.

            self.set_spectrum()

            # -# Call clim_aerinit() to invoke tropospheric aerosol initialization.

            if self.iaermdl == 0 or self.iaermdl == 5:  # opac-climatology scheme

                self.clim_aerinit()

            else:
                raise ValueError(
                    "!!! ERROR in aerosol model scheme selection",
                    f" iaermdl = {self.iaermdl}",
                )

        # -# Call set_volcaer() to invoke stratospheric volcanic aerosol
        # initialization.

        if self.lavoflg:
            self.ivolae = np.zeros((12, 4, 10))

    def return_initdata(self):
        outdict = {
            "extrhi": self.extrhi,
            "scarhi": self.scarhi,
            "ssarhi": self.ssarhi,
            "asyrhi": self.asyrhi,
            "extrhd": self.extrhd,
            "scarhd": self.scarhd,
            "ssarhd": self.ssarhd,
            "asyrhd": self.asyrhd,
            "extstra": self.extstra,
            "prsref": self.prsref,
            "haer": self.haer,
            "eirfwv": self.eirfwv,
            "solfwv": self.solfwv,
        }
        return outdict

    def return_updatedata(self):
        outdict = {
            "kprfg": self.kprfg,
            "idxcg": self.idxcg,
            "cmixg": self.cmixg,
            "denng": self.denng,
            "ivolae": self.ivolae,
        }
        return outdict

    def wrt_aerlog(self):
        #  ==================================================================  !
        #                                                                      !
        #  subprogram : wrt_aerlog                                             !
        #                                                                      !
        #    write aerosol parameter configuration to run log file.            !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        #  external module variables:  (in physparam)                          !
        #   iaermdl  - aerosol scheme flag: 0:opac-clm; 1:gocart-clim;         !
        #              2:gocart-prog; 5:opac-clim+new mapping                  !
        #   iaerflg  - aerosol effect control flag: 3-digits (volc,lw,sw)      !
        #   lalwflg  - toposphere lw aerosol effect: =f:no; =t:yes             !
        #   laswflg  - toposphere sw aerosol effect: =f:no; =t:yes             !
        #   lavoflg  - stratospherer volcanic aeros effect: =f:no; =t:yes      !
        #                                                                      !
        #  outputs: ( none )                                                   !
        #                                                                      !
        #  subroutines called: none                                            !
        #                                                                      !
        #  usage:    call wrt_aerlog                                           !
        #                                                                      !
        #  ==================================================================  !

        print(self.VTAGAER)  # print out version tag

        if self.iaermdl == 0 or self.iaermdl == 5:
            print(
                "- Using OPAC-seasonal climatology for tropospheric", " aerosol effect"
            )
        elif self.iaermdl == 1:
            print("- Using GOCART-climatology for tropospheric", " aerosol effect")
            raise NotImplementedError("GOCART climatology not yet implemented")
        elif self.iaermdl == 2:
            print(
                " - Using GOCART-prognostic aerosols for tropospheric",
                " aerosol effect",
            )
            raise NotImplementedError("GOCART climatology not yet implemented")
        else:
            raise ValueError(
                "!!! ERROR in selection of aerosol model scheme",
                f" IAER_MDL = {self.iaermdl}",
            )

        print(
            f"IAER={self.iaerflg},  LW-trop-aer={self.lalwflg}",
            f"SW-trop-aer={self.laswflg}, Volc-aer={self.lavoflg}",
        )

        if self.iaerflg <= 0:  # turn off all aerosol effects
            print("- No tropospheric/volcanic aerosol effect included")
            print(
                "Input values of aerosol optical properties to",
                " both SW and LW radiations are set to zeros",
            )
        else:
            if self.iaerflg >= 100:  # incl stratospheric volcanic aerosols
                print("- Include stratospheric volcanic aerosol effect")
            else:  # no stratospheric volcanic aerosols
                print("- No stratospheric volcanic aerosol effect")

            if self.laswflg:  # chcek for sw effect
                print(
                    "- Compute multi-band aerosol optical",
                    " properties for SW input parameters",
                )
            else:
                print(
                    "- No SW radiation aerosol effect, values of",
                    " aerosol properties to SW input are set to zeros",
                )

            if self.lalwflg:  # check for lw effect
                if lalw1bd:
                    print(
                        "- Compute 1 broad-band aerosol optical",
                        " properties for LW input parameters",
                    )
                else:
                    print(
                        "- Compute multi-band aerosol optical",
                        " properties for LW input parameters",
                    )
            else:
                print(
                    "- No LW radiation aerosol effect, values of",
                    " aerosol properties to LW input are set to zeros",
                )

    # This subroutine defines the one wavenumber solar fluxes based on toa
    # solar spectral distribution, and define the one wavenumber IR fluxes
    # based on black-body emission distribution at a predefined temperature.
    # \section gel_set_spec General Algorithm

    def set_spectrum(self):
        #  ==================================================================  !
        #                                                                      !
        #  subprogram : set_spectrum                                           !
        #                                                                      !
        #    define the one wavenumber solar fluxes based on toa solar spectral!
        #    distrobution, and define the one wavenumber ir fluxes based on    !
        #    black-body emission distribution at a predefined temperature.     !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        # > -  inputs:  (module constants)
        #!  -   NWVTOT:  total num of wave numbers used in sw spectrum
        #!  -   NWVTIR:  total num of wave numbers used in the ir region
        #!
        # > -  outputs: (in-scope variables)
        #!  -   solfwv(NWVTOT):   solar flux for each individual wavenumber
        #!                        (\f$W/m^2\f$)
        #!  -   eirfwv(NWVTIR):   ir flux(273k) for each individual wavenumber
        #!                        (\f$W/m^2\f$)
        #                                                                      !
        #  subroutines called: none                                            !
        #                                                                      !
        #  usage:    call set_spectrum                                         !
        #                                                                      !
        #  ==================================================================  !

        self.solfwv = np.zeros(self.NWVTOT)

        for nb in range(self.NWVSOL):
            if nb == 0:
                nw1 = 1
            else:
                nw1 = nw1 + self.nwvns0[nb - 1]

            nw2 = nw1 + self.nwvns0[nb] - 1

            for nw in range(nw1 - 1, nw2):
                self.solfwv[nw] = self.s0intv[nb]

        #  --- ...  define the one wavenumber ir fluxes based on black-body
        #           emission distribution at a predefined temperature

        tmp1 = (con_pi + con_pi) * con_plnk * con_c * con_c
        tmp2 = con_plnk * con_c / (con_boltz * con_t0c)

        self.eirfwv = np.zeros(self.NWVTIR)

        for nw in range(self.NWVTIR):
            tmp3 = 100.0 * (nw + 1)
            self.eirfwv[nw] = (tmp1 * tmp3 ** 3) / (np.exp(tmp2 * tmp3) - 1.0)

    def clim_aerinit(self):
        #  ==================================================================  !
        #                                                                      !
        #  clim_aerinit is the opac-climatology aerosol initialization program !
        #  to set up necessary parameters and working arrays.                  !
        #                                                                      !
        #  inputs:                                                             !
        #   solfwv(NWVTOT)   - solar flux for each individual wavenumber (w/m2)!
        #   eirfwv(NWVTIR)   - ir flux(273k) for each individual wavenum (w/m2)!
        #   me               - print message control flag                      !
        #                                                                      !
        #  outputs: (to module variables)                                      !
        #                                                                      !
        #  external module variables: (in physparam)                           !
        #     iaerflg - abc 3-digit integer aerosol flag (abc:volc,lw,sw)      !
        #               a: =0 use background stratospheric aerosol             !
        #                  =1 incl stratospheric vocanic aeros (MINVYR-MAXVYR) !
        #               b: =0 no topospheric aerosol in lw radiation           !
        #                  =1 include tropspheric aerosols for lw radiation    !
        #               c: =0 no topospheric aerosol in sw radiation           !
        #                  =1 include tropspheric aerosols for sw radiation    !
        #     lalwflg - logical lw aerosols effect control flag                !
        #               =t compute lw aerosol optical prop                     !
        #     laswflg - logical sw aerosols effect control flag                !
        #               =t compute sw aerosol optical prop                     !
        #     lalw1bd = logical lw aeros propty 1 band vs multi-band cntl flag !
        #               =t use 1 broad band optical property                   !
        #               =f use multi bands optical property                    !
        #                                                                      !
        #  module constants:                                                   !
        #     NWVSOL  - num of wvnum regions where solar flux is constant      !
        #     NWVTOT  - total num of wave numbers used in sw spectrum          !
        #     NWVTIR  - total num of wave numbers used in the ir region        !
        #     NSWBND  - total number of sw spectral bands                      !
        #     NLWBND  - total number of lw spectral bands                      !
        #     NAERBND - number of bands for climatology aerosol data           !
        #     NCM1    - number of rh independent aeros species                 !
        #     NCM2    - number of rh dependent aeros species                   !
        #                                                                      !
        #  usage:    call clim_aerinit                                         !
        #                                                                      !
        #  subprograms called:  set_aercoef, optavg                            !
        #                                                                      !
        #  ==================================================================  !#

        #  --- ...  invoke tropospheric aerosol initialization

        # - call set_aercoef() to invoke tropospheric aerosol initialization.
        self.set_aercoef()

        # The initialization program for climatological aerosols. The program
        # reads and maps the pre-tabulated aerosol optical spectral data onto
        # corresponding SW radiation spectral bands.
        # \section det_set_aercoef General Algorithm
        # @{

    def set_aercoef(self):
        #  ==================================================================  !
        #                                                                      !
        #  subprogram : set_aercoef                                            !
        #                                                                      !
        #    this is the initialization progrmam for climatological aerosols   !
        #                                                                      !
        #    the program reads and maps the pre-tabulated aerosol optical      !
        #    spectral data onto corresponding sw radiation spectral bands.     !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        #  inputs:  (in-scope variables, module constants)                     !
        #   solfwv(:)    - real, solar flux for individual wavenumber (w/m2)   !
        #   eirfwv(:)    - real, lw flux(273k) for individual wavenum (w/m2)   !
        #   me           - integer, select cpu number as print control flag    !
        #                                                                      !
        #  outputs: (to the module variables)                                  !
        #                                                                      !
        #  external module variables:  (in physparam)                          !
        #   lalwflg   - module control flag for lw trop-aer: =f:no; =t:yes     !
        #   laswflg   - module control flag for sw trop-aer: =f:no; =t:yes     !
        #   aeros_file- external aerosol data file name                        !
        #                                                                      !
        #  internal module variables:                                          !
        #     IMXAE   - number of longitude points in global aeros data set    !
        #     JMXAE   - number of latitude points in global aeros data set     !
        #     wvnsw1,wvnsw2 (NSWSTR:NSWEND)                                    !
        #             - start/end wavenumbers for each of sw bands             !
        #     wvnlw1,wvnlw2 (     1:NBDLW)                                     !
        #             - start/end wavenumbers for each of lw bands             !
        #     NSWLWBD - total num of bands (sw+lw) for aeros optical properties!
        #     NSWBND  - number of sw spectral bands actually invloved          !
        #     NLWBND  - number of lw spectral bands actually invloved          !
        #     NIAERCM - unit number for reading input data set                 !
        #     extrhi  - extinction coef for rh-indep aeros         NCM1*NSWLWBD!
        #     scarhi  - scattering coef for rh-indep aeros         NCM1*NSWLWBD!
        #     ssarhi  - single-scat-alb for rh-indep aeros         NCM1*NSWLWBD!
        #     asyrhi  - asymmetry factor for rh-indep aeros        NCM1*NSWLWBD!
        #     extrhd  - extinction coef for rh-dep aeros    NRHLEV*NCM2*NSWLWBD!
        #     scarhd  - scattering coef for rh-dep aeros    NRHLEV*NCM2*NSWLWBD!
        #     ssarhd  - single-scat-alb for rh-dep aeros    NRHLEV*NCM2*NSWLWBD!
        #     asyrhd  - asymmetry factor for rh-dep aeros   NRHLEV*NCM2*NSWLWBD!
        #                                                                      !
        #  major local variables:                                              !
        #   for handling spectral band structures                              !
        #     iendwv   - ending wvnum (cm**-1) for each band  NAERBND          !
        #   for handling optical properties of rh independent species (NCM1)   !
        #         1. insoluble        (inso); 2. soot             (soot);      !
        #         3. mineral nuc mode (minm); 4. mineral acc mode (miam);      !
        #         5. mineral coa mode (micm); 6. mineral transport(mitr).      !
        #     rhidext0 - extinction coefficient             NAERBND*NCM1       !
        #     rhidsca0 - scattering coefficient             NAERBND*NCM1       !
        #     rhidssa0 - single scattering albedo           NAERBND*NCM1       !
        #     rhidasy0 - asymmetry parameter                NAERBND*NCM1       !
        #   for handling optical properties of rh ndependent species (NCM2)    !
        #         1. water soluble    (waso); 2. sea salt acc mode(ssam);      !
        #         3. sea salt coa mode(sscm); 4. sulfate droplets (suso).      !
        #         rh level (NRHLEV): 00%, 50%, 70%, 80%, 90%, 95%, 98%, 99%    !
        #     rhdpext0 - extinction coefficient             NAERBND,NRHLEV,NCM2!
        #     rhdpsca0 - scattering coefficient             NAERBND,NRHLEV,NCM2!
        #     rhdpssa0 - single scattering albedo           NAERBND,NRHLEV,NCM2!
        #     rhdpasy0 - asymmetry parameter                NAERBND,NRHLEV,NCM2!
        #   for handling optical properties of stratospheric bkgrnd aerosols   !
        #     straext0 - extingction coefficients             NAERBND          !
        #                                                                      !
        #  usage:    call set_aercoef                                          !
        #                                                                      !
        #  subprograms called:  optavg                                         !
        #                                                                      !
        #  ==================================================================  !

        file_exist = os.path.isfile(self.aeros_file)

        if file_exist:
            print(f"Using file {aeros_file}")
        else:
            raise FileNotFoundError(
                f'Requested aerosol data file "{aeros_file}" not found!',
                "*** Stopped in subroutine aero_init !!",
            )

        extrhi = np.zeros((self.NCM1, self.NSWLWBD))
        scarhi = np.zeros((self.NCM1, self.NSWLWBD))
        ssarhi = np.zeros((self.NCM1, self.NSWLWBD))
        asyrhi = np.zeros((self.NCM1, self.NSWLWBD))

        extrhd = np.zeros((self.NRHLEV, self.NCM2, self.NSWLWBD))
        scarhd = np.zeros((self.NRHLEV, self.NCM2, self.NSWLWBD))
        ssarhd = np.zeros((self.NRHLEV, self.NCM2, self.NSWLWBD))
        asyrhd = np.zeros((self.NRHLEV, self.NCM2, self.NSWLWBD))

        self.extstra = np.zeros((self.NSWLWBD))

        #  --- ...  aloocate and input aerosol optical data
        ds = xr.open_dataset(self.aeros_file)

        iendwv = ds["iendwv"].data
        haer = ds["haer"].data
        prsref = ds["prsref"].data
        rhidext0 = ds["rhidext0"].data
        rhidsca0 = ds["rhidsca0"].data
        rhidssa0 = ds["rhidssa0"].data
        rhidasy0 = ds["rhidasy0"].data
        rhdpext0 = ds["rhdpext0"].data
        rhdpsca0 = ds["rhdpsca0"].data
        rhdpssa0 = ds["rhdpssa0"].data
        rhdpasy0 = ds["rhdpasy0"].data
        straext0 = ds["straext0"].data

        # -# Convert pressure reference level (in mb) to sigma reference level
        #    assume an 1000mb reference surface pressure.

        self.sigref = 0.001 * prsref

        # -# Compute solar flux weights and interval indices for mapping
        #    spectral bands between SW radiation and aerosol data.

        self.nv1 = np.zeros(self.NSWBND, dtype=np.int32)
        self.nv2 = np.zeros(self.NSWBND, dtype=np.int32)

        if self.laswflg:
            self.solbnd = np.zeros(self.NSWBND)
            self.solwaer = np.zeros((self.NSWBND, self.NAERBND))

            ibs = 1
            ibe = 1
            wvs = self.wvn_sw1[0]
            wve = self.wvn_sw1[0]
            self.nv_aod = 1
            for ib in range(1, self.NSWBND):
                mb = ib + NSWSTR - 1
                if self.wvn_sw2[mb] >= self.wvn550 and self.wvn550 >= self.wvn_sw1[mb]:
                    self.nv_aod = ib + 1  # sw band number covering 550nm wavelenth

                if self.wvn_sw1[mb] < wvs:
                    wvs = self.wvn_sw1[mb]
                    ibs = ib
                if self.wvn_sw1[mb] > wve:
                    wve = self.wvn_sw1[mb]
                    ibe = ib

            #!$o    mp parallel do private(ib,mb,ii,iw1,iw2,iw,sumsol,fac,tmp,ibs,ibe)
            for ib in range(self.NSWBND):
                mb = ib + NSWSTR - 1
                ii = 0
                iw1 = round(self.wvn_sw1[mb])
                iw2 = round(self.wvn_sw2[mb])

                while iw1 > iendwv[ii]:
                    if ii == self.NAERBND - 1:
                        break
                    ii += 1

                if self.lmap_new:
                    if ib == ibs:
                        sumsol = 0.0
                    else:
                        sumsol = -0.5 * self.solfwv[iw1 - 1]

                    if ib == ibe:
                        fac = 0.0
                    else:
                        fac = -0.5
                    self.solbnd[ib] = sumsol
                else:
                    sumsol = 0.0

                self.nv1[ib] = ii

                for iw in range(iw1 - 1, iw2):
                    self.solbnd[ib] += self.solfwv[iw]
                    sumsol = sumsol + self.solfwv[iw]

                    if iw == iendwv[ii] - 1:
                        self.solwaer[ib, ii] = sumsol

                        if ii < self.NAERBND - 1:
                            sumsol = 0.0
                            ii += 1

                if iw2 != iendwv[ii] - 1:
                    self.solwaer[ib, ii] = sumsol

                if self.lmap_new:
                    tmp = fac * self.solfwv[iw2 - 1]
                    self.solwaer[ib, ii] = self.solwaer[ib, ii] + tmp
                    self.solbnd[ib] += tmp

                self.nv2[ib] = ii

        # -# Compute LW flux weights and interval indices for mapping
        #    spectral bands between lw radiation and aerosol data.

        self.nr1 = np.zeros(self.NLWBND, dtype=np.int32)
        self.nr2 = np.zeros(self.NLWBND, dtype=np.int32)
        NLWSTR = 1

        if self.lalwflg:
            self.eirbnd = np.zeros(self.NLWBND)
            self.eirwaer = np.zeros((self.NLWBND, self.NAERBND))

            ibs = 1
            ibe = 1
            if self.NLWBND > 1:
                wvs = self.wvn_lw1[0]
                wve = self.wvn_lw1[0]
                for ib in range(1, self.NLWBND):
                    mb = ib + NLWSTR - 1
                    if self.wvn_lw1[mb] < wvs:
                        wvs = self.wvn_lw1[mb]
                        ibs = ib
                    if self.wvn_lw1[mb] > wve:
                        wve = self.wvn_lw1[mb]
                        ibe = ib

            for ib in range(self.NLWBND):
                ii = 0
                if self.NLWBND == 1:
                    iw1 = 400  # corresponding 25 mu
                    iw2 = 2500  # corresponding 4  mu
                else:
                    mb = ib + NLWSTR - 1
                    iw1 = round(self.wvn_lw1[mb])
                    iw2 = round(self.wvn_lw2[mb])

                while iw1 > iendwv[ii]:
                    if ii == self.NAERBND - 1:
                        break
                    ii += 1

                if self.lmap_new:
                    if ib == ibs:
                        sumir = 0.0
                    else:
                        sumir = -0.5 * self.eirfwv[iw1 - 1]

                    if ib == ibe:
                        fac = 0.0
                    else:
                        fac = -0.5

                    self.eirbnd[ib] = sumir
                else:
                    sumir = 0.0

                self.nr1[ib] = ii

                for iw in range(iw1 - 1, iw2):
                    self.eirbnd[ib] += self.eirfwv[iw]
                    sumir = sumir + self.eirfwv[iw]

                    if iw == iendwv[ii] - 1:
                        self.eirwaer[ib, ii] = sumir

                        if ii < self.NAERBND - 1:
                            sumir = 0.0
                            ii += 1

                if iw2 != iendwv[ii] - 1:
                    self.eirwaer[ib, ii] = sumir

                if self.lmap_new:
                    tmp = fac * self.eirfwv[iw2 - 1]
                    self.eirwaer[ib, ii] = self.eirwaer[ib, ii] + tmp
                    self.eirbnd[ib] += tmp

                self.nr2[ib] = ii

        # -# Call optavg() to compute spectral band mean properties for each
        # species.

        self.prsref = prsref
        self.haer = haer
        self.rhidext0 = rhidext0
        self.rhidsca0 = rhidsca0
        self.rhidssa0 = rhidssa0
        self.rhidasy0 = rhidasy0
        self.rhdpext0 = rhdpext0
        self.rhdpsca0 = rhdpsca0
        self.rhdpssa0 = rhdpssa0
        self.rhdpasy0 = rhdpasy0
        self.straext0 = straext0
        self.extrhi = extrhi
        self.scarhi = scarhi
        self.ssarhi = ssarhi
        self.asyrhi = asyrhi
        self.extrhd = extrhd
        self.scarhd = scarhd
        self.ssarhd = ssarhd
        self.asyrhd = asyrhd

        self.optavg()

    # This subroutine computes mean aerosols optical properties over each
    # SW radiation spectral band for each of the species components. This
    # program follows GFDL's approach for thick cloud optical property in
    # SW radiation scheme (2000).
    def optavg(self):
        # ==================================================================== !
        #                                                                      !
        # subprogram: optavg                                                   !
        #                                                                      !
        #   compute mean aerosols optical properties over each sw radiation    !
        #   spectral band for each of the species components.  This program    !
        #   follows gfdl's approach for thick cloud opertical property in      !
        #   sw radiation scheme (2000).                                        !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        # major input variables:                                               !
        #   nv1,nv2 (NSWBND) - start/end spectral band indices of aerosol data !
        #                      for each sw radiation spectral band             !
        #   nr1,nr2 (NLWBND) - start/end spectral band indices of aerosol data !
        #                      for each ir radiation spectral band             !
        #   solwaer (NSWBND,NAERBND)                                           !
        #                    - solar flux weight over each sw radiation band   !
        #                      vs each aerosol data spectral band              !
        #   eirwaer (NLWBND,NAERBND)                                           !
        #                    - ir flux weight over each lw radiation band      !
        #                      vs each aerosol data spectral band              !
        #   solbnd  (NSWBND) - solar flux weight over each sw radiation band   !
        #   eirbnd  (NLWBND) - ir flux weight over each lw radiation band      !
        #   NSWBND           - total number of sw spectral bands               !
        #   NLWBND           - total number of lw spectral bands               !
        #                                                                      !
        # external module variables:  (in physparam)                           !
        #   laswflg          - control flag for sw spectral region             !
        #   lalwflg          - control flag for lw spectral region             !
        #                                                                      !
        # output variables: (to module variables)                              !
        #                                                                      !
        #  ==================================================================  !    #

        #  --- ...  loop for each sw radiation spectral band

        if self.laswflg:

            for nb in range(self.NSWBND):
                rsolbd = 1.0 / self.solbnd[nb]

                #  ---  for rh independent aerosol species

                for nc in range(self.NCM1):  #  ---  for rh independent aerosol species
                    sumk = 0.0
                    sums = 0.0
                    sumok = 0.0
                    sumokg = 0.0
                    sumreft = 0.0

                    for ni in range(self.nv1[nb], self.nv2[nb] + 1):
                        sp = np.sqrt(
                            (1.0 - self.rhidssa0[ni, nc])
                            / (1.0 - self.rhidssa0[ni, nc] * self.rhidasy0[ni, nc])
                        )
                        reft = (1.0 - sp) / (1.0 + sp)
                        sumreft = sumreft + reft * self.solwaer[nb, ni]

                        sumk = sumk + self.rhidext0[ni, nc] * self.solwaer[nb, ni]
                        sums = sums + self.rhidsca0[ni, nc] * self.solwaer[nb, ni]
                        sumok = (
                            sumok
                            + self.rhidssa0[ni, nc]
                            * self.solwaer[nb, ni]
                            * self.rhidext0[ni, nc]
                        )
                        sumokg = (
                            sumokg
                            + self.rhidssa0[ni, nc]
                            * self.solwaer[nb, ni]
                            * self.rhidext0[ni, nc]
                            * self.rhidasy0[ni, nc]
                        )

                    refb = sumreft * rsolbd

                    self.extrhi[nc, nb] = sumk * rsolbd
                    self.scarhi[nc, nb] = sums * rsolbd
                    self.asyrhi[nc, nb] = sumokg / (sumok + 1.0e-10)
                    self.ssarhi[nc, nb] = (
                        4.0
                        * refb
                        / ((1.0 + refb) ** 2 - self.asyrhi[nc, nb] * (1.0 - refb) ** 2)
                    )

                for nc in range(self.NCM2):  #  ---  for rh dependent aerosols species
                    for nh in range(self.NRHLEV):
                        sumk = 0.0
                        sums = 0.0
                        sumok = 0.0
                        sumokg = 0.0
                        sumreft = 0.0

                        for ni in range(self.nv1[nb], self.nv2[nb] + 1):
                            sp = np.sqrt(
                                (1.0 - self.rhdpssa0[ni, nh, nc])
                                / (
                                    1.0
                                    - self.rhdpssa0[ni, nh, nc]
                                    * self.rhdpasy0[ni, nh, nc]
                                )
                            )
                            reft = (1.0 - sp) / (1.0 + sp)
                            sumreft = sumreft + reft * self.solwaer[nb, ni]

                            sumk = (
                                sumk + self.rhdpext0[ni, nh, nc] * self.solwaer[nb, ni]
                            )
                            sums = (
                                sums + self.rhdpsca0[ni, nh, nc] * self.solwaer[nb, ni]
                            )
                            sumok = (
                                sumok
                                + self.rhdpssa0[ni, nh, nc]
                                * self.solwaer[nb, ni]
                                * self.rhdpext0[ni, nh, nc]
                            )
                            sumokg = (
                                sumokg
                                + self.rhdpssa0[ni, nh, nc]
                                * self.solwaer[nb, ni]
                                * self.rhdpext0[ni, nh, nc]
                                * self.rhdpasy0[ni, nh, nc]
                            )

                        refb = sumreft * rsolbd

                        self.extrhd[nh, nc, nb] = sumk * rsolbd
                        self.scarhd[nh, nc, nb] = sums * rsolbd
                        self.asyrhd[nh, nc, nb] = sumokg / (sumok + 1.0e-10)
                        self.ssarhd[nh, nc, nb] = (
                            4.0
                            * refb
                            / (
                                (1.0 + refb) ** 2
                                - self.asyrhd[nh, nc, nb] * (1.0 - refb) ** 2
                            )
                        )

                #  ---  for stratospheric background aerosols

                sumk = 0.0
                for ni in range(self.nv1[nb], self.nv2[nb] + 1):
                    sumk += self.straext0[ni] * self.solwaer[nb, ni]

                self.extstra[nb] = sumk * rsolbd

        #  --- ...  loop for each lw radiation spectral band

        if self.lalwflg:
            for nb in range(self.NLWBND):
                ib = self.NSWBND + nb
                rirbd = 1.0 / self.eirbnd[nb]

                for nc in range(self.NCM1):  #  ---  for rh independent aerosol species
                    sumk = 0.0
                    sums = 0.0
                    sumok = 0.0
                    sumokg = 0.0
                    sumreft = 0.0

                    for ni in range(self.nr1[nb], self.nr2[nb] + 1):
                        sp = np.sqrt(
                            (1.0 - self.rhidssa0[ni, nc])
                            / (1.0 - self.rhidssa0[ni, nc] * self.rhidasy0[ni, nc])
                        )
                        reft = (1.0 - sp) / (1.0 + sp)
                        sumreft = sumreft + reft * self.eirwaer[nb, ni]

                        sumk = sumk + self.rhidext0[ni, nc] * self.eirwaer[nb, ni]
                        sums = sums + self.rhidsca0[ni, nc] * self.eirwaer[nb, ni]
                        sumok = (
                            sumok
                            + self.rhidssa0[ni, nc]
                            * self.eirwaer[nb, ni]
                            * self.rhidext0[ni, nc]
                        )
                        sumokg += (
                            self.rhidssa0[ni, nc]
                            * self.eirwaer[nb, ni]
                            * self.rhidext0[ni, nc]
                            * self.rhidasy0[ni, nc]
                        )

                    refb = sumreft * rirbd

                    self.extrhi[nc, ib] = sumk * rirbd
                    self.scarhi[nc, ib] = sums * rirbd
                    self.asyrhi[nc, ib] = sumokg / (sumok + 1.0e-10)
                    self.ssarhi[nc, ib] = (
                        4.0
                        * refb
                        / ((1.0 + refb) ** 2 - self.asyrhi[nc, ib] * (1.0 - refb) ** 2)
                    )

                for nc in range(self.NCM2):  #  ---  for rh dependent aerosols species
                    for nh in range(self.NRHLEV):
                        sumk = 0.0
                        sums = 0.0
                        sumok = 0.0
                        sumokg = 0.0
                        sumreft = 0.0

                        for ni in range(self.nr1[nb], self.nr2[nb] + 1):
                            sp = np.sqrt(
                                (1.0 - self.rhdpssa0[ni, nh, nc])
                                / (
                                    1.0
                                    - self.rhdpssa0[ni, nh, nc]
                                    * self.rhdpasy0[ni, nh, nc]
                                )
                            )
                            reft = (1.0 - sp) / (1.0 + sp)
                            sumreft = sumreft + reft * self.eirwaer[nb, ni]

                            sumk = (
                                sumk + self.rhdpext0[ni, nh, nc] * self.eirwaer[nb, ni]
                            )
                            sums = (
                                sums + self.rhdpsca0[ni, nh, nc] * self.eirwaer[nb, ni]
                            )
                            sumok = (
                                sumok
                                + self.rhdpssa0[ni, nh, nc]
                                * self.eirwaer[nb, ni]
                                * self.rhdpext0[ni, nh, nc]
                            )
                            sumokg += (
                                self.rhdpssa0[ni, nh, nc]
                                * self.eirwaer[nb, ni]
                                * self.rhdpext0[ni, nh, nc]
                                * self.rhdpasy0[ni, nh, nc]
                            )

                        refb = sumreft * rirbd

                        self.extrhd[nh, nc, ib] = sumk * rirbd
                        self.scarhd[nh, nc, ib] = sums * rirbd
                        self.asyrhd[nh, nc, ib] = sumokg / (sumok + 1.0e-10)
                        self.ssarhd[nh, nc, ib] = (
                            4.0
                            * refb
                            / (
                                (1.0 + refb) ** 2
                                - self.asyrhd[nh, nc, ib] * (1.0 - refb) ** 2
                            )
                        )

                #  ---  for stratospheric background aerosols

                sumk = 0.0
                for ni in range(self.nr1[nb], self.nr2[nb] + 1):
                    sumk += self.straext0[ni] * self.eirwaer[nb, ni]

                self.extstra[ib] = sumk * rirbd

    def aer_update(self, iyear, imon, me):
        #  ==================================================================
        #
        #  aer_update checks and update time varying climatology aerosol
        #    data sets.
        #
        #  inputs:                                          size
        #     iyear   - 4-digit calender year                 1
        #     imon    - month of the year                     1
        #     me      - print message control flag            1
        #
        #  outputs: ( none )
        #
        #  external module variables: (in physparam)
        #     lalwflg     - control flag for tropospheric lw aerosol
        #     laswflg     - control flag for tropospheric sw aerosol
        #     lavoflg     - control flag for stratospheric volcanic aerosol
        #
        #  usage:    call aero_update
        #
        #  subprograms called:  trop_update, volc_update
        #
        #  ==================================================================
        #
        # ===> ...  begin here
        #

        self.iyear = iyear
        self.imon = imon
        self.me = me

        if self.imon < 1 or self.imon > 12:
            raise ValueError(
                "***** ERROR in specifying requested month !!! ",
                f"imon = {imon}",
                "***** STOPPED in subroutinte aer_update !!!",
            )

        # -# Call trop_update() to update monthly tropospheric aerosol data.
        if self.lalwflg or self.laswflg:
            self.trop_update()

        # -# Call volc_update() to update yearly stratospheric volcanic aerosol data.
        if self.lavoflg:
            self.volc_update()

    def trop_update(self):
        # This subroutine updates the monthly global distribution of aerosol
        # profiles in five degree horizontal resolution.

        #  ==================================================================  !
        #                                                                      !
        #  subprogram : trop_update                                            !
        #                                                                      !
        #    updates the  monthly global distribution of aerosol profiles in   !
        #    five degree horizontal resolution.                                !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        #  inputs:  (in-scope variables, module constants)                     !
        #   imon     - integer, month of the year                              !
        #   me       - integer, print message control flag                     !
        #                                                                      !
        #  outputs: (module variables)                                         !
        #                                                                      !
        #  external module variables: (in physparam)                           !
        #    aeros_file   - external aerosol data file name                    !
        #                                                                      !
        #  internal module variables:                                          !
        #    kprfg (    IMXAE*JMXAE)   - aeros profile index                   !
        #    idxcg (NXC*IMXAE*JMXAE)   - aeros component index                 !
        #    cmixg (NXC*IMXAE*JMXAE)   - aeros component mixing ratio          !
        #    denng ( 2 *IMXAE*JMXAE)   - aerosols number density               !
        #                                                                      !
        #    NIAERCM      - unit number for input data set                     !
        #                                                                      !
        #  subroutines called: none                                            !
        #                                                                      !
        #  usage:    call trop_update                                          !
        #                                                                      !
        #  ==================================================================  !
        #
        #
        # ===>  ...  begin here
        #
        #  --- ...  reading climatological aerosols data

        file_exist = os.path.isfile(self.aeros_file)

        if file_exist:
            if self.me == 0:
                print(f"Opened aerosol data file: {aeros_file}")
        else:
            raise FileNotFoundError(
                f'Requested aerosol data file "{aeros_file}" not found!',
                "*** Stopped in subroutine trop_update !!",
            )

        ds = xr.open_dataset(self.aeros_file)
        self.kprfg = ds["kprfg"].data
        self.idxcg = ds["idxcg"].data
        self.cmixg = ds["cmixg"].data
        self.denng = ds["denng"].data
        cline = ds["cline"].data

        if self.me == 0:
            print(f"  --- Reading {cline[self.imon-1]}")

    def volc_update(self):
        # This subroutine searches historical volcanic data sets to find and
        # read in monthly 45-degree lat-zone band of optical depth.

        #  ==================================================================  !
        #                                                                      !
        #  subprogram : volc_update                                            !
        #                                                                      !
        #    searches historical volcanic data sets to find and read in        !
        #    monthly 45-degree lat-zone band data of optical depth.            !
        #                                                                      !
        #  ====================  defination of variables  ===================  !
        #                                                                      !
        #  inputs:  (in-scope variables, module constants)                     !
        #   iyear    - integer, 4-digit calender year                 1        !
        #   imon     - integer, month of the year                     1        !
        #   me       - integer, print message control flag            1        !
        #   NIAERCM  - integer, unit number for input data set        1        !
        #                                                                      !
        #  outputs: (module variables)                                         !
        #   ivolae   - integer, monthly, 45-deg lat-zone volc odp      12*4*10 !
        #   kyrstr   - integer, starting year of data in the input file        !
        #   kyrend   - integer, ending   year of data in the input file        !
        #   kyrsav   - integer, the year of data in use in the input file      !
        #   kmonsav  - integer, the month of data in use in the input file     !
        #                                                                      !
        #  subroutines called: none                                            !
        #                                                                      !
        #  usage:    call volc_aerinit                                         !
        #                                                                      !
        #  ==================================================================  !

        #  ---  locals:
        volcano_file = "volcanic_aerosols_1850-1859.txt"
        #
        # ===>  ...  begin here
        #
        self.kmonsav = self.imon

        if (
            self.kyrstr <= self.iyear and self.iyear <= self.kyrend
        ):  # use previously input data
            self.kyrsav = self.iyear
            return
        else:  # need to input new data
            self.kyrsav = self.iyear
            self.kyrstr = self.iyear - self.iyear % 10
            self.kyrend = self.kyrstr + 9

            if self.iyear < self.MINVYR or self.iyear > self.MAXVYR:
                self.ivolae = np.ones((12, 4, 10))  # set as lowest value
                if self.me == 0:
                    print(
                        "Requested volcanic date out of range,",
                        " optical depth set to lowest value",
                    )
            else:
                file_exist = os.path.isfile(volcano_file)
                if file_exist:
                    ds = xr.open_dataset(volcano_file)
                    cline = ds["cline"]
                    #  ---  check print
                    if self.me == 0:
                        print(f"Opened volcanic data file: {volcano_file}")
                        print(cline)

                    self.ivolae = ds["ivolae"]
                else:
                    raise FileNotFoundError(
                        f'Requested volcanic data file "{volcano_file}" not found!',
                        "*** Stopped in subroutine VOLC_AERINIT !!",
                    )

        #  ---  check print
        if self.me == 0:
            k = (self.kyrsav % 10) + 1
            print(
                f"CHECK: Sample Volcanic data used for month, year: {self.imon}, {self.iyear}"
            )
            print(self.ivolae[self.kmonsav, :, k])

    def setaer(
        self,
        prsi,
        prsl,
        prslk,
        tvly,
        rhlay,
        slmsk,
        tracer,
        xlon,
        xlat,
        IMAX,
        NLAY,
        NLP1,
        lsswr,
        lslwr,
    ):
        #  ==================================================================  !
        #                                                                      !
        #  setaer computes aerosols optical properties                         !
        #                                                                      !
        #  inputs:                                                   size      !
        #     prsi    - pressure at interface              mb      IMAX*NLP1   !
        #     prsl    - layer mean pressure                mb      IMAX*NLAY   !
        #     prslk   - exner function = (p/p0)**rocp              IMAX*NLAY   !
        #     tvly    - layer virtual temperature          k       IMAX*NLAY   !
        #     rhlay   - layer mean relative humidity               IMAX*NLAY   !
        #     slmsk   - sea/land mask (sea:0,land:1,sea-ice:2)       IMAX      !
        #     tracer  - aerosol tracer concentration           IMAX*NLAY*NTRAC !
        #     xlon    - longitude of given points in radiance        IMAX      !
        #               ok for both 0->2pi or -pi->+pi ranges                  !
        #     xlat    - latitude of given points in radiance         IMAX      !
        #               default to pi/2 -> -pi/2, otherwise see in-line comment!
        #     IMAX    - horizontal dimension of arrays                  1      !
        #     NLAY,NLP1-vertical dimensions of arrays                   1      !
        #     lsswr,lslwr                                                      !
        #             - logical flags for sw/lw radiation calls         1      !
        #                                                                      !
        #  outputs:                                                            !
        #     aerosw - aeros opt properties for sw      IMAX*NLAY*NBDSW*NF_AESW!
        #               (:,:,:,1): optical depth                               !
        #               (:,:,:,2): single scattering albedo                    !
        #               (:,:,:,3): asymmetry parameter                         !
        #     aerolw - aeros opt properties for lw      IMAX*NLAY*NBDLW*NF_AELW!
        #               (:,:,:,1): optical depth                               !
        #               (:,:,:,2): single scattering albedo                    !
        #               (:,:,:,3): asymmetry parameter                         !
        #     tau_gocart - 550nm aeros opt depth     IMAX*NLAY*MAX_NUM_GRIDCOMP!
        #!    aerodp - vertically integrated optical depth         IMAX*NSPC1  !
        #                                                                      !
        #  external module variable: (in physparam)                            !
        #     iaerflg - aerosol effect control flag (volc,lw,sw, 3-dig)        !
        #     laswflg - tropospheric aerosol control flag for sw radiation     !
        #               =f: no sw aeros calc.  =t: do sw aeros calc.           !
        #     lalwflg - tropospheric aerosol control flag for lw radiation     !
        #               =f: no lw aeros calc.  =t: do lw aeros calc.           !
        #     lavoflg - control flag for stratospheric vocanic aerosols        !
        #               =t: add volcanic aerosols to the background aerosols   !
        #     ivflip  - control flag for direction of vertical index           !
        #               =0: index from toa to surface                          !
        #               =1: index from surface to toa                          !
        #                                                                      !
        #  internal module variable: (set by subroutine aer_init)              !
        #     ivolae  - stratosphere volcanic aerosol optical depth (fac 1.e4) !
        #                                                     12*4*10          !
        #  usage:    call setaer                                               !
        #                                                                      !
        #  subprograms called:  aer_property                                   !
        #                                                                      !
        #  ==================================================================  !

        #  ---  outputs:
        aerosw = np.zeros((IMAX, NLAY, nbdsw, self.NF_AESW))
        aerolw = np.zeros((IMAX, NLAY, NBDLW, self.NF_AELW))

        aerodp = np.zeros((IMAX, self.NSPC1))

        #  ---  locals:
        psrfh = 5.0  # ref press (mb) for upper bound

        alon = np.zeros(IMAX)
        alat = np.zeros(IMAX)
        volcae = np.zeros(IMAX)
        rdelp = np.zeros(IMAX)

        prsln = np.zeros(NLP1)
        hz = np.zeros((IMAX, NLP1))
        dz = np.zeros((IMAX, NLAY))

        kcutl = np.zeros(IMAX, dtype=DTYPE_INT)
        kcuth = np.zeros(IMAX, dtype=DTYPE_INT)

        laddsw = False
        laersw = False
        laddlw = False
        laerlw = False

        #  ---  conversion constants
        rdg = 180.0 / con_pi
        rovg = 0.001 * con_rd / con_g

        # ===>  ...  begin here

        if not (lsswr or lslwr):
            return

        if self.iaerflg <= 0:
            return

        laersw = lsswr and self.laswflg
        laerlw = lslwr and self.lalwflg

        # Convert lat/lon from radiance to degree.

        for i in range(IMAX):
            alon[i] = xlon[i] * rdg
            if alon[i] < 0.0:
                alon[i] = alon[i] + 360.0

            alat[i] = xlat[i] * rdg  # if xlat in pi/2 -> -pi/2 range

        # Compute level height and layer thickness.

        if self.laswflg or self.lalwflg:

            for i in range(IMAX):

                if self.ivflip == 1:  # input from sfc to toa

                    for k in range(NLAY):
                        prsln[k] = np.log(prsi[i, k])

                    prsln[NLP1 - 1] = np.log(prsl[i, NLAY - 1])

                    for k in range(NLAY - 1, -1, -1):
                        dz[i, k] = rovg * (prsln[k] - prsln[k + 1]) * tvly[i, k]

                    dz[i, NLAY - 1] = 2.0 * dz[i, NLAY - 1]

                    hz[i, 0] = 0.0
                    for k in range(NLAY):
                        hz[i, k + 1] = hz[i, k] + dz[i, k]

                else:  # input from toa to sfc

                    prsln[0] = np.log(prsl[i, 0])
                    for k in range(1, NLP1):
                        prsln[k] = np.log(prsi[i, k])

                    for k in range(NLAY):
                        dz[i, k] = rovg * (prsln[k + 1] - prsln[k]) * tvly[i, k]

                    dz[i, 0] = 2.0 * dz[i, 0]

                    hz[i, NLP1 - 1] = 0.0
                    for k in range(NLAY - 1, -1, -1):
                        hz[i, k] = hz[i, k + 1] + dz[i, k]

            # -# Calculate SW aerosol optical properties for the corresponding
            #    frequency bands:
            #    - if opac aerosol climatology is used, call aer_property(): this
            #      subroutine maps the 5 degree global climatological aerosol data
            #      set onto model grids, and compute aerosol optical properties for
            #      SW and LW radiations.
            #    - if gocart aerosol scheme is used, call setgocartaer(): this
            #      subroutine computes sw + lw aerosol optical properties for gocart
            #      aerosol species (merged from fcst and clim fields).

            if self.iaermdl == 0 or self.iaermdl == 5:  # use opac aerosol climatology
                aerosw, aerolw, aerodp = self.aer_property(
                    prsi,
                    prsl,
                    prslk,
                    tvly,
                    rhlay,
                    dz,
                    hz,
                    tracer,
                    alon,
                    alat,
                    slmsk,
                    laersw,
                    laerlw,
                    IMAX,
                    NLAY,
                    NLP1,
                )

            elif self.iaermdl == 1:  # use gocart aerosol scheme

                raise NotImplementedError("GOCART Aerosol scheme not implemented")

        # -# Compute stratosphere volcanic forcing:
        #    - select data in 4 lat bands, interpolation at the boundaries
        #    - Find lower boundary of stratosphere: polar, fixed at 25000pa
        #      (250mb); tropic, fixed at 15000pa (150mb); mid-lat, interpolation
        #    - SW: add volcanic aerosol optical depth to the background value
        #    - Smoothing profile at boundary if needed
        #    - LW: add volcanic aerosol optical depth to the background value
        # ---  ...  stratosphere volcanic forcing

        if self.lavoflg:

            if self.iaerflg == 100:
                laddsw = lsswr
                laddlw = lslwr
            else:
                laddsw = lsswr and self.laswflg
                laddlw = lslwr and self.lalwflg

            i1 = np.mod(self.kyrsav, 10)

            #  ---  select data in 4 lat bands, interpolation at the boundaires

            for i in range(IMAX):
                if alat[i] > 46.0:
                    volcae[i] = 1.0e-4 * self.ivolae[self.kmonsav - 1, 0, i1]
                elif alat[i] > 44.0:
                    volcae[i] = 5.0e-5 * (
                        self.ivolae[self.kmonsav - 1, 0, i1]
                        + self.ivolae[self.kmonsav - 1, 1, i1]
                    )
                elif alat[i] > 1.0:
                    volcae[i] = 1.0e-4 * self.ivolae[self.kmonsav - 1, 1, i1]
                elif alat[i] > -1.0:
                    volcae[i] = 5.0e-5 * (
                        self.ivolae[self.kmonsav - 1, 1, i1]
                        + self.ivolae[self.kmonsav - 1, 2, i1]
                    )
                elif alat[i] > -44.0:
                    volcae[i] = 1.0e-4 * self.ivolae[self.kmonsav - 1, 2, i1]
                elif alat[i] > -46.0:
                    volcae[i] = 5.0e-5 * (
                        self.ivolae[self.kmonsav - 1, 2, i1]
                        + self.ivolae[self.kmonsav - 1, 3, i1]
                    )
                else:
                    volcae[i] = 1.0e-4 * self.ivolae[self.kmonsav - 1, 3, i1]

            if self.ivflip == 0:  # input data from toa to sfc

                #  ---  find lower boundary of stratosphere

                for i in range(IMAX):

                    tmp1 = np.abs(alat[i])
                    if tmp1 > 70.0:  # polar, fixed at 25000pa (250mb)
                        psrfl = 250.0
                    elif tmp1 < 20.0:  # tropic, fixed at 15000pa (150mb)
                        psrfl = 150.0
                    else:  # mid-lat, interpolation
                        psrfl = 110.0 + 2.0 * tmp1

                    kcuth[i] = NLAY - 1
                    kcutl[i] = 2
                    rdelp[i] = 1.0 / prsi[i, 1]

                    for k in range(1, NLAY - 2):
                        if prsi[i, k] >= psrfh:
                            kcuth[i] = k
                            break

                    for k in range(1, NLAY - 2):
                        if prsi[i, k] >= psrfl:
                            kcutl[i] = k
                            rdelp[i] = 1.0 / (prsi[i, k] - prsi[i, kcuth[i] - 1])
                            break

                #  ---  sw: add volcanic aerosol optical depth to the background value

                if laddsw:
                    for m in range(nbdsw):
                        mb = NSWSTR + m - 1

                        if self.wvn_sw1[mb] > 20000:  # range of wvlth < 0.5mu
                            tmp2 = 0.74
                        elif self.wvn_sw2[mb] < 20000:  # range of wvlth > 0.5mu
                            tmp2 = 1.14
                        else:  # range of wvlth in btwn
                            tmp2 = 0.94

                        tmp1 = (
                            0.275e-4 * (self.wvn_sw2[mb] + self.wvn_sw1[mb])
                        ) ** tmp2

                        for i in range(IMAX):
                            kh = kcuth[i]
                            kl = kcutl[i]
                            for k in range(kh - 1, kl):
                                tmp2 = tmp1 * ((prsi[i, k + 1] - prsi[i, k]) * rdelp[i])
                                aerosw[i, k, m, 0] = (
                                    aerosw[i, k, m, 0] + tmp2 * volcae[i]
                                )

                            #  ---  smoothing profile at boundary if needed

                            if aerosw[i, kl, m, 0] > 10.0 * aerosw[i, kl + 1, m, 0]:
                                tmp2 = aerosw[i, kl, m, 0] + aerosw[i, kl + 1, m, 0]
                                aerosw[i, kl, m, 0] = 0.8 * tmp2
                                aerosw[i, kl + 1, m, 0] = 0.2 * tmp2

                #  ---  lw: add volcanic aerosol optical depth to the background value

                if laddlw:
                    if self.NLWBND == 1:

                        tmp1 = (0.55 / 11.0) ** 1.2
                        for i in range(IMAX):
                            kh = kcuth[i]
                            kl = kcutl[i]
                            for k in range(kh - 1, kl):
                                tmp2 = (
                                    tmp1
                                    * ((prsi[i, k + 1] - prsi[i, k]) * rdelp[i])
                                    * volcae[i]
                                )

                                for m in range(NBDLW):
                                    aerolw[i, k, m, 0] = aerolw[i, k, m, 0] + tmp2

                    else:

                        for m in range(NBDLW):
                            tmp1 = (
                                0.275e-4 * (self.wvn_lw2[m] + self.wvn_lw1[m])
                            ) ** 1.2

                            for i in range(IMAX):
                                kh = kcuth[i]
                                kl = kcutl[i]
                                for k in range(kh - 1, kl):
                                    tmp2 = tmp1 * (
                                        (prsi[i, k + 1] - prsi[i, k]) * rdelp[i]
                                    )
                                    aerolw[i, k, m, 0] = (
                                        aerolw[i, k, m, 0] + tmp2 * volcae[i]
                                    )

            else:  # input data from sfc to toa

                #  ---  find lower boundary of stratosphere

                for i in range(IMAX):

                    tmp1 = np.abs(alat[i])

                    if tmp1 > 70.0:  # polar, fixed at 25000pa (250mb)
                        psrfl = 250.0
                    elif tmp1 < 20.0:  # tropic, fixed at 15000pa (150mb)
                        psrfl = 150.0
                    else:  # mid-lat, interpolation
                        psrfl = 110.0 + 2.0 * tmp1

                    kcuth[i] = 2
                    kcutl[i] = NLAY - 1
                    rdelp[i] = 1.0 / prsi[i, NLAY - 2]

                    for k in range(NLAY - 2, 0, -1):
                        if prsi[i, k] >= psrfh:
                            kcuth[i] = k + 1
                            break

                    for k in range(NLAY - 1, 0, -1):
                        if prsi[i, k] >= psrfl:
                            kcutl[i] = k + 1
                            rdelp[i] = 1.0 / (prsi[i, k] - prsi[i, kcuth[i]])
                            break

                #  ---  sw: add volcanic aerosol optical depth to the background value

                if laddsw:
                    for m in range(nbdsw):
                        mb = NSWSTR + m - 1

                        if self.wvn_sw1[mb] > 20000:  # range of wvlth < 0.5mu
                            tmp2 = 0.74
                        elif self.wvn_sw2[mb] < 20000:  # range of wvlth > 0.5mu
                            tmp2 = 1.14
                        else:  # range of wvlth in btwn
                            tmp2 = 0.94

                        tmp1 = (
                            0.275e-4 * (self.wvn_sw2[mb] + self.wvn_sw1[mb])
                        ) ** tmp2

                        for i in range(IMAX):
                            kh = kcuth[i] - 1
                            kl = kcutl[i] - 1
                            for k in range(kl, kh + 1):
                                tmp2 = tmp1 * ((prsi[i, k] - prsi[i, k + 1]) * rdelp[i])
                                aerosw[i, k, m, 0] = (
                                    aerosw[i, k, m, 0] + tmp2 * volcae[i]
                                )

                            #  ---  smoothing profile at boundary if needed

                            if aerosw[i, kl, m, 0] > 10.0 * aerosw[i, kl - 1, m, 0]:
                                tmp2 = aerosw[i, kl, m, 0] + aerosw[i, kl - 1, m, 0]
                                aerosw[i, kl, m, 0] = 0.8 * tmp2
                                aerosw[i, kl - 1, m, 0] = 0.2 * tmp2

                #  ---  lw: add volcanic aerosol optical depth to the background value

                if laddlw:
                    if self.NLWBND == 1:

                        tmp1 = (0.55 / 11.0) ** 1.2
                        for i in range(IMAX):
                            kh = kcuth[i] - 1
                            kl = kcutl[i] - 1
                            for k in range(kl, kh + 1):
                                tmp2 = (
                                    tmp1
                                    * ((prsi[i, k] - prsi[i, k + 1]) * rdelp[i])
                                    * volcae[i]
                                )
                                for m in range(NBDLW):
                                    aerolw[i, k, m, 0] = aerolw[i, k, m, 0] + tmp2

                    else:

                        for m in range(NBDLW):
                            tmp1 = (
                                0.275e-4 * (self.wvn_lw2[m] + self.wvn_lw1[m])
                            ) ** 1.2

                            for i in range(IMAX):
                                kh = kcuth[i] - 1
                                kl = kcutl[i] - 1
                                for k in range(kl, kh + 1):
                                    tmp2 = tmp1 * (
                                        (prsi[i, k] - prsi[i, k + 1]) * rdelp[i]
                                    )
                                    aerolw[i, k, m, 0] = (
                                        aerolw[i, k, m, 0] + tmp2 * volcae[i]
                                    )
        return aerosw, aerolw, aerodp

    def aer_property(
        self,
        prsi,
        prsl,
        prslk,
        tvly,
        rhlay,
        dz,
        hz,
        tracer,
        alon,
        alat,
        slmsk,
        laersw,
        laerlw,
        IMAX,
        NLAY,
        NLP1,
    ):
        #  ==================================================================  !
        #                                                                      !
        #  aer_property maps the 5 degree global climatological aerosol data   !
        #  set onto model grids, and compute aerosol optical properties for sw !
        #  and lw radiations.                                                  !
        #                                                                      !
        #  inputs:                                                             !
        #     prsi    - pressure at interface              mb      IMAX*NLP1   !
        #     prsl    - layer mean pressure         (not used)     IMAX*NLAY   !
        #     prslk   - exner function=(p/p0)**rocp (not used)     IMAX*NLAY   !
        #     tvly    - layer virtual temperature   (not used)     IMAX*NLAY   !
        #     rhlay   - layer mean relative humidity               IMAX*NLAY   !
        #     dz      - layer thickness                    m       IMAX*NLAY   !
        #     hz      - level high                         m       IMAX*NLP1   !
        #     tracer  - aer tracer concentrations   (not used)  IMAX*NLAY*NTRAC!
        #     alon, alat                                             IMAX      !
        #             - longitude and latitude of given points in degree       !
        #     slmsk   - sea/land mask (sea:0,land:1,sea-ice:2)       IMAX      !
        #     laersw,laerlw                                             1      !
        #             - logical flag for sw/lw aerosol calculations            !
        #     IMAX    - horizontal dimension of arrays                  1      !
        #     NLAY,NLP1-vertical dimensions of arrays                   1      !
        #!    NSPC    - num of species for optional aod output fields   1      !
        #                                                                      !
        #  outputs:                                                            !
        #     aerosw - aeros opt properties for sw      IMAX*NLAY*NBDSW*NF_AESW!
        #               (:,:,:,1): optical depth                               !
        #               (:,:,:,2): single scattering albedo                    !
        #               (:,:,:,3): asymmetry parameter                         !
        #     aerolw - aeros opt properties for lw      IMAX*NLAY*NBDLW*NF_AELW!
        #               (:,:,:,1): optical depth                               !
        #               (:,:,:,2): single scattering albedo                    !
        #               (:,:,:,3): asymmetry parameter                         !
        #!    aerodp - vertically integrated aer-opt-depth         IMAX*NSPC+1 !
        #                                                                      !
        #  module parameters and constants:                                    !
        #     NSWBND  - total number of actual sw spectral bands computed      !
        #     NLWBND  - total number of actual lw spectral bands computed      !
        #     NSWLWBD - total number of sw+lw bands computed                   !
        #                                                                      !
        #  external module variables: (in physparam)                           !
        #     ivflip  - control flag for direction of vertical index           !
        #               =0: index from toa to surface                          !
        #               =1: index from surface to toa                          !
        #                                                                      !
        #  module variable: (set by subroutine aer_init)                       !
        #     kprfg   - aerosols profile index                IMXAE*JMXAE      !
        #               1:ant  2:arc  3:cnt  4:mar  5:des  6:marme 7:cntme     !
        #     idxcg   - aerosols component index              NXC*IMXAE*JMXAE  !
        #               1:inso    2:soot    3:minm    4:miam    5:micm         !
        #               6:mitr    7:waso    8:ssam    9:sscm   10:suso         !
        #     cmixg   - aerosols component mixing ratio       NXC*IMXAE*JMXAE  !
        #     denng   - aerosols number density                2 *IMXAE*JMXAE  !
        #               1:for domain-1   2:domain-2 (prof marme/cntme only)    !
        #                                                                      !
        #  usage:    call aer_property                                         !
        #                                                                      !
        #  subprograms called:  radclimaer                                     !
        #                                                                      !
        #  ==================================================================  !

        #  ---  outputs:
        aerosw = np.zeros((IMAX, NLAY, nbdsw, self.NF_AESW))
        aerolw = np.zeros((IMAX, NLAY, NBDLW, self.NF_AELW))

        aerodp = np.zeros((IMAX, self.NSPC1))

        #  ---  locals:
        self.cmix = np.zeros(self.NCM)
        self.denn = np.zeros(2)
        self.spcodp = np.zeros(self.NSPC)

        self.delz = np.zeros(NLAY)
        self.rh1 = np.zeros(NLAY)
        self.dz1 = np.zeros(NLAY)
        self.idmaer = np.zeros(NLAY, dtype=DTYPE_INT)

        self.tauae = np.zeros((NLAY, self.NSWLWBD))
        self.ssaae = np.zeros((NLAY, self.NSWLWBD))
        self.asyae = np.zeros((NLAY, self.NSWLWBD))

        #  ---  conversion constants
        dltg = 360.0 / float(self.IMXAE)
        hdlt = 0.5 * dltg
        rdlt = 1.0 / dltg

        # -# Map aerosol data to model grids
        #    - Map grid in longitude direction, lon from 0 to 355 deg resolution
        #    - Map grid in latitude direction, lat from 90n to 90s in 5 deg resolution

        i1 = 1
        i2 = 2
        j1 = 1
        j2 = 2

        for i in range(IMAX):

            #  ---  map grid in longitude direction, lon from 0 to 355 deg resolution
            i3 = i1
            while i3 <= self.IMXAE:
                tmp1 = dltg * (i3 - 1)
                dtmp = alon[i] - tmp1

                if dtmp > dltg:
                    i3 += 1
                    if i3 > self.IMXAE:
                        raise ValueError(
                            f"ERROR! In setclimaer alon>360. ipt = {i}",
                            f"dltg,alon,tlon,dlon = {dltg},{alon[i]},{tmp1},{dtmp}",
                        )
                elif dtmp >= 0.0:
                    i1 = i3
                    i2 = np.mod(i3, self.IMXAE) + 1
                    wi = dtmp * rdlt
                    if dtmp <= hdlt:
                        kpi = i3
                    else:
                        kpi = i2
                    break
                else:
                    i3 -= 1
                    if i3 < 1:
                        raise ValueError(
                            f"ERROR! In setclimaer alon< 0. ipt = {i}",
                            f"dltg, alon, tlon, dlon = {dltg}, {alon[i]}, {tmp1},{dtmp}",
                        )

            #  ---  map grid in latitude direction, lat from 90n to 90s in 5 deg resolution
            j3 = j1
            while j3 <= self.JMXAE:
                tmp2 = 90.0 - dltg * (j3 - 1)
                dtmp = tmp2 - alat[i]

                if dtmp > dltg:
                    j3 += 1
                    if j3 >= self.JMXAE:
                        raise ValueError(
                            f"ERROR! In setclimaer alat<-90. ipt = {i}",
                            f"dltg, alat, tlat, dlat = {dltg}, {alat[i]}, {tmp2}, {dtmp}",
                        )
                elif dtmp >= 0.0:
                    j1 = j3
                    j2 = j3 + 1
                    wj = dtmp * rdlt
                    if dtmp <= hdlt:
                        kpj = j3
                    else:
                        kpj = j2

                    break
                else:
                    j3 -= 1
                    if j3 < 1:
                        raise ValueError(
                            f"ERROR! In setclimaer alat>90. ipt ={i}",
                            f"dltg, alat, tlat, dlat = {dltg}, {alat[i]}, {tmp2}, {dtmp}",
                        )

            # -# Determin the type of aerosol profile (kp) and scale hight for
            #    domain 1 (h1) to be used at this grid point.

            kp = self.kprfg[
                kpi - 1, kpj - 1
            ]  # nearest typical aeros profile as default
            kpa = max(
                self.kprfg[i1 - 1, j1 - 1],
                self.kprfg[i1 - 1, j2 - 1],
                self.kprfg[i2 - 1, j1 - 1],
                self.kprfg[i2 - 1, j2 - 1],
            )
            h1 = self.haer[0, kp - 1]
            self.denn[1] = 0.0
            ii = 1

            if kp != kpa:
                if kpa == 6:  # if ocean prof with mineral aeros overlay
                    ii = 2  # need 2 types of densities
                    if slmsk[i] > 0.0:  # but actually a land/sea-ice point
                        kp = 7  # reset prof index to land
                        h1 = 0.5 * (
                            self.haer[0, 5] + self.haer[0, 6]
                        )  # use a transition scale hight
                    else:
                        kp = kpa
                        h1 = self.haer[0, 5]
                elif kpa == 7:  # if land prof with mineral aeros overlay
                    ii = 2  # need 2 types of densities
                    if slmsk[i] <= 0.0:  # but actually an ocean point
                        kp = 6  # reset prof index to ocean
                        h1 = 0.5 * (
                            self.haer[0, 5] + self.haer[0, 6]
                        )  # use a transition scale hight
                    else:
                        kp = kpa
                        h1 = self.haer[0, 6]
                else:  # lower atmos without mineral aeros overlay
                    h1 = self.haer[0, kpa - 1]
                    kp = kpa

            # Compute horizontal bi-linear interpolation weights

            w11 = (1.0 - wi) * (1.0 - wj)
            w12 = (1.0 - wi) * wj
            w21 = wi * (1.0 - wj)
            w22 = wi * wj

            # -# Do horizontal bi-linear interpolation on aerosol partical density
            #   (denn)

            for m in range(ii):  # ii=1 for domain 1; =2 for domain 2.
                self.denn[m] = (
                    w11 * self.denng[m, i1 - 1, j1 - 1]
                    + w12 * self.denng[m, i1 - 1, j2 - 1]
                    + w21 * self.denng[m, i2 - 1, j1 - 1]
                    + w22 * self.denng[m, i2 - 1, j2 - 1]
                )

            # -# Do horizontal bi-linear interpolation on mixing ratios

            self.cmix[:] = 0.0
            for m in range(self.NXC):
                ii = self.idxcg[m, i1 - 1, j1 - 1] - 1
                if ii > -1:
                    self.cmix[ii] = self.cmix[ii] + w11 * self.cmixg[m, i1 - 1, j1 - 1]
                ii = self.idxcg[m, i1 - 1, j2 - 1] - 1
                if ii > -1:
                    self.cmix[ii] = self.cmix[ii] + w12 * self.cmixg[m, i1 - 1, j2 - 1]
                ii = self.idxcg[m, i2 - 1, j1 - 1] - 1
                if ii > -1:
                    self.cmix[ii] = self.cmix[ii] + w21 * self.cmixg[m, i2 - 1, j1 - 1]
                ii = self.idxcg[m, i2 - 1, j2 - 1] - 1
                if ii > -1:
                    self.cmix[ii] = self.cmix[ii] + w22 * self.cmixg[m, i2 - 1, j2 - 1]

            # -# Prepare to setup domain index array and effective layer thickness,
            #    also convert pressure level to sigma level to follow the terrain.

            for k in range(NLAY):
                self.rh1[k] = rhlay[i, k]
                self.dz1[k] = dz[i, k]

            if self.ivflip == 1:  # input from sfc to toa

                if prsi[i, 0] > 100.0:
                    rps = 1.0 / prsi[i, 0]
                else:
                    raise ValueError(
                        f"!!! (1) Error in subr radiation_aerosols:",
                        f" unrealistic surface pressure = {i},{prsi[i, 0]}",
                    )

                ii = 0
                for k in range(NLAY):
                    if prsi[i, k + 1] * rps < self.sigref[ii, kp - 1]:
                        ii += 1
                        if ii == 1 and self.prsref[1, kp - 1] == self.prsref[2, kp - 1]:
                            ii = 2

                    self.idmaer[k] = ii + 1

                    if ii > 0:
                        tmp1 = self.haer[ii, kp - 1]
                    else:
                        tmp1 = h1

                    if tmp1 > 0.0:
                        tmp2 = 1.0 / tmp1
                        self.delz[k] = tmp1 * (
                            np.exp(-hz[i, k] * tmp2) - np.exp(-hz[i, k + 1] * tmp2)
                        )
                    else:
                        self.delz[k] = self.dz1[k]

            else:  # input from toa to sfc

                if prsi[i, NLP1 - 1] > 100.0:
                    rps = 1.0 / prsi[i, NLP1 - 1]
                else:
                    raise ValueError(
                        f"!!! (2) Error in subr radiation_aerosols:",
                        f"unrealistic surface pressure = {i}, {prsi[i, NLP1-1]}",
                    )

                ii = 0
                for k in range(NLAY - 1, -1, -1):
                    if prsi[i, k] * rps < self.sigref[ii, kp - 1]:
                        ii += 1
                        if ii == 1 and self.prsref[1, kp - 1] == self.prsref[2, kp - 1]:
                            ii = 2

                    self.idmaer[k] = ii + 1

                    if ii > 0:
                        tmp1 = self.haer[ii, kp - 1]
                    else:
                        tmp1 = h1

                    if tmp1 > 0.0:
                        tmp2 = 1.0 / tmp1
                        self.delz[k] = tmp1 * (
                            np.exp(-hz[i, k + 1] * tmp2) - np.exp(-hz[i, k] * tmp2)
                        )
                    else:
                        self.delz[k] = self.dz1[k]

            # -# Call radclimaer() to calculate SW/LW aerosol optical properties
            #    for the corresponding frequency bands.

            self.radclimaer()

            if laersw:
                for m in range(nbdsw):
                    for k in range(NLAY):
                        aerosw[i, k, m, 0] = self.tauae[k, m]
                        aerosw[i, k, m, 1] = self.ssaae[k, m]
                        aerosw[i, k, m, 2] = self.asyae[k, m]

                #  ---  total aod (optional)
                for k in range(NLAY):
                    aerodp[i, 0] = aerodp[i, 0] + self.tauae[k, self.nv_aod - 1]

                #  ---  for diagnostic output (optional)
                for m in range(self.NSPC):
                    aerodp[i, m + 1] = self.spcodp[m]

            if laerlw:
                if self.NLWBND == 1:
                    m1 = self.NSWBND + 1
                    for m in range(NBDLW):
                        for k in range(NLAY):
                            aerolw[i, k, m, 0] = self.tauae[k, m1]
                            aerolw[i, k, m, 1] = self.ssaae[k, m1]
                            aerolw[i, k, m, 2] = self.asyae[k, m1]
                else:
                    for m in range(NBDLW):
                        m1 = self.NSWBND + m
                        for k in range(NLAY):
                            aerolw[i, k, m, 0] = self.tauae[k, m1]
                            aerolw[i, k, m, 1] = self.ssaae[k, m1]
                            aerolw[i, k, m, 2] = self.asyae[k, m1]

        return aerosw, aerolw, aerodp

    # This subroutine computes aerosols optical properties in NSWLWBD
    # bands. there are seven different vertical profile structures. in the
    # troposphere, aerosol distribution at each grid point is composed
    # from up to six components out of ten different substances.

    def radclimaer(self):
        #  ==================================================================  !
        #                                                                      !
        #  compute aerosols optical properties in NSWLWBD bands. there are     !
        #  seven different vertical profile structures. in the troposphere,    !
        #  aerosol distribution at each grid point is composed from up to      !
        #  six components out of a total of ten different substances.          !
        #                                                                      !
        #  ref: wmo report wcp-112 (1986)                                      !
        #                                                                      !
        #  input variables:                                                    !
        #     cmix   - mixing ratioes of aerosol components  -     NCM         !
        #     denn   - aerosol number densities              -     2           !
        #     rh1    - relative humidity                     -     NLAY        !
        #     delz   - effective layer thickness             km    NLAY        !
        #     idmaer - aerosol domain index                  -     NLAY        !
        #     NXC    - number of different aerosol components-     1           !
        #     NLAY   - vertical dimensions                   -     1           !
        #                                                                      !
        #  output variables:                                                   !
        #     tauae  - optical depth                         -     NLAY*NSWLWBD!
        #     ssaae  - single scattering albedo              -     NLAY*NSWLWBD!
        #     asyae  - asymmetry parameter                   -     NLAY*NSWLWBD!
        #!    aerodp - vertically integrated aer-opt-depth   -     IMAX*NSPC+1 !
        #                                                                      !
        #  ==================================================================  !
        #
        crt1 = 30.0
        crt2 = 0.03333

        self.spcodp = np.zeros(self.NSPC)

        # ===> ... loop over vertical layers from top to surface

        for kk in range(self.NLAY):

            # --- linear interp coeffs for rh-dep species

            ih2 = 1
            while self.rh1[kk] > self.rhlev[ih2 - 1]:
                ih2 += 1
                if ih2 > self.NRHLEV:
                    break

            ih1 = max(1, ih2 - 1) - 1
            ih2 = min(self.NRHLEV, ih2) - 1

            drh0 = self.rhlev[ih2] - self.rhlev[ih1]
            drh1 = self.rh1[kk] - self.rhlev[ih1]
            if ih1 == ih2:
                rdrh = 0.0
            else:
                rdrh = drh1 / drh0

            # --- assign optical properties in each domain

            idom = self.idmaer[kk]

            if idom == 5:
                # --- 5th domain - upper stratosphere assume no aerosol

                for ib in range(self.NSWLWBD):
                    self.tauae[kk, ib] = 0.0
                    if ib <= self.NSWBND - 1:
                        self.ssaae[kk, ib] = 0.99
                        self.asyae[kk, ib] = 0.696
                    else:
                        self.ssaae[kk, ib] = 0.5
                        self.asyae[kk, ib] = 0.3

            elif idom == 4:
                # --- 4th domain - stratospheric layers

                for ib in range(self.NSWLWBD):
                    self.tauae[kk, ib] = self.extstra[ib] * self.delz[kk]
                    if ib <= self.NSWBND - 1:
                        self.ssaae[kk, ib] = 0.99
                        self.asyae[kk, ib] = 0.696
                    else:
                        self.ssaae[kk, ib] = 0.5
                        self.asyae[kk, ib] = 0.3

                # --- compute aod from individual species' contribution (optional)
                idx = self.idxspc[9] - 1  # for sulfate
                self.spcodp[idx] = self.spcodp[idx] + self.tauae[kk, self.nv_aod - 1]

            elif idom == 3:
                # --- 3rd domain - free tropospheric layers
                #   1:inso 0.17e-3; 2:soot 0.4; 7:waso 0.59983; n:730

                for ib in range(self.NSWLWBD):
                    ex01 = self.extrhi[0, ib]
                    sc01 = self.scarhi[0, ib]
                    ss01 = self.ssarhi[0, ib]
                    as01 = self.asyrhi[0, ib]

                    ex02 = self.extrhi[1, ib]
                    sc02 = self.scarhi[1, ib]
                    ss02 = self.ssarhi[1, ib]
                    as02 = self.asyrhi[1, ib]

                    ex03 = self.extrhd[ih1, 0, ib] + rdrh * (
                        self.extrhd[ih2, 0, ib] - self.extrhd[ih1, 0, ib]
                    )
                    sc03 = self.scarhd[ih1, 0, ib] + rdrh * (
                        self.scarhd[ih2, 0, ib] - self.scarhd[ih1, 0, ib]
                    )
                    ss03 = self.ssarhd[ih1, 0, ib] + rdrh * (
                        self.ssarhd[ih2, 0, ib] - self.ssarhd[ih1, 0, ib]
                    )
                    as03 = self.asyrhd[ih1, 0, ib] + rdrh * (
                        self.asyrhd[ih2, 0, ib] - self.asyrhd[ih1, 0, ib]
                    )

                    ext1 = 0.17e-3 * ex01 + 0.4 * ex02 + 0.59983 * ex03
                    sca1 = 0.17e-3 * sc01 + 0.4 * sc02 + 0.59983 * sc03
                    ssa1 = (
                        0.17e-3 * ss01 * ex01
                        + 0.4 * ss02 * ex02
                        + 0.59983 * ss03 * ex03
                    )
                    asy1 = (
                        0.17e-3 * as01 * sc01
                        + 0.4 * as02 * sc02
                        + 0.59983 * as03 * sc03
                    )

                    self.tauae[kk, ib] = ext1 * 730.0 * self.delz[kk]
                    self.ssaae[kk, ib] = min(1.0, ssa1 / ext1)
                    self.asyae[kk, ib] = min(1.0, asy1 / sca1)

                    # --- compute aod from individual species' contribution (optional)
                    if ib == self.nv_aod - 1:
                        self.spcodp[0] = (
                            self.spcodp[0] + 0.17e-3 * ex01 * 730.0 * self.delz[kk]
                        )  # dust (inso)   #1
                        self.spcodp[1] = (
                            self.spcodp[1] + 0.4 * ex02 * 730.0 * self.delz[kk]
                        )  # black carbon  #2
                        self.spcodp[2] = (
                            self.spcodp[2] + 0.59983 * ex03 * 730.0 * self.delz[kk]
                        )  # water soluble #7

            elif idom == 1:
                # --- 1st domain - mixing layer

                for ib in range(self.NSWLWBD):
                    ext1 = 0.0
                    sca1 = 0.0
                    ssa1 = 0.0
                    asy1 = 0.0

                    for icmp in range(self.NCM):
                        ic = icmp
                        idx = self.idxspc[icmp] - 1

                        cm = self.cmix[icmp]
                        if cm > 0.0:

                            if ic <= self.NCM1 - 1:  # component withour rh dep
                                tt0 = cm * self.extrhi[ic, ib]
                                ext1 = ext1 + tt0
                                sca1 = sca1 + cm * self.scarhi[ic, ib]
                                ssa1 = (
                                    ssa1
                                    + cm * self.ssarhi[ic, ib] * self.extrhi[ic, ib]
                                )
                                asy1 = (
                                    asy1
                                    + cm * self.asyrhi[ic, ib] * self.scarhi[ic, ib]
                                )
                            else:  # component with rh dep
                                ic1 = ic - self.NCM1

                                ex00 = self.extrhd[ih1, ic1, ib] + rdrh * (
                                    self.extrhd[ih2, ic1, ib]
                                    - self.extrhd[ih1, ic1, ib]
                                )
                                sc00 = self.scarhd[ih1, ic1, ib] + rdrh * (
                                    self.scarhd[ih2, ic1, ib]
                                    - self.scarhd[ih1, ic1, ib]
                                )
                                ss00 = self.ssarhd[ih1, ic1, ib] + rdrh * (
                                    self.ssarhd[ih2, ic1, ib]
                                    - self.ssarhd[ih1, ic1, ib]
                                )
                                as00 = self.asyrhd[ih1, ic1, ib] + rdrh * (
                                    self.asyrhd[ih2, ic1, ib]
                                    - self.asyrhd[ih1, ic1, ib]
                                )

                                tt0 = cm * ex00
                                ext1 = ext1 + tt0
                                sca1 = sca1 + cm * sc00
                                ssa1 = ssa1 + cm * ss00 * ex00
                                asy1 = asy1 + cm * as00 * sc00

                            # --- compute aod from individual species' contribution (optional)
                            if ib == self.nv_aod - 1:
                                self.spcodp[idx] = (
                                    self.spcodp[idx]
                                    + tt0 * self.denn[0] * self.delz[kk]
                                )  # idx for dif species

                    self.tauae[kk, ib] = ext1 * self.denn[0] * self.delz[kk]
                    self.ssaae[kk, ib] = min(1.0, ssa1 / ext1)
                    self.asyae[kk, ib] = min(1.0, asy1 / sca1)

            elif idom == 2:
                # --- 2nd domain - mineral transport layers

                for ib in range(self.NSWLWBD):
                    self.tauae[kk, ib] = (
                        self.extrhi[5, ib] * self.denn[1] * self.delz[kk]
                    )
                    self.ssaae[kk, ib] = self.ssarhi[5, ib]
                    self.asyae[kk, ib] = self.asyrhi[5, ib]

                # --- compute aod from individual species' contribution (optional)
                self.spcodp[0] = (
                    self.spcodp[0] + self.tauae[kk, self.nv_aod - 1]
                )  # dust

            else:
                # --- domain index out off range, assume no aerosol

                for ib in range(self.NSWLWBD):
                    self.tauae[kk, ib] = 0.0
                    self.ssaae[kk, ib] = 1.0
                    self.asyae[kk, ib] = 0.0

        #
        # ===> ... smooth profile at domain boundaries
        #
        if self.ivflip == 0:  # input from toa to sfc

            for ib in range(self.NSWLWBD):
                for kk in range(1, self.NLAY):
                    if self.tauae[kk, ib] > 0.0:
                        ratio = self.tauae[kk - 1, ib] / self.tauae[kk, ib]
                    else:
                        ratio = 1.0

                    tt0 = self.tauae[kk, ib] + self.tauae[kk - 1, ib]
                    tt1 = 0.2 * tt0
                    tt2 = tt0 - tt1

                    if ratio > crt1:
                        self.tauae[kk, ib] = tt1
                        self.tauae[kk - 1, ib] = tt2

                    if ratio < crt2:
                        self.tauae[kk, ib] = tt2
                        self.tauae[kk - 1, ib] = tt1

        else:  # input from sfc to toa

            for ib in range(self.NSWLWBD):
                for kk in range(self.NLAY - 2, -1, -1):
                    if self.tauae[kk, ib] > 0.0:
                        ratio = self.tauae[kk + 1, ib] / self.tauae[kk, ib]
                    else:
                        ratio = 1.0

                    tt0 = self.tauae[kk, ib] + self.tauae[kk + 1, ib]
                    tt1 = 0.2 * tt0
                    tt2 = tt0 - tt1

                    if ratio > crt1:
                        self.tauae[kk, ib] = tt1
                        self.tauae[kk + 1, ib] = tt2

                    if ratio < crt2:
                        self.tauae[kk, ib] = tt2
                        self.tauae[kk + 1, ib] = tt1
