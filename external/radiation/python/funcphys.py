import numpy as np
from phys_const import *


def fpvs(t):
    # $$$     Subprogram Documentation Block
    #
    # Subprogram: fpvs         Compute saturation vapor pressure
    #   Author: N Phillips            w/NMC2X2   Date: 30 dec 82
    #
    # Abstract: Compute saturation vapor pressure from the temperature.
    #   A linear interpolation is done between values in a lookup table
    #   computed in gpvs. See documentation for fpvsx for details.
    #   Input values outside table range are reset to table extrema.
    #   The interpolation accuracy is almost 6 decimal places.
    #   On the Cray, fpvs is about 4 times faster than exact calculation.
    #   This function should be expanded inline in the calling routine.
    #
    # Program History Log:
    #   91-05-07  Iredell             made into inlinable function
    #   94-12-30  Iredell             expand table
    # 1999-03-01  Iredell             f90 module
    # 2001-02-26  Iredell             ice phase
    #
    # Usage:   pvs=fpvs(t)
    #
    #   Input argument list:
    #     t          Real(krealfp) temperature in Kelvin
    #
    #   Output argument list:
    #     fpvs       Real(krealfp) saturation vapor pressure in Pascals
    #
    # Attributes:
    #   Language: Fortran 90.
    #
    # $$$

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    xmin = 180.0
    xmax = 330.0
    nxpvs = 7501
    tbpvs = np.zeros(nxpvs)

    xinc = (xmax - xmin) / (nxpvs - 1)
    c2xpvs = 1.0 / xinc
    c1xpvs = 1.0 - xmin * c2xpvs

    for jx in range(nxpvs):
        x = xmin + jx * xinc
        tt = x
        tbpvs[jx] = fpvsx(tt)

    xj = min(max(c1xpvs + c2xpvs * t, 1.0), nxpvs)
    jx = int(min(xj, nxpvs - 1))
    fpvs = tbpvs[jx - 1] + (xj - jx) * (tbpvs[jx] - tbpvs[jx - 1])

    return fpvs


def fpvsx(t):
    # $$$     Subprogram Documentation Block
    #
    # Subprogram: fpvsx        Compute saturation vapor pressure
    #   Author: N Phillips            w/NMC2X2   Date: 30 dec 82
    #
    # Abstract: Exactly compute saturation vapor pressure from temperature.
    #   The saturation vapor pressure over either liquid and ice is computed
    #   over liquid for temperatures above the triple point,
    #   over ice for temperatures 20 degress below the triple point,
    #   and a linear combination of the two for temperatures in between.
    #   The water model assumes a perfect gas, constant specific heats
    #   for gas, liquid and ice, and neglects the volume of the condensate.
    #   The model does account for the variation of the latent heat
    #   of condensation and sublimation with temperature.
    #   The Clausius-Clapeyron equation is integrated from the triple point
    #   to get the formula
    #       pvsl=con_psat*(tr**xa)*exp(xb*(1.-tr))
    #   where tr is ttp/t and other values are physical constants.
    #   The reference for this computation is Emanuel(1994), pages 116-117.
    #   This function should be expanded inline in the calling routine.
    #
    # Program History Log:
    #   91-05-07  Iredell             made into inlinable function
    #   94-12-30  Iredell             exact computation
    # 1999-03-01  Iredell             f90 module
    # 2001-02-26  Iredell             ice phase
    #
    # Usage:   pvs=fpvsx(t)
    #
    #   Input argument list:
    #     t          Real(krealfp) temperature in Kelvin
    #
    #   Output argument list:
    #     fpvsx      Real(krealfp) saturation vapor pressure in Pascals
    #
    # Attributes:
    #   Language: Fortran 90.
    #
    # $$$

    tliq = con_ttp
    tice = con_ttp - 20.0
    dldtl = con_cvap - con_cliq
    heatl = con_hvap
    xponal = -dldtl / con_rv
    xponbl = -dldtl / con_rv + heatl / (con_rv * con_ttp)
    dldti = con_cvap - con_csol
    heati = con_hvap + con_hfus
    xponai = -dldti / con_rv
    xponbi = -dldti / con_rv + heati / (con_rv * con_ttp)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    tr = con_ttp / t
    if t >= tliq:
        fpvsx = con_psat * (tr ** xponal) * np.exp(xponbl * (1.0 - tr))
    elif t < tice:
        fpvsx = con_psat * (tr ** xponai) * np.exp(xponbi * (1.0 - tr))
    else:
        w = (t - tice) / (tliq - tice)
        pvl = con_psat * (tr ** xponal) * np.exp(xponbl * (1.0 - tr))
        pvi = con_psat * (tr ** xponai) * np.exp(xponbi * (1.0 - tr))
        fpvsx = w * pvl + (1.0 - w) * pvi

    return fpvsx
