from typing import Optional, Tuple
import numpy as np
import numba
from ._constants import (
    retab,
    ccn_l,
    ccn_o,
    qcmin,
    grav,
    zvir,
    rdgas,
    pi,
    rhow,
    rewmin,
    rewmax,
    reimin,
    reimax,
    rermin,
    rermax,
    resmin,
    resmax,
    regmin,
    regmax,
    mur,
    mus,
    mug,
    edar,
    edas,
    edag,
    edbr,
    edbs,
    edbg,
)

liq_ice_combine = False  # combine all liquid water, combine all solid water
snow_graupel_combine = True  # combine snow and graupel
prog_ccn = False  # do prognostic ccn (yi ming's method)

rewflag = 1  # cloud water effective radius scheme (only ported option is 1)
# 1: Martin et al. (1994)
# 2: Martin et al. (1994), GFDL revision
# 3: Kiehl et al. (1994)
# 4: effective radius from PSD

reiflag = 4  # cloud ice effective radius scheme (only ported option is 4)
# 1: Heymsfield and Mcfarquhar (1996)
# 2: Donner et al. (1997)
# 3: Fu (2007)
# 4: Kristjansson et al. (2000)
# 5: Wyser (1998)
# 6: Sun and Rikus (1999), Sun (2001)
# 7: effective radius from PSD

rerflag = 1  # rain effective radius scheme (only ported option is 1)
# 1: effective radius from PSD

resflag = 1  # snow effective radius scheme (only ported option is 1)
# 1: effective radius from PSD

regflag = 1  # graupel effective radius scheme (only ported option is 1)
# 1: effective radius from PSD


@numba.jit
def cld_eff_rad(
    _is: int,
    ie: int,
    ks: int,
    ke: int,
    lsm: np.ndarray,
    p: np.ndarray,
    delp: np.ndarray,
    t: np.ndarray,
    qv: np.ndarray,
    qw: np.ndarray,
    qi: np.ndarray,
    qr: np.ndarray,
    qs: np.ndarray,
    qg: np.ndarray,
    qa: np.ndarray,
    cloud: np.ndarray,
    cnvw: Optional[np.ndarray] = None,
    cnvi: Optional[np.ndarray] = None,
    cnvc: Optional[np.ndarray] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    qmw = qw
    qmi = qi
    qmr = qr
    qms = qs
    qmg = qg

    # outputs
    qcw = np.zeros((ie, ke))
    qci = np.zeros((ie, ke))
    qcr = np.zeros((ie, ke))
    qcs = np.zeros((ie, ke))
    qcg = np.zeros((ie, ke))
    rew = np.zeros((ie, ke))
    rei = np.zeros((ie, ke))
    rer = np.zeros((ie, ke))
    res = np.zeros((ie, ke))
    reg = np.zeros((ie, ke))
    cld = cloud

    # -----------------------------------------------------------------------
    # merge convective cloud to total cloud
    # -----------------------------------------------------------------------
    if cnvw is not None:
        qmw = qmw + cnvw
    if cnvi is not None:
        qmi = qmi + cnvi
    if cnvc is not None:
        cld = cnvc + (1 - cnvc) * cld

    for i in range(_is, ie):
        for k in range(ks, ke):
            # -----------------------------------------------------------------------
            # combine liquid and solid phases
            # -----------------------------------------------------------------------
            if liq_ice_combine:
                qmw[i, k] = qmw[i, k] + qmr[i, k]
                qmr[i, k] = 0.0
                qmi[i, k] = qmi[i, k] + qms[i, k] + qmg[i, k]
                qms[i, k] = 0.0
                qmg[i, k] = 0.0

            # -----------------------------------------------------------------------
            # combine snow and graupel
            # -----------------------------------------------------------------------
            if snow_graupel_combine:
                qms[i, k] = qms[i, k] + qmg[i, k]
                qmg[i, k] = 0.0

            qmw[i, k] = max(qmw[i, k], qcmin)
            qmi[i, k] = max(qmi[i, k], qcmin)
            qmr[i, k] = max(qmr[i, k], qcmin)
            qms[i, k] = max(qms[i, k], qcmin)
            qmg[i, k] = max(qmg[i, k], qcmin)

            cld[i, k] = min(max(cld[i, k], 0.0), 1.0)

            mask = min(max(lsm[i], 0.0), 2.0)

            dpg = np.abs(delp[i, k]) / grav
            rho = p[i, k] / (rdgas * t[i, k] * (1.0 + zvir * qv[i, k]))

            if rewflag == 1:

                # -----------------------------------------------------------------------
                # cloud water (Martin et al. 1994)
                # -----------------------------------------------------------------------

                if prog_ccn:
                    # boucher and lohmann (1995)
                    ccnw = (1.0 - abs(mask - 1.0)) * (
                        10.0 ** 2.24 * (qa[i, k] * rho * 1.0e9) ** 0.257
                    ) + abs(mask - 1.0) * (
                        10.0 ** 2.06 * (qa[i, k] * rho * 1.0e9) ** 0.48
                    )
                else:
                    ccnw = ccn_o * abs(mask - 1.0) + ccn_l * (1.0 - abs(mask - 1.0))

                if qmw[i, k] > qcmin:
                    qcw[i, k] = dpg * qmw[i, k] * 1.0e3
                    rew[i, k] = (
                        np.exp(
                            1.0
                            / 3.0
                            * np.log((3.0 * qmw[i, k] * rho) / (4.0 * pi * rhow * ccnw))
                        )
                        * 1.0e4
                    )
                    rew[i, k] = max(rewmin, min(rewmax, rew[i, k]))
                else:
                    qcw[i, k] = 0.0
                    rew[i, k] = rewmin

            else:
                raise ValueError(
                    f"Only ported option for `rewflag` is 1, got {rewflag}."
                )

            if reiflag == 4:

                # -----------------------------------------------------------------------
                # cloud ice (Kristjansson et al. 2000)
                # -----------------------------------------------------------------------

                if qmi[i, k] > qcmin:
                    qci[i, k] = dpg * qmi[i, k] * 1.0e3
                    ind = min(max(int(t[i, k] - 136.0), 44), 138 - 1)
                    cor = t[i, k] - int(t[i, k])
                    rei[i, k] = retab[ind] * (1.0 - cor) + retab[ind] * cor
                    rei[i, k] = max(reimin, min(reimax, rei[i, k]))
                else:
                    qci[i, k] = 0.0
                    rei[i, k] = reimin
            else:
                raise ValueError(
                    f"Only ported option for `reiflag` is 4, got {reiflag}."
                )

            if rerflag == 1:

                # -----------------------------------------------------------------------
                # rain derived from PSD
                # -----------------------------------------------------------------------

                if qmr[i, k] > qcmin:
                    qcr[i, k] = dpg * qmr[i, k] * 1.0e3
                    der = calc_ed(qmr[i, k], rho, mur, eda=edar, edb=edbr)
                    rer[i, k] = 0.5 * der * 1.0e6
                    rer[i, k] = max(rermin, min(rermax, rer[i, k]))
                else:
                    qcr[i, k] = 0.0
                    rer[i, k] = rermin
            else:
                raise ValueError(
                    f"Only ported option for `rerflag` is 4, got {rerflag}."
                )

            if resflag == 1:

                # -----------------------------------------------------------------------
                # snow derived from PSD
                # -----------------------------------------------------------------------

                if qms[i, k] > qcmin:
                    qcs[i, k] = dpg * qms[i, k] * 1.0e3
                    des = calc_ed(qms[i, k], rho, mus, eda=edas, edb=edbs)
                    res[i, k] = 0.5 * des * 1.0e6
                    res[i, k] = max(resmin, min(resmax, res[i, k]))
                else:
                    qcs[i, k] = 0.0
                    res[i, k] = resmin
            else:
                raise ValueError(
                    f"Only ported option for `resflag` is 1, got {resflag}."
                )

            if regflag == 1:

                # -----------------------------------------------------------------------
                # graupel derived from PSD
                # -----------------------------------------------------------------------

                if qmg[i, k] > qcmin:
                    qcg[i, k] = dpg * qmg[i, k] * 1.0e3
                    deg = calc_ed(qmg[i, k], rho, mug, eda=edag, edb=edbg)
                    reg[i, k] = 0.5 * deg * 1.0e6
                    reg[i, k] = max(regmin, min(regmax, rer[i, k]))
                else:
                    qcg[i, k] = 0.0
                    reg[i, k] = regmin
            else:
                raise ValueError(
                    f"Only ported option for `regflag` is 4, got {regflag}."
                )

    return (
        qcw,
        qci,
        qcr,
        qcs,
        qcg,
        rew,
        rei,
        rer,
        res,
        reg,
        cld,
    )


def calc_ed(q: float, den: float, mu: float, eda: float, edb: float) -> float:
    """calculation of effective diameter (ed)"""
    ed = eda / edb * np.exp(1.0 / (mu + 3) * np.log(6 * den * q))
    return ed
