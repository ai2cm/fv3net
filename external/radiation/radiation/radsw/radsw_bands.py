import numpy as np
from numba import jit
from .radsw_param import (
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

ngs = np.array(ngs)
ng = np.array(ng)
nspa = np.array(nspa)
nspb = np.array(nspb)

np.set_printoptions(precision=15)


@jit(nopython=True, cache=True)
def taumol16(
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
    selfref,
    forref,
    absa,
    absb,
    rayl,
):

    #  --- ... compute the optical depth by interpolating in ln(pressure),
    #          temperature, and appropriate species.  below laytrop, the water
    #          vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG16):
            taur[k, NS16 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[0] * colamt[k, 4]
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
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
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
@jit(nopython=True, cache=True)
def taumol17(
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
    selfref,
    forref,
    absa,
    absb,
    rayl,
):

    #  ------------------------------------------------------------------  !
    #     band 17:  3250-4000 cm-1 (low - h2o,co2; high - h2o,co2)         !
    #  ------------------------------------------------------------------  !
    #

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG17):
            taur[k, NS17 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[1] * colamt[k, 1]
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
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
            )

    for k in range(laytrop, nlay):
        speccomb = colamt[k, 0] + strrat[1] * colamt[k, 1]
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
@jit(nopython=True, cache=True)
def taumol18(
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
    selfref,
    forref,
    absa,
    absb,
    rayl,
):

    #  ------------------------------------------------------------------  !
    #     band 18:  4000-4650 cm-1 (low - h2o,ch4; high - ch4)             !
    #  ------------------------------------------------------------------  !
    #

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG18):
            taur[k, NS18 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[2] * colamt[k, 4]
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
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
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
@jit(nopython=True, cache=True)
def taumol19(
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
    selfref,
    forref,
    absa,
    absb,
    rayl,
):

    #  ------------------------------------------------------------------  !
    #     band 19:  4650-5150 cm-1 (low - h2o,co2; high - co2)             !
    #  ------------------------------------------------------------------  !
    #

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG19):
            taur[k, NS19 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[3] * colamt[k, 1]
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
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
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
@jit(nopython=True, cache=True)
def taumol20(
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
    selfref,
    forref,
    absa,
    absb,
    absch4,
    rayl,
):

    #  ------------------------------------------------------------------  !
    #     band 20:  5150-6150 cm-1 (low - h2o; high - h2o)                 !
    #  ------------------------------------------------------------------  !
    #

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
@jit(nopython=True, cache=True)
def taumol21(
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
    selfref,
    forref,
    absa,
    absb,
    rayl,
):

    #  ------------------------------------------------------------------  !
    #     band 21:  6150-7700 cm-1 (low - h2o,co2; high - h2o,co2)         !
    #  ------------------------------------------------------------------  !
    #

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG21):
            taur[k, NS21 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[5] * colamt[k, 1]
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
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
            )

    for k in range(laytrop, nlay):
        speccomb = colamt[k, 0] + strrat[5] * colamt[k, 1]
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
@jit(nopython=True, cache=True)
def taumol22(
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
    selfref,
    forref,
    absa,
    absb,
    rayl,
):

    #  ------------------------------------------------------------------  !
    #     band 22:  7700-8050 cm-1 (low - h2o,o2; high - o2)               !
    #  ------------------------------------------------------------------  !
    #

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
        speccomb = colamt[k, 0] + strrat[6] * colamt[k, 5]
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
@jit(nopython=True, cache=True)
def taumol23(
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
    selfref,
    forref,
    absa,
    rayl,
    givfac,
):

    #  ------------------------------------------------------------------  !
    #     band 23:  8050-12850 cm-1 (low - h2o; high - nothing)            !
    #  ------------------------------------------------------------------  !
    #

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
                * (forref[indf, j] + forfrac[k] * (forref[indfp, j] - forref[indf, j]))
            )

    for k in range(laytrop, nlay):
        for j in range(NG23):
            taug[k, NS23 + j] = 0.0

    return taug, taur


# The subroutine computes the optical depth in band 24:  12850-16000
# cm-1 (low - h2o,o2; high - o2)
@jit(nopython=True, cache=True)
def taumol24(
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
    selfref,
    forref,
    absa,
    absb,
    abso3a,
    abso3b,
    rayla,
    raylb,
):

    #  ------------------------------------------------------------------  !
    #     band 24:  12850-16000 cm-1 (low - h2o,o2; high - o2)             !
    #  ------------------------------------------------------------------  !
    #

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(laytrop):
        speccomb = colamt[k, 0] + strrat[8] * colamt[k, 5]
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
@jit(nopython=True, cache=True)
def taumol25(
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
    absa,
    abso3a,
    abso3b,
    rayl,
):

    #  ------------------------------------------------------------------  !
    #     band 25:  16000-22650 cm-1 (low - h2o; high - nothing)           !
    #  ------------------------------------------------------------------  !
    #

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
@jit(nopython=True, cache=True)
def taumol26(
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
    rayl,
):

    #  ------------------------------------------------------------------  !
    #     band 26:  22650-29000 cm-1 (low - nothing; high - nothing)       !
    #  ------------------------------------------------------------------  !
    #

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
@jit(nopython=True, cache=True)
def taumol27(
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
    absa,
    absb,
    rayl,
):

    #  ------------------------------------------------------------------  !
    #     band 27:  29000-38000 cm-1 (low - o3; high - o3)                 !
    #  ------------------------------------------------------------------  !
    #

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
@jit(nopython=True, cache=True)
def taumol28(
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
    absa,
    absb,
    rayl,
):

    #  ------------------------------------------------------------------  !
    #     band 28:  38000-50000 cm-1 (low - o3,o2; high - o3,o2)           !
    #  ------------------------------------------------------------------  !
    #

    #  --- ...  compute the optical depth by interpolating in ln(pressure),
    #           temperature, and appropriate species.  below laytrop, the water
    #           vapor self-continuum is interpolated (in temperature) separately.

    for k in range(nlay):
        tauray = colmol[k] * rayl

        for j in range(NG28):
            taur[k, NS28 + j] = tauray

    for k in range(laytrop):
        speccomb = colamt[k, 2] + strrat[12] * colamt[k, 5]
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
        speccomb = colamt[k, 2] + strrat[12] * colamt[k, 5]
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
@jit(nopython=True, cache=True)
def taumol29(
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
    forref,
    absa,
    absb,
    selfref,
    absh2o,
    absco2,
    rayl,
):

    #  ------------------------------------------------------------------  !
    #     band 29:  820-2600 cm-1 (low - h2o; high - co2)                  !
    #  ------------------------------------------------------------------  !
    #

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
