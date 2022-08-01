import numpy as np

from radlw.radlw_param import (
    ngptlw,
    ng01,
    ng02,
    ng03,
    ng04,
    ng05,
    ng06,
    ng07,
    ng08,
    ng09,
    ng10,
    ng11,
    ng12,
    ng13,
    ng14,
    ng15,
    ng16,
    ns02,
    ns03,
    ns04,
    ns05,
    ns06,
    ns07,
    ns08,
    ns09,
    ns10,
    ns11,
    ns12,
    ns13,
    ns14,
    ns15,
    ns16,
)

np.set_printoptions(precision=15)

# band 1:  10-350 cm-1 (low key - h2o; low minor - n2);
#  (high key - h2o; high minor - n2)
def taugb01(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    selfref,
    forref,
    ka_mn2,
    absa,
    absb,
    fracrefa,
    fracrefb,
    nspa,
    nspb,
    npts,
):
    #  ------------------------------------------------------------------  !
    #  written by eli j. mlawer, atmospheric & environmental research.     !
    #  revised by michael j. iacono, atmospheric & environmental research. !
    #                                                                      !
    #     band 1:  10-350 cm-1 (low key - h2o; low minor - n2)             !
    #                          (high key - h2o; high minor - n2)           !
    #                                                                      !
    #  compute the optical depth by interpolating in ln(pressure) and      !
    #  temperature.  below laytrop, the water vapor self-continuum and     !
    #  foreign continuum is interpolated (in temperature) separately.      !
    #  ------------------------------------------------------------------  !

    #  ---  minor gas mapping levels:
    #     lower - n2, p = 142.5490 mbar, t = 215.70 k
    #     upper - n2, p = 142.5490 mbar, t = 215.70 k

    #  --- ...  lower atmosphere loop

    taug = np.zeros((ngptlw, nlay, npts))
    fracs = np.zeros((ngptlw, nlay, npts))

    ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa[0]
    ind1 = (jp * 5 + (jt1 - 1)) * nspa[0]
    inds = indself - 1
    indf = indfor - 1
    indm = indminor - 1

    ind0 = ind0[:laytrop]
    ind1 = ind1[:laytrop]
    inds = inds[:laytrop]
    indf = indf[:laytrop]
    indm = indm[:laytrop]

    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indsp = inds + 1
    indfp = indf + 1
    indmp = indm + 1

    pp = pavel[:laytrop]
    scalen2 = colbrd[:laytrop] * scaleminorn2[:laytrop]
    corradj = np.where(pp < 250.0, 1.0 - 0.15 * (250.0 - pp) / 154.4, 1.0)
    # if pp < 250.0:
    #     corradj = 1.0 - 0.15 * (250.0-pp) / 154.4
    # else:
    #     corradj = 1.0

    for ig in range(ng01):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )
        taun2 = scalen2 * (
            ka_mn2[ig, indm]
            + minorfrac[:laytrop] * (ka_mn2[ig, indmp] - ka_mn2[ig, indm])
        )

        taug[ig, :laytrop] = corradj * (
            colamt[:laytrop, 0]
            * (
                fac00[:laytrop] * absa[ig, ind0]
                + fac10[:laytrop] * absa[ig, ind0p]
                + fac01[:laytrop] * absa[ig, ind1]
                + fac11[:laytrop] * absa[ig, ind1p]
            )
            + tauself
            + taufor
            + taun2
        )

        fracs[ig, :laytrop] = fracrefa[ig]

    #  --- ...  upper atmosphere loop

    ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb[0]
    ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb[0]
    indf = indfor - 1
    indm = indminor - 1

    ind0 = ind0[laytrop:nlay]
    ind1 = ind1[laytrop:nlay]
    indf = indf[laytrop:nlay]
    indm = indm[laytrop:nlay]

    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indfp = indf + 1
    indmp = indm + 1

    scalen2 = colbrd[laytrop:nlay] * scaleminorn2[laytrop:nlay]
    corradj = 1.0 - 0.15 * (pavel[laytrop:nlay] / 95.6)

    for ig in range(ng01):
        taufor = forfac[laytrop:nlay] * (
            forref[ig, indf]
            + forfrac[laytrop:nlay] * (forref[ig, indfp] - forref[ig, indf])
        )
        taun2 = scalen2 * (
            ka_mn2[ig, indm]
            + minorfrac[laytrop:nlay] * (ka_mn2[ig, indmp] - ka_mn2[ig, indm])
        )

        taug[ig, laytrop:nlay] = corradj * (
            colamt[laytrop:nlay, 0]
            * (
                fac00[laytrop:nlay] * absb[ig, ind0]
                + fac10[laytrop:nlay] * absb[ig, ind0p]
                + fac01[laytrop:nlay] * absb[ig, ind1]
                + fac11[laytrop:nlay] * absb[ig, ind1p]
            )
            + taufor
            + taun2
        )

        fracs[ig, laytrop:nlay] = fracrefb[ig]

    return taug, fracs


# Band 2:  350-500 cm-1 (low key - h2o; high key - h2o)
def taugb02(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    selfref,
    forref,
    absa,
    absb,
    fracrefa,
    fracrefb,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 2:  350-500 cm-1 (low key - h2o; high key - h2o)            !
    #  ------------------------------------------------------------------  !
    #
    # ===> ...  begin here
    #
    #  --- ...  lower atmosphere loop

    ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa[1]
    ind1 = (jp * 5 + (jt1 - 1)) * nspa[1]
    inds = indself - 1
    indf = indfor - 1

    ind0 = ind0[:laytrop]
    ind1 = ind1[:laytrop]
    inds = inds[:laytrop]
    indf = indf[:laytrop]

    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indsp = inds + 1
    indfp = indf + 1

    corradj = 1.0 - 0.05 * (pavel[:laytrop] - 100.0) / 900.0

    for ig in range(ng02):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )

        taug[ns02 + ig, :laytrop] = corradj * (
            colamt[:laytrop, 0]
            * (
                fac00[:laytrop] * absa[ig, ind0]
                + fac10[:laytrop] * absa[ig, ind0p]
                + fac01[:laytrop] * absa[ig, ind1]
                + fac11[:laytrop] * absa[ig, ind1p]
            )
            + +tauself
            + taufor
        )

        fracs[ns02 + ig, :laytrop] = fracrefa[ig]

    #  --- ...  upper atmosphere loop

    ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb[1]
    ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb[1]
    indf = indfor - 1

    ind0 = ind0[laytrop:nlay]
    ind1 = ind1[laytrop:nlay]
    indf = indf[laytrop:nlay]

    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indfp = indf + 1

    for ig in range(ng02):
        taufor = forfac[laytrop:nlay] * (
            forref[ig, indf]
            + forfrac[laytrop:nlay] * (forref[ig, indfp] - forref[ig, indf])
        )

        taug[ns02 + ig, laytrop:nlay] = (
            colamt[laytrop:nlay, 0]
            * (
                fac00[laytrop:nlay] * absb[ig, ind0]
                + fac10[laytrop:nlay] * absb[ig, ind0p]
                + fac01[laytrop:nlay] * absb[ig, ind1]
                + fac11[laytrop:nlay] * absb[ig, ind1p]
            )
            + taufor
        )

        fracs[ns02 + ig, laytrop:nlay] = fracrefb[ig]

    return taug, fracs, tauself


# Band 3:  500-630 cm-1 (low key - h2o,co2; low minor - n2o);
#                        (high key - h2o,co2; high minor - n2o)
def taugb03(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    tauself,
    chi_mls,
    selfref,
    forref,
    ka_mn2o,
    kb_mn2o,
    absa,
    absb,
    fracrefa,
    fracrefb,
    oneminus,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 3:  500-630 cm-1 (low key - h2o,co2; low minor - n2o)       !
    #                           (high key - h2o,co2; high minor - n2o)     !
    #  ------------------------------------------------------------------  !

    #
    # ===> ...  begin here
    #
    #  --- ...  minor gas mapping levels:
    #     lower - n2o, p = 706.272 mbar, t = 278.94 k
    #     upper - n2o, p = 95.58 mbar, t = 215.7 k

    refrat_planck_a = chi_mls[0, 8] / chi_mls[1, 8]  # P = 212.725 mb
    refrat_planck_b = chi_mls[0, 12] / chi_mls[1, 12]  # P = 95.58   mb
    refrat_m_a = chi_mls[0, 2] / chi_mls[1, 2]  # P = 706.270 mb
    refrat_m_b = chi_mls[0, 12] / chi_mls[1, 12]  # P = 95.58   mb

    #  --- ...  lower atmosphere loop

    speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 0] * colamt[:laytrop, 1]
    specparm = colamt[:laytrop, 0] / speccomb
    specmult = 8.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[2] + js - 1

    speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 1] * colamt[:laytrop, 1]
    specparm1 = colamt[:laytrop, 0] / speccomb1
    specmult1 = 8.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[2] + js1 - 1

    speccomb_mn2o = colamt[:laytrop, 0] + refrat_m_a * colamt[:laytrop, 1]
    specparm_mn2o = colamt[:laytrop, 0] / speccomb_mn2o
    specmult_mn2o = 8.0 * np.minimum(specparm_mn2o, oneminus)
    jmn2o = 1 + specmult_mn2o.astype(np.int32) - 1
    fmn2o = specmult_mn2o % 1.0

    speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 1]
    specparm_planck = colamt[:laytrop, 0] / speccomb_planck
    specmult_planck = 8.0 * np.minimum(specparm_planck, oneminus)
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indm = indminor[:laytrop] - 1
    indsp = inds + 1
    indfp = indf + 1
    indmp = indm + 1
    jmn2op = jmn2o + 1
    jplp = jpl + 1

    #  --- ...  in atmospheres where the amount of n2O is too great to be considered
    #           a minor species, adjust the column amount of n2O by an empirical factor
    #           to obtain the proper contribution.

    p = coldry[:laytrop] * chi_mls[3, jp[:laytrop]]
    ratn2o = colamt[:laytrop, 3] / p

    adjcoln2o = np.where(
        ratn2o > 1.5, (0.5 + (ratn2o - 0.5) ** 0.65) * p, colamt[:laytrop, 3]
    )

    p = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(specparm > 0.875, -fs, 0)
    p = np.where(p == 0, 0, p)

    p4 = np.where(specparm < 0.125, p ** 4, 0) + np.where(specparm > 0.875, p ** 4, 0)
    p4 = np.where(p4 == 0, 0, p4)

    fk0 = np.where(specparm < 0.125, p4, 0) + np.where(specparm > 0.875, p ** 4, 0)
    fk0 = np.where(fk0 == 0, 1.0 - fs, fk0)

    fk1 = np.where(specparm < 0.125, 1.0 - p - 2.0 * p4, 0) + np.where(
        specparm > 0.875, 1.0 - p - 2.0 * p4, 0
    )
    fk1 = np.where(fk1 == 0, fs, fk1)

    fk2 = np.where(specparm < 0.125, p + p4, 0) + np.where(specparm > 0.875, p + p4, 0)
    fk2 = np.where(fk2 == 0, 0.0, fk2)

    id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
        specparm > 0.875, ind0 + 1, 0
    )
    id000 = np.where(id000 == 0, ind0, id000)

    id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
        specparm > 0.875, ind0 + 10, 0
    )
    id010 = np.where(id010 == 0, ind0 + 9, id010)

    id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
        specparm > 0.875, ind0, 0
    )
    id100 = np.where(id100 == 0, ind0 + 1, id100)

    id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
        specparm > 0.875, ind0 + 9, 0
    )
    id110 = np.where(id110 == 0, ind0 + 10, id110)

    id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
        specparm > 0.875, ind0 - 1, 0
    )
    id200 = np.where(id200 == 0, ind0, id200)

    id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
        specparm > 0.875, ind0 + 8, 0
    )
    id210 = np.where(id210 == 0, ind0, id210)

    fac000 = fk0 * fac00[:laytrop]
    fac100 = fk1 * fac00[:laytrop]
    fac200 = fk2 * fac00[:laytrop]
    fac010 = fk0 * fac10[:laytrop]
    fac110 = fk1 * fac10[:laytrop]
    fac210 = fk2 * fac10[:laytrop]

    p = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(specparm1 > 0.875, -fs1, 0)
    p = np.where(p == 0, 0, p)

    p4 = np.where(specparm1 < 0.125, p ** 4, 0) + np.where(specparm1 > 0.875, p ** 4, 0)
    p4 = np.where(p4 == 0, 0, p4)

    fk0 = np.where(specparm1 < 0.125, p4, 0) + np.where(specparm1 > 0.875, p ** 4, 0)
    fk0 = np.where(fk0 == 0, 1.0 - fs1, fk0)

    fk1 = np.where(specparm1 < 0.125, 1.0 - p - 2.0 * p4, 0) + np.where(
        specparm1 > 0.875, 1.0 - p - 2.0 * p4, 0
    )
    fk1 = np.where(fk1 == 0, fs1, fk1)

    fk2 = np.where(specparm1 < 0.125, p + p4, 0) + np.where(
        specparm1 > 0.875, p + p4, 0
    )
    fk2 = np.where(fk2 == 0, 0.0, fk2)

    id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
        specparm1 > 0.875, ind1 + 1, 0
    )
    id001 = np.where(id001 == 0, ind1, id001)

    id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
        specparm1 > 0.875, ind1 + 10, 0
    )
    id011 = np.where(id011 == 0, ind1 + 9, id011)

    id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
        specparm1 > 0.875, ind1, 0
    )
    id101 = np.where(id101 == 0, ind1 + 1, id101)

    id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
        specparm1 > 0.875, ind1 + 9, 0
    )
    id111 = np.where(id111 == 0, ind1 + 10, id111)

    id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
        specparm1 > 0.875, ind1 - 1, 0
    )
    id201 = np.where(id201 == 0, ind1, id201)

    id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
        specparm1 > 0.875, ind1 + 8, 0
    )
    id211 = np.where(id211 == 0, ind1, id211)

    fac001 = fk0 * fac01[:laytrop]
    fac101 = fk1 * fac01[:laytrop]
    fac201 = fk2 * fac01[:laytrop]
    fac011 = fk0 * fac11[:laytrop]
    fac111 = fk1 * fac11[:laytrop]
    fac211 = fk2 * fac11[:laytrop]

    for ig in range(ng03):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )
        n2om1 = ka_mn2o[ig, jmn2o, indm] + fmn2o * (
            ka_mn2o[ig, jmn2op, indm] - ka_mn2o[ig, jmn2o, indm]
        )
        n2om2 = ka_mn2o[ig, jmn2o, indmp] + fmn2o * (
            ka_mn2o[ig, jmn2op, indmp] - ka_mn2o[ig, jmn2o, indmp]
        )
        absn2o = n2om1 + minorfrac[:laytrop] * (n2om2 - n2om1)

        tau_major = speccomb * (
            fac000 * absa[ig, id000]
            + fac010 * absa[ig, id010]
            + fac100 * absa[ig, id100]
            + fac110 * absa[ig, id110]
            + fac200 * absa[ig, id200]
            + fac210 * absa[ig, id210]
        )

        tau_major1 = speccomb1 * (
            fac001 * absa[ig, id001]
            + fac011 * absa[ig, id011]
            + fac101 * absa[ig, id101]
            + fac111 * absa[ig, id111]
            + fac201 * absa[ig, id201]
            + fac211 * absa[ig, id211]
        )

        taug[ns03 + ig, :laytrop] = (
            tau_major + tau_major1 + tauself + taufor + adjcoln2o * absn2o
        )

        fracs[ns03 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
            fracrefa[ig, jplp] - fracrefa[ig, jpl]
        )

    #  --- ...  upper atmosphere loop

    speccomb = (
        colamt[laytrop:nlay, 0] + rfrate[laytrop:nlay, 0, 0] * colamt[laytrop:nlay, 1]
    )
    specparm = colamt[laytrop:nlay, 0] / speccomb
    specmult = 4.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * nspb[2] + js - 1

    speccomb1 = (
        colamt[laytrop:nlay, 0] + rfrate[laytrop:nlay, 0, 1] * colamt[laytrop:nlay, 1]
    )
    specparm1 = colamt[laytrop:nlay, 0] / speccomb1
    specmult1 = 4.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * nspb[2] + js1 - 1

    speccomb_mn2o = colamt[laytrop:nlay, 0] + refrat_m_b * colamt[laytrop:nlay, 1]
    specparm_mn2o = colamt[laytrop:nlay, 0] / speccomb_mn2o
    specmult_mn2o = 4.0 * np.minimum(specparm_mn2o, oneminus)
    jmn2o = 1 + specmult_mn2o.astype(np.int32) - 1
    fmn2o = specmult_mn2o % 1.0

    speccomb_planck = (
        colamt[laytrop:nlay, 0] + refrat_planck_b * colamt[laytrop:nlay, 1]
    )
    specparm_planck = colamt[laytrop:nlay, 0] / speccomb_planck
    specmult_planck = 4.0 * np.minimum(specparm_planck, oneminus)
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0

    indf = indfor[laytrop:nlay] - 1
    indm = indminor[laytrop:nlay] - 1
    indfp = indf + 1
    indmp = indm + 1
    jmn2op = jmn2o + 1
    jplp = jpl + 1

    id000 = ind0
    id010 = ind0 + 5
    id100 = ind0 + 1
    id110 = ind0 + 6
    id001 = ind1
    id011 = ind1 + 5
    id101 = ind1 + 1
    id111 = ind1 + 6

    #  --- ...  in atmospheres where the amount of n2o is too great to be considered
    #           a minor species, adjust the column amount of N2O by an empirical factor
    #           to obtain the proper contribution.

    p = coldry[laytrop:nlay] * chi_mls[3, jp[laytrop:nlay]]
    ratn2o = colamt[laytrop:nlay, 3] / p
    adjcoln2o = np.where(
        ratn2o > 1.5, (0.5 + (ratn2o - 0.5) ** 0.65) * p, colamt[laytrop:nlay, 3]
    )
    # if ratn2o > 1.5:
    #     adjfac = 0.5 + (ratn2o - 0.5)**0.65
    #     adjcoln2o = adjfac * p
    # else:
    #     adjcoln2o = colamt[laytrop:nlay, 3]

    fk0 = 1.0 - fs
    fk1 = fs
    fac000 = fk0 * fac00[laytrop:nlay]
    fac010 = fk0 * fac10[laytrop:nlay]
    fac100 = fk1 * fac00[laytrop:nlay]
    fac110 = fk1 * fac10[laytrop:nlay]

    fk0 = 1.0 - fs1
    fk1 = fs1
    fac001 = fk0 * fac01[laytrop:nlay]
    fac011 = fk0 * fac11[laytrop:nlay]
    fac101 = fk1 * fac01[laytrop:nlay]
    fac111 = fk1 * fac11[laytrop:nlay]

    for ig in range(ng03):
        taufor = forfac[laytrop:nlay] * (
            forref[ig, indf]
            + forfrac[laytrop:nlay] * (forref[ig, indfp] - forref[ig, indf])
        )
        n2om1 = kb_mn2o[ig, jmn2o, indm] + fmn2o * (
            kb_mn2o[ig, jmn2op, indm] - kb_mn2o[ig, jmn2o, indm]
        )
        n2om2 = kb_mn2o[ig, jmn2o, indmp] + fmn2o * (
            kb_mn2o[ig, jmn2op, indmp] - kb_mn2o[ig, jmn2o, indmp]
        )
        absn2o = n2om1 + minorfrac[laytrop:nlay] * (n2om2 - n2om1)

        tau_major = speccomb * (
            fac000 * absb[ig, id000]
            + fac010 * absb[ig, id010]
            + fac100 * absb[ig, id100]
            + fac110 * absb[ig, id110]
        )

        tau_major1 = speccomb1 * (
            fac001 * absb[ig, id001]
            + fac011 * absb[ig, id011]
            + fac101 * absb[ig, id101]
            + fac111 * absb[ig, id111]
        )

        taug[ns03 + ig, laytrop:nlay] = (
            tau_major + tau_major1 + taufor + adjcoln2o * absn2o
        )

        fracs[ns03 + ig, laytrop:nlay] = fracrefb[ig, jpl] + fpl * (
            fracrefb[ig, jplp] - fracrefb[ig, jpl]
        )

    return taug, fracs


# Band 4:  630-700 cm-1 (low key - h2o,co2; high key - o3,co2)
# ----------------------------------
def taugb04(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    chi_mls,
    selfref,
    forref,
    absa,
    absb,
    fracrefa,
    fracrefb,
    oneminus,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 4:  630-700 cm-1 (low key - h2o,co2; high key - o3,co2)     !
    #  ------------------------------------------------------------------  !
    #
    # ===> ...  begin here
    #

    refrat_planck_a = chi_mls[0, 10] / chi_mls[1, 10]  # P = 142.5940 mb
    refrat_planck_b = chi_mls[2, 12] / chi_mls[1, 12]  # P = 95.58350 mb

    #  --- ...  lower atmosphere loop
    speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 0] * colamt[:laytrop, 1]
    specparm = colamt[:laytrop, 0] / speccomb
    specmult = 8.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[3] + js - 1

    speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 1] * colamt[:laytrop, 1]
    specparm1 = colamt[:laytrop, 0] / speccomb1
    specmult1 = 8.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[3] + js1 - 1

    speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 1]
    specparm_planck = colamt[:laytrop, 0] / speccomb_planck
    specmult_planck = 8.0 * np.minimum(specparm_planck, oneminus)
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indsp = inds + 1
    indfp = indf + 1
    jplp = jpl + 1

    p = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(specparm > 0.875, -fs, 0)
    p = np.where(p == 0, 0, p)

    p4 = np.where(specparm < 0.125, p ** 4, 0) + np.where(specparm > 0.875, p ** 4, 0)
    p4 = np.where(p4 == 0, 0, p4)

    fk0 = np.where(specparm < 0.125, p4, 0) + np.where(specparm > 0.875, p ** 4, 0)
    fk0 = np.where(fk0 == 0, 1.0 - fs, fk0)

    fk1 = np.where(specparm < 0.125, 1.0 - p - 2.0 * p4, 0) + np.where(
        specparm > 0.875, 1.0 - p - 2.0 * p4, 0
    )
    fk1 = np.where(fk1 == 0, fs, fk1)

    fk2 = np.where(specparm < 0.125, p + p4, 0) + np.where(specparm > 0.875, p + p4, 0)
    fk2 = np.where(fk2 == 0, 0.0, fk2)

    id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
        specparm > 0.875, ind0 + 1, 0
    )
    id000 = np.where(id000 == 0, ind0, id000)

    id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
        specparm > 0.875, ind0 + 10, 0
    )
    id010 = np.where(id010 == 0, ind0 + 9, id010)

    id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
        specparm > 0.875, ind0, 0
    )
    id100 = np.where(id100 == 0, ind0 + 1, id100)

    id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
        specparm > 0.875, ind0 + 9, 0
    )
    id110 = np.where(id110 == 0, ind0 + 10, id110)

    id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
        specparm > 0.875, ind0 - 1, 0
    )
    id200 = np.where(id200 == 0, ind0, id200)

    id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
        specparm > 0.875, ind0 + 8, 0
    )
    id210 = np.where(id210 == 0, ind0, id210)

    fac000 = fk0 * fac00[:laytrop]
    fac100 = fk1 * fac00[:laytrop]
    fac200 = fk2 * fac00[:laytrop]
    fac010 = fk0 * fac10[:laytrop]
    fac110 = fk1 * fac10[:laytrop]
    fac210 = fk2 * fac10[:laytrop]

    p = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(specparm1 > 0.875, -fs1, 0)
    p = np.where(p == 0, 0, p)

    p4 = np.where(specparm1 < 0.125, p ** 4, 0) + np.where(specparm1 > 0.875, p ** 4, 0)
    p4 = np.where(p4 == 0, 0, p4)

    fk0 = np.where(specparm1 < 0.125, p4, 0) + np.where(specparm1 > 0.875, p ** 4, 0)
    fk0 = np.where(fk0 == 0, 1.0 - fs1, fk0)

    fk1 = np.where(specparm1 < 0.125, 1.0 - p - 2.0 * p4, 0) + np.where(
        specparm1 > 0.875, 1.0 - p - 2.0 * p4, 0
    )
    fk1 = np.where(fk1 == 0, fs1, fk1)

    fk2 = np.where(specparm1 < 0.125, p + p4, 0) + np.where(
        specparm1 > 0.875, p + p4, 0
    )
    fk2 = np.where(fk2 == 0, 0.0, fk2)

    id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
        specparm1 > 0.875, ind1 + 1, 0
    )
    id001 = np.where(id001 == 0, ind1, id001)

    id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
        specparm1 > 0.875, ind1 + 10, 0
    )
    id011 = np.where(id011 == 0, ind1 + 9, id011)

    id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
        specparm1 > 0.875, ind1, 0
    )
    id101 = np.where(id101 == 0, ind1 + 1, id101)

    id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
        specparm1 > 0.875, ind1 + 9, 0
    )
    id111 = np.where(id111 == 0, ind1 + 10, id111)

    id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
        specparm1 > 0.875, ind1 - 1, 0
    )
    id201 = np.where(id201 == 0, ind1, id201)

    id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
        specparm1 > 0.875, ind1 + 8, 0
    )
    id211 = np.where(id211 == 0, ind1, id211)

    fac001 = fk0 * fac01[:laytrop]
    fac101 = fk1 * fac01[:laytrop]
    fac201 = fk2 * fac01[:laytrop]
    fac011 = fk0 * fac11[:laytrop]
    fac111 = fk1 * fac11[:laytrop]
    fac211 = fk2 * fac11[:laytrop]

    for ig in range(ng04):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )

        tau_major = speccomb * (
            fac000 * absa[ig, id000]
            + fac010 * absa[ig, id010]
            + fac100 * absa[ig, id100]
            + fac110 * absa[ig, id110]
            + fac200 * absa[ig, id200]
            + fac210 * absa[ig, id210]
        )

        tau_major1 = speccomb1 * (
            fac001 * absa[ig, id001]
            + fac011 * absa[ig, id011]
            + fac101 * absa[ig, id101]
            + fac111 * absa[ig, id111]
            + fac201 * absa[ig, id201]
            + fac211 * absa[ig, id211]
        )

        taug[ns04 + ig, :laytrop] = tau_major + tau_major1 + tauself + taufor

        fracs[ns04 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
            fracrefa[ig, jplp] - fracrefa[ig, jpl]
        )

    #  --- ...  upper atmosphere loop
    speccomb = (
        colamt[laytrop:nlay, 2] + rfrate[laytrop:nlay, 5, 0] * colamt[laytrop:nlay, 1]
    )
    specparm = colamt[laytrop:nlay, 2] / speccomb
    specmult = 4.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * nspb[3] + js - 1

    speccomb1 = (
        colamt[laytrop:nlay, 2] + rfrate[laytrop:nlay, 5, 1] * colamt[laytrop:nlay, 1]
    )
    specparm1 = colamt[laytrop:nlay, 2] / speccomb1
    specmult1 = 4.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * nspb[3] + js1 - 1

    speccomb_planck = (
        colamt[laytrop:nlay, 2] + refrat_planck_b * colamt[laytrop:nlay, 1]
    )
    specparm_planck = colamt[laytrop:nlay, 2] / speccomb_planck
    specmult_planck = 4.0 * np.minimum(specparm_planck, oneminus)
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0
    jplp = jpl + 1

    id000 = ind0
    id010 = ind0 + 5
    id100 = ind0 + 1
    id110 = ind0 + 6
    id001 = ind1
    id011 = ind1 + 5
    id101 = ind1 + 1
    id111 = ind1 + 6

    fk0 = 1.0 - fs
    fk1 = fs
    fac000 = fk0 * fac00[laytrop:nlay]
    fac010 = fk0 * fac10[laytrop:nlay]
    fac100 = fk1 * fac00[laytrop:nlay]
    fac110 = fk1 * fac10[laytrop:nlay]

    fk0 = 1.0 - fs1
    fk1 = fs1
    fac001 = fk0 * fac01[laytrop:nlay]
    fac011 = fk0 * fac11[laytrop:nlay]
    fac101 = fk1 * fac01[laytrop:nlay]
    fac111 = fk1 * fac11[laytrop:nlay]

    for ig in range(ng04):
        tau_major = speccomb * (
            fac000 * absb[ig, id000]
            + fac010 * absb[ig, id010]
            + fac100 * absb[ig, id100]
            + fac110 * absb[ig, id110]
        )
        tau_major1 = speccomb1 * (
            fac001 * absb[ig, id001]
            + fac011 * absb[ig, id011]
            + fac101 * absb[ig, id101]
            + fac111 * absb[ig, id111]
        )

        taug[ns04 + ig, laytrop:nlay] = tau_major + tau_major1

        fracs[ns04 + ig, laytrop:nlay] = fracrefb[ig, jpl] + fpl * (
            fracrefb[ig, jplp] - fracrefb[ig, jpl]
        )

        # empirical modification to code to improve stratospheric cooling rates
        # for co2. revised to apply weighting for g-point reduction in this band.

    taug[ns04 + 7, laytrop:nlay] = taug[ns04 + 7, laytrop:nlay] * 0.92
    taug[ns04 + 8, laytrop:nlay] = taug[ns04 + 8, laytrop:nlay] * 0.88
    taug[ns04 + 9, laytrop:nlay] = taug[ns04 + 9, laytrop:nlay] * 1.07
    taug[ns04 + 10, laytrop:nlay] = taug[ns04 + 10, laytrop:nlay] * 1.1
    taug[ns04 + 11, laytrop:nlay] = taug[ns04 + 11, laytrop:nlay] * 0.99
    taug[ns04 + 12, laytrop:nlay] = taug[ns04 + 12, laytrop:nlay] * 0.88
    taug[ns04 + 13, laytrop:nlay] = taug[ns04 + 13, laytrop:nlay] * 0.943

    return taug, fracs


# Band 5:  700-820 cm-1 (low key - h2o,co2; low minor - o3, ccl4)
#                       (high key - o3,co2)
def taugb05(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    chi_mls,
    selfref,
    forref,
    absa,
    absb,
    fracrefa,
    fracrefb,
    ka_mo3,
    ccl4,
    oneminus,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 5:  700-820 cm-1 (low key - h2o,co2; low minor - o3, ccl4)  !
    #                           (high key - o3,co2)                        !
    #  ------------------------------------------------------------------  !
    #
    # ===> ...  begin here
    #
    #  --- ...  minor gas mapping level :
    #     lower - o3, p = 317.34 mbar, t = 240.77 k
    #     lower - ccl4

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower/upper atmosphere.

    refrat_planck_a = chi_mls[0, 4] / chi_mls[1, 4]  # P = 473.420 mb
    refrat_planck_b = chi_mls[2, 42] / chi_mls[1, 42]  # P = 0.2369  mb
    refrat_m_a = chi_mls[0, 6] / chi_mls[1, 6]  # P = 317.348 mb

    #  --- ...  lower atmosphere loop

    speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 0] * colamt[:laytrop, 1]
    specparm = colamt[:laytrop, 0] / speccomb
    specmult = 8.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[4] + js - 1

    speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 1] * colamt[:laytrop, 1]
    specparm1 = colamt[:laytrop, 0] / speccomb1
    specmult1 = 8.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[4] + js1 - 1

    speccomb_mo3 = colamt[:laytrop, 0] + refrat_m_a * colamt[:laytrop, 1]
    specparm_mo3 = colamt[:laytrop, 0] / speccomb_mo3
    specmult_mo3 = 8.0 * np.minimum(specparm_mo3, oneminus)
    jmo3 = 1 + specmult_mo3.astype(np.int32) - 1
    fmo3 = specmult_mo3 % 1.0

    speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 1]
    specparm_planck = colamt[:laytrop, 0] / speccomb_planck
    specmult_planck = 8.0 * np.minimum(specparm_planck, oneminus)
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indm = indminor[:laytrop] - 1
    indsp = inds + 1
    indfp = indf + 1
    indmp = indm + 1
    jplp = jpl + 1
    jmo3p = jmo3 + 1

    p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(specparm > 0.875, -fs, 0)
    p0 = np.where(p0 == 0, 0, p0)

    p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
        specparm > 0.875, p0 ** 4, 0
    )
    p40 = np.where(p40 == 0, 0, p40)

    fk00 = np.where(specparm < 0.125, p40, 0) + np.where(specparm > 0.875, p0 ** 4, 0)
    fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

    fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
        specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
    )
    fk10 = np.where(fk10 == 0, fs, fk10)

    fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
        specparm > 0.875, p0 + p40, 0
    )
    fk20 = np.where(fk20 == 0, 0.0, fk20)

    id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
        specparm > 0.875, ind0 + 1, 0
    )
    id000 = np.where(id000 == 0, ind0, id000)

    id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
        specparm > 0.875, ind0 + 10, 0
    )
    id010 = np.where(id010 == 0, ind0 + 9, id010)

    id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
        specparm > 0.875, ind0, 0
    )
    id100 = np.where(id100 == 0, ind0 + 1, id100)

    id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
        specparm > 0.875, ind0 + 9, 0
    )
    id110 = np.where(id110 == 0, ind0 + 10, id110)

    id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
        specparm > 0.875, ind0 - 1, 0
    )
    id200 = np.where(id200 == 0, ind0, id200)

    id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
        specparm > 0.875, ind0 + 8, 0
    )
    id210 = np.where(id210 == 0, ind0, id210)

    fac000 = fk00 * fac00[:laytrop]
    fac100 = fk10 * fac00[:laytrop]
    fac200 = fk20 * fac00[:laytrop]
    fac010 = fk00 * fac10[:laytrop]
    fac110 = fk10 * fac10[:laytrop]
    fac210 = fk20 * fac10[:laytrop]

    p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
        specparm1 > 0.875, -fs1, 0
    )
    p1 = np.where(p1 == 0, 0, p1)

    p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
        specparm1 > 0.875, p1 ** 4, 0
    )
    p41 = np.where(p41 == 0, 0, p41)

    fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(specparm1 > 0.875, p1 ** 4, 0)
    fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

    fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
        specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
    )
    fk11 = np.where(fk11 == 0, fs1, fk11)

    fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
        specparm1 > 0.875, p1 + p41, 0
    )
    fk21 = np.where(fk21 == 0, 0.0, fk21)

    id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
        specparm1 > 0.875, ind1 + 1, 0
    )
    id001 = np.where(id001 == 0, ind1, id001)

    id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
        specparm1 > 0.875, ind1 + 10, 0
    )
    id011 = np.where(id011 == 0, ind1 + 9, id011)

    id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
        specparm1 > 0.875, ind1, 0
    )
    id101 = np.where(id101 == 0, ind1 + 1, id101)

    id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
        specparm1 > 0.875, ind1 + 9, 0
    )
    id111 = np.where(id111 == 0, ind1 + 10, id111)

    id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
        specparm1 > 0.875, ind1 - 1, 0
    )
    id201 = np.where(id201 == 0, ind1, id201)

    id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
        specparm1 > 0.875, ind1 + 8, 0
    )
    id211 = np.where(id211 == 0, ind1, id211)

    fac001 = fk01 * fac01[:laytrop]
    fac101 = fk11 * fac01[:laytrop]
    fac201 = fk21 * fac01[:laytrop]
    fac011 = fk01 * fac11[:laytrop]
    fac111 = fk11 * fac11[:laytrop]
    fac211 = fk21 * fac11[:laytrop]

    for ig in range(ng05):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )
        o3m1 = ka_mo3[ig, jmo3, indm] + fmo3 * (
            ka_mo3[ig, jmo3p, indm] - ka_mo3[ig, jmo3, indm]
        )
        o3m2 = ka_mo3[ig, jmo3, indmp] + fmo3 * (
            ka_mo3[ig, jmo3p, indmp] - ka_mo3[ig, jmo3, indmp]
        )
        abso3 = o3m1 + minorfrac[:laytrop] * (o3m2 - o3m1)

        taug[ns05 + ig, :laytrop] = (
            speccomb
            * (
                fac000 * absa[ig, id000]
                + fac010 * absa[ig, id010]
                + fac100 * absa[ig, id100]
                + fac110 * absa[ig, id110]
                + fac200 * absa[ig, id200]
                + fac210 * absa[ig, id210]
            )
            + speccomb1
            * (
                fac001 * absa[ig, id001]
                + fac011 * absa[ig, id011]
                + fac101 * absa[ig, id101]
                + fac111 * absa[ig, id111]
                + fac201 * absa[ig, id201]
                + fac211 * absa[ig, id211]
            )
            + tauself
            + taufor
            + abso3 * colamt[:laytrop, 2]
            + wx[:laytrop, 0] * ccl4[ig]
        )

        fracs[ns05 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
            fracrefa[ig, jplp] - fracrefa[ig, jpl]
        )

    #  --- ...  upper atmosphere loop

    speccomb = (
        colamt[laytrop:nlay, 2] + rfrate[laytrop:nlay, 5, 0] * colamt[laytrop:nlay, 1]
    )
    specparm = colamt[laytrop:nlay, 2] / speccomb
    specmult = 4.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * nspb[4] + js - 1

    speccomb1 = (
        colamt[laytrop:nlay, 2] + rfrate[laytrop:nlay, 5, 1] * colamt[laytrop:nlay, 1]
    )
    specparm1 = colamt[laytrop:nlay, 2] / speccomb1
    specmult1 = 4.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * nspb[4] + js1 - 1

    speccomb_planck = (
        colamt[laytrop:nlay, 2] + refrat_planck_b * colamt[laytrop:nlay, 1]
    )
    specparm_planck = colamt[laytrop:nlay, 2] / speccomb_planck
    specmult_planck = 4.0 * np.minimum(specparm_planck, oneminus)
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0
    jplp = jpl + 1

    id000 = ind0
    id010 = ind0 + 5
    id100 = ind0 + 1
    id110 = ind0 + 6
    id001 = ind1
    id011 = ind1 + 5
    id101 = ind1 + 1
    id111 = ind1 + 6

    fk00 = 1.0 - fs
    fk10 = fs

    fk01 = 1.0 - fs1
    fk11 = fs1

    fac000 = fk00 * fac00[laytrop:nlay]
    fac010 = fk00 * fac10[laytrop:nlay]
    fac100 = fk10 * fac00[laytrop:nlay]
    fac110 = fk10 * fac10[laytrop:nlay]

    fac001 = fk01 * fac01[laytrop:nlay]
    fac011 = fk01 * fac11[laytrop:nlay]
    fac101 = fk11 * fac01[laytrop:nlay]
    fac111 = fk11 * fac11[laytrop:nlay]

    for ig in range(ng05):
        taug[ns05 + ig, laytrop:nlay] = (
            speccomb
            * (
                fac000 * absb[ig, id000]
                + fac010 * absb[ig, id010]
                + fac100 * absb[ig, id100]
                + fac110 * absb[ig, id110]
            )
            + speccomb1
            * (
                fac001 * absb[ig, id001]
                + fac011 * absb[ig, id011]
                + fac101 * absb[ig, id101]
                + fac111 * absb[ig, id111]
            )
            + wx[laytrop:nlay, 0] * ccl4[ig]
        )

        fracs[ns05 + ig, laytrop:nlay] = fracrefb[ig, jpl] + fpl * (
            fracrefb[ig, jplp] - fracrefb[ig, jpl]
        )

    return taug, fracs


# Band 6:  820-980 cm-1 (low key - h2o; low minor - co2)
#                       (high key - none; high minor - cfc11, cfc12)
def taugb06(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    chi_mls,
    selfref,
    forref,
    absa,
    fracrefa,
    ka_mco2,
    cfc11adj,
    cfc12,
    nspa,
):
    #  ------------------------------------------------------------------  !
    #     band 6:  820-980 cm-1 (low key - h2o; low minor - co2)           !
    #                           (high key - none; high minor - cfc11, cfc12)
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping level:
    #     lower - co2, p = 706.2720 mb, t = 294.2 k
    #     upper - cfc11, cfc12

    #  --- ...  lower atmosphere loop
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[5]
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[5]

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indm = indminor[:laytrop] - 1
    indsp = inds + 1
    indfp = indf + 1
    indmp = indm + 1
    ind0p = ind0 + 1
    ind1p = ind1 + 1

    #  --- ...  in atmospheres where the amount of co2 is too great to be considered
    #           a minor species, adjust the column amount of co2 by an empirical factor
    #           to obtain the proper contribution.

    temp = coldry[:laytrop] * chi_mls[1, jp[:laytrop] + 1]
    ratco2 = colamt[:laytrop, 1] / temp
    adjcolco2 = np.where(
        ratco2 > 3.0, (2.0 + (ratco2 - 2.0) ** 0.77) * temp, colamt[:laytrop, 1]
    )

    for ig in range(ng06):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )
        absco2 = ka_mco2[ig, indm] + minorfrac[:laytrop] * (
            ka_mco2[ig, indmp] - ka_mco2[ig, indm]
        )

        taug[ns06 + ig, :laytrop] = (
            colamt[:laytrop, 0]
            * (
                fac00[:laytrop] * absa[ig, ind0]
                + fac10[:laytrop] * absa[ig, ind0p]
                + fac01[:laytrop] * absa[ig, ind1]
                + fac11[:laytrop] * absa[ig, ind1p]
            )
            + tauself
            + taufor
            + adjcolco2 * absco2
            + wx[:laytrop, 1] * cfc11adj[ig]
            + wx[:laytrop, 2] * cfc12[ig]
        )

        fracs[ns06 + ig, :laytrop] = fracrefa[ig]

    #  --- ...  upper atmosphere loop
    #           nothing important goes on above laytrop in this band.

    for ig in range(ng06):
        taug[ns06 + ig, laytrop:nlay] = (
            wx[laytrop:nlay, 1] * cfc11adj[ig] + wx[laytrop:nlay, 2] * cfc12[ig]
        )
        fracs[ns06 + ig, laytrop:nlay] = fracrefa[ig]

    return taug, fracs


# Band 7:  980-1080 cm-1 (low key - h2o,o3; low minor - co2)
#                        (high key - o3; high minor - co2)
def taugb07(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    chi_mls,
    selfref,
    forref,
    absa,
    absb,
    fracrefa,
    fracrefb,
    ka_mco2,
    kb_mco2,
    oneminus,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 7:  980-1080 cm-1 (low key - h2o,o3; low minor - co2)       !
    #                            (high key - o3; high minor - co2)         !
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping level :
    #     lower - co2, p = 706.2620 mbar, t= 278.94 k
    #     upper - co2, p = 12.9350 mbar, t = 234.01 k

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower atmosphere.

    refrat_planck_a = chi_mls[0, 2] / chi_mls[2, 2]  # P = 706.2620 mb
    refrat_m_a = chi_mls[0, 2] / chi_mls[2, 2]  # P = 706.2720 mb

    #  --- ...  lower atmosphere loop
    speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 1, 0] * colamt[:laytrop, 2]
    specparm = colamt[:laytrop, 0] / speccomb
    specmult = 8.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[6] + js - 1

    speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 1, 1] * colamt[:laytrop, 2]
    specparm1 = colamt[:laytrop, 0] / speccomb1
    specmult1 = 8.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[6] + js1 - 1

    speccomb_mco2 = colamt[:laytrop, 0] + refrat_m_a * colamt[:laytrop, 2]
    specparm_mco2 = colamt[:laytrop, 0] / speccomb_mco2
    specmult_mco2 = 8.0 * np.minimum(specparm_mco2, oneminus)
    jmco2 = 1 + specmult_mco2.astype(np.int32) - 1
    fmco2 = specmult_mco2 % 1.0

    speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 2]
    specparm_planck = colamt[:laytrop, 0] / speccomb_planck
    specmult_planck = 8.0 * np.minimum(specparm_planck, oneminus)
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indm = indminor[:laytrop] - 1
    indsp = inds + 1
    indfp = indf + 1
    indmp = indm + 1
    jplp = jpl + 1
    jmco2p = jmco2 + 1
    ind0p = ind0 + 1
    ind1p = ind1 + 1

    #  --- ...  in atmospheres where the amount of CO2 is too great to be considered
    #           a minor species, adjust the column amount of CO2 by an empirical factor
    #           to obtain the proper contribution.

    temp = coldry[:laytrop] * chi_mls[1, jp[:laytrop]]
    ratco2 = colamt[:laytrop, 1] / temp
    adjcolco2 = np.where(
        ratco2 > 3.0, (3.0 + (ratco2 - 3.0) ** 0.79) * temp, colamt[:laytrop, 1]
    )

    p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(specparm > 0.875, -fs, 0)
    p0 = np.where(p0 == 0, 0, p0)

    p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
        specparm > 0.875, p0 ** 4, 0
    )
    p40 = np.where(p40 == 0, 0, p40)

    fk00 = np.where(specparm < 0.125, p40, 0) + np.where(specparm > 0.875, p0 ** 4, 0)
    fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

    fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
        specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
    )
    fk10 = np.where(fk10 == 0, fs, fk10)

    fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
        specparm > 0.875, p0 + p40, 0
    )
    fk20 = np.where(fk20 == 0, 0.0, fk20)

    id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
        specparm > 0.875, ind0 + 1, 0
    )
    id000 = np.where(id000 == 0, ind0, id000)

    id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
        specparm > 0.875, ind0 + 10, 0
    )
    id010 = np.where(id010 == 0, ind0 + 9, id010)

    id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
        specparm > 0.875, ind0, 0
    )
    id100 = np.where(id100 == 0, ind0 + 1, id100)

    id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
        specparm > 0.875, ind0 + 9, 0
    )
    id110 = np.where(id110 == 0, ind0 + 10, id110)

    id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
        specparm > 0.875, ind0 - 1, 0
    )
    id200 = np.where(id200 == 0, ind0, id200)

    id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
        specparm > 0.875, ind0 + 8, 0
    )
    id210 = np.where(id210 == 0, ind0, id210)

    fac000 = fk00 * fac00[:laytrop]
    fac100 = fk10 * fac00[:laytrop]
    fac200 = fk20 * fac00[:laytrop]
    fac010 = fk00 * fac10[:laytrop]
    fac110 = fk10 * fac10[:laytrop]
    fac210 = fk20 * fac10[:laytrop]

    p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
        specparm1 > 0.875, -fs1, 0
    )
    p1 = np.where(p1 == 0, 0, p1)

    p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
        specparm1 > 0.875, p1 ** 4, 0
    )
    p41 = np.where(p41 == 0, 0, p41)

    fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(specparm1 > 0.875, p1 ** 4, 0)
    fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

    fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
        specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
    )
    fk11 = np.where(fk11 == 0, fs1, fk11)

    fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
        specparm1 > 0.875, p1 + p41, 0
    )
    fk21 = np.where(fk21 == 0, 0.0, fk21)

    id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
        specparm1 > 0.875, ind1 + 1, 0
    )
    id001 = np.where(id001 == 0, ind1, id001)

    id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
        specparm1 > 0.875, ind1 + 10, 0
    )
    id011 = np.where(id011 == 0, ind1 + 9, id011)

    id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
        specparm1 > 0.875, ind1, 0
    )
    id101 = np.where(id101 == 0, ind1 + 1, id101)

    id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
        specparm1 > 0.875, ind1 + 9, 0
    )
    id111 = np.where(id111 == 0, ind1 + 10, id111)

    id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
        specparm1 > 0.875, ind1 - 1, 0
    )
    id201 = np.where(id201 == 0, ind1, id201)

    id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
        specparm1 > 0.875, ind1 + 8, 0
    )
    id211 = np.where(id211 == 0, ind1, id211)

    fac001 = fk01 * fac01[:laytrop]
    fac101 = fk11 * fac01[:laytrop]
    fac201 = fk21 * fac01[:laytrop]
    fac011 = fk01 * fac11[:laytrop]
    fac111 = fk11 * fac11[:laytrop]
    fac211 = fk21 * fac11[:laytrop]

    for ig in range(ng07):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )
        co2m1 = ka_mco2[ig, jmco2, indm] + fmco2 * (
            ka_mco2[ig, jmco2p, indm] - ka_mco2[ig, jmco2, indm]
        )
        co2m2 = ka_mco2[ig, jmco2, indmp] + fmco2 * (
            ka_mco2[ig, jmco2p, indmp] - ka_mco2[ig, jmco2, indmp]
        )
        absco2 = co2m1 + minorfrac[:laytrop] * (co2m2 - co2m1)

        taug[ns07 + ig, :laytrop] = (
            speccomb
            * (
                fac000 * absa[ig, id000]
                + fac010 * absa[ig, id010]
                + fac100 * absa[ig, id100]
                + fac110 * absa[ig, id110]
                + fac200 * absa[ig, id200]
                + fac210 * absa[ig, id210]
            )
            + speccomb1
            * (
                fac001 * absa[ig, id001]
                + fac011 * absa[ig, id011]
                + fac101 * absa[ig, id101]
                + fac111 * absa[ig, id111]
                + fac201 * absa[ig, id201]
                + fac211 * absa[ig, id211]
            )
            + tauself
            + taufor
            + adjcolco2 * absco2
        )

        fracs[ns07 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
            fracrefa[ig, jplp] - fracrefa[ig, jpl]
        )

    #  --- ...  upper atmosphere loop

    #  --- ...  in atmospheres where the amount of co2 is too great to be considered
    #           a minor species, adjust the column amount of co2 by an empirical factor
    #           to obtain the proper contribution.

    temp = coldry[laytrop:nlay] * chi_mls[1, jp[laytrop:nlay]]
    ratco2 = colamt[laytrop:nlay, 1] / temp
    adjcolco2 = np.where(
        ratco2 > 3.0, (2.0 + (ratco2 - 2.0) ** 0.79) * temp, colamt[laytrop:nlay, 1]
    )

    ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * nspb[6]
    ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * nspb[6]

    indm = indminor[laytrop:nlay] - 1
    indmp = indm + 1
    ind0p = ind0 + 1
    ind1p = ind1 + 1

    for ig in range(ng07):
        absco2 = kb_mco2[ig, indm] + minorfrac[laytrop:nlay] * (
            kb_mco2[ig, indmp] - kb_mco2[ig, indm]
        )

        taug[ns07 + ig, laytrop:nlay] = (
            colamt[laytrop:nlay, 2]
            * (
                fac00[laytrop:nlay] * absb[ig, ind0]
                + fac10[laytrop:nlay] * absb[ig, ind0p]
                + fac01[laytrop:nlay] * absb[ig, ind1]
                + fac11[laytrop:nlay] * absb[ig, ind1p]
            )
            + adjcolco2 * absco2
        )

        fracs[ns07 + ig, laytrop:nlay] = fracrefb[ig]

    #  --- ...  empirical modification to code to improve stratospheric cooling rates
    #           for o3.  revised to apply weighting for g-point reduction in this band.

    taug[ns07 + 5, laytrop:nlay] = taug[ns07 + 5, laytrop:nlay] * 0.92
    taug[ns07 + 6, laytrop:nlay] = taug[ns07 + 6, laytrop:nlay] * 0.88
    taug[ns07 + 7, laytrop:nlay] = taug[ns07 + 7, laytrop:nlay] * 1.07
    taug[ns07 + 8, laytrop:nlay] = taug[ns07 + 8, laytrop:nlay] * 1.1
    taug[ns07 + 9, laytrop:nlay] = taug[ns07 + 9, laytrop:nlay] * 0.99
    taug[ns07 + 10, laytrop:nlay] = taug[ns07 + 10, laytrop:nlay] * 0.855

    return taug, fracs


# Band 8:  1080-1180 cm-1 (low key - h2o; low minor - co2,o3,n2o)
#                         (high key - o3; high minor - co2, n2o)
def taugb08(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    chi_mls,
    selfref,
    forref,
    absa,
    absb,
    fracrefa,
    fracrefb,
    ka_mo3,
    ka_mco2,
    kb_mco2,
    cfc12,
    ka_mn2o,
    kb_mn2o,
    cfc22adj,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 8:  1080-1180 cm-1 (low key - h2o; low minor - co2,o3,n2o)  !
    #                             (high key - o3; high minor - co2, n2o)   !
    #  ------------------------------------------------------------------  !
    #  --- ...  minor gas mapping level:
    #     lower - co2, p = 1053.63 mb, t = 294.2 k
    #     lower - o3,  p = 317.348 mb, t = 240.77 k
    #     lower - n2o, p = 706.2720 mb, t= 278.94 k
    #     lower - cfc12,cfc11
    #     upper - co2, p = 35.1632 mb, t = 223.28 k
    #     upper - n2o, p = 8.716e-2 mb, t = 226.03 k

    #  --- ...  lower atmosphere loop
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[7]
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[7]

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indm = indminor[:laytrop] - 1
    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indsp = inds + 1
    indfp = indf + 1
    indmp = indm + 1

    #  --- ...  in atmospheres where the amount of co2 is too great to be considered
    #           a minor species, adjust the column amount of co2 by an empirical factor
    #           to obtain the proper contribution.

    temp = coldry[:laytrop] * chi_mls[1, jp[:laytrop]]
    ratco2 = colamt[:laytrop, 1] / temp
    adjcolco2 = np.where(
        ratco2 > 3.0, (2.0 + (ratco2 - 2.0) ** 0.65) * temp, colamt[:laytrop, 1]
    )

    for ig in range(ng08):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )
        absco2 = ka_mco2[ig, indm] + minorfrac[:laytrop] * (
            ka_mco2[ig, indmp] - ka_mco2[ig, indm]
        )
        abso3 = ka_mo3[ig, indm] + minorfrac[:laytrop] * (
            ka_mo3[ig, indmp] - ka_mo3[ig, indm]
        )
        absn2o = ka_mn2o[ig, indm] + minorfrac[:laytrop] * (
            ka_mn2o[ig, indmp] - ka_mn2o[ig, indm]
        )

        taug[ns08 + ig, :laytrop] = (
            colamt[:laytrop, 0]
            * (
                fac00[:laytrop] * absa[ig, ind0]
                + fac10[:laytrop] * absa[ig, ind0p]
                + fac01[:laytrop] * absa[ig, ind1]
                + fac11[:laytrop] * absa[ig, ind1p]
            )
            + tauself
            + taufor
            + adjcolco2 * absco2
            + colamt[:laytrop, 2] * abso3
            + colamt[:laytrop, 3] * absn2o
            + wx[:laytrop, 2] * cfc12[ig]
            + wx[:laytrop, 3] * cfc22adj[ig]
        )

        fracs[ns08 + ig, :laytrop] = fracrefa[ig]

    #  --- ...  upper atmosphere loop

    ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * nspb[7]
    ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * nspb[7]

    indm = indminor[laytrop:nlay] - 1
    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indmp = indm + 1

    #  --- ...  in atmospheres where the amount of co2 is too great to be considered
    #           a minor species, adjust the column amount of co2 by an empirical factor
    #           to obtain the proper contribution.

    temp = coldry[laytrop:nlay] * chi_mls[1, jp[laytrop:nlay]]
    ratco2 = colamt[laytrop:nlay, 1] / temp
    adjcolco2 = np.where(
        ratco2 > 3.0, (2.0 + (ratco2 - 2.0) ** 0.65) * temp, colamt[laytrop:nlay, 1]
    )

    for ig in range(ng08):
        absco2 = kb_mco2[ig, indm] + minorfrac[laytrop:nlay] * (
            kb_mco2[ig, indmp] - kb_mco2[ig, indm]
        )
        absn2o = kb_mn2o[ig, indm] + minorfrac[laytrop:nlay] * (
            kb_mn2o[ig, indmp] - kb_mn2o[ig, indm]
        )

        taug[ns08 + ig, laytrop:nlay] = (
            colamt[laytrop:nlay, 2]
            * (
                fac00[laytrop:nlay] * absb[ig, ind0]
                + fac10[laytrop:nlay] * absb[ig, ind0p]
                + fac01[laytrop:nlay] * absb[ig, ind1]
                + fac11[laytrop:nlay] * absb[ig, ind1p]
            )
            + adjcolco2 * absco2
            + colamt[laytrop:nlay, 3] * absn2o
            + wx[laytrop:nlay, 2] * cfc12[ig]
            + wx[laytrop:nlay, 3] * cfc22adj[ig]
        )

        fracs[ns08 + ig, laytrop:nlay] = fracrefb[ig]

    return taug, fracs


# Band 9:  1180-1390 cm-1 (low key - h2o,ch4; low minor - n2o)
#                         (high key - ch4; high minor - n2o)
def taugb09(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    chi_mls,
    selfref,
    forref,
    absa,
    absb,
    fracrefa,
    fracrefb,
    ka_mn2o,
    kb_mn2o,
    oneminus,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 9:  1180-1390 cm-1 (low key - h2o,ch4; low minor - n2o)     !
    #                             (high key - ch4; high minor - n2o)       !
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping level :
    #     lower - n2o, p = 706.272 mbar, t = 278.94 k
    #     upper - n2o, p = 95.58 mbar, t = 215.7 k

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower/upper atmosphere.

    refrat_planck_a = chi_mls[0, 8] / chi_mls[5, 8]  # P = 212 mb
    refrat_m_a = chi_mls[0, 2] / chi_mls[5, 2]  # P = 706.272 mb

    #  --- ...  lower atmosphere loop
    speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 3, 0] * colamt[:laytrop, 4]
    specparm = colamt[:laytrop, 0] / speccomb
    specmult = 8.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[8] + js - 1

    speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 3, 1] * colamt[:laytrop, 4]
    specparm1 = colamt[:laytrop, 0] / speccomb1
    specmult1 = 8.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[8] + js1 - 1

    speccomb_mn2o = colamt[:laytrop, 0] + refrat_m_a * colamt[:laytrop, 4]
    specparm_mn2o = colamt[:laytrop, 0] / speccomb_mn2o
    specmult_mn2o = 8.0 * np.minimum(specparm_mn2o, oneminus)
    jmn2o = 1 + specmult_mn2o.astype(np.int32) - 1
    fmn2o = specmult_mn2o % 1.0

    speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 4]
    specparm_planck = colamt[:laytrop, 0] / speccomb_planck
    specmult_planck = 8.0 * np.minimum(specparm_planck, oneminus)
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indm = indminor[:laytrop] - 1
    indsp = inds + 1
    indfp = indf + 1
    indmp = indm + 1
    jplp = jpl + 1
    jmn2op = jmn2o + 1

    #  --- ...  in atmospheres where the amount of n2o is too great to be considered
    #           a minor species, adjust the column amount of n2o by an empirical factor
    #           to obtain the proper contribution.

    temp = coldry[:laytrop] * chi_mls[3, jp[:laytrop]]
    ratn2o = colamt[:laytrop, 3] / temp
    adjcoln2o = np.where(
        ratn2o > 1.5, (0.5 + (ratn2o - 0.5) ** 0.65) * temp, colamt[:laytrop, 3]
    )

    p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(specparm > 0.875, -fs, 0)
    p0 = np.where(p0 == 0, 0, p0)

    p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
        specparm > 0.875, p0 ** 4, 0
    )
    p40 = np.where(p40 == 0, 0, p40)

    fk00 = np.where(specparm < 0.125, p40, 0) + np.where(specparm > 0.875, p0 ** 4, 0)
    fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

    fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
        specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
    )
    fk10 = np.where(fk10 == 0, fs, fk10)

    fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
        specparm > 0.875, p0 + p40, 0
    )
    fk20 = np.where(fk20 == 0, 0.0, fk20)

    id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
        specparm > 0.875, ind0 + 1, 0
    )
    id000 = np.where(id000 == 0, ind0, id000)

    id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
        specparm > 0.875, ind0 + 10, 0
    )
    id010 = np.where(id010 == 0, ind0 + 9, id010)

    id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
        specparm > 0.875, ind0, 0
    )
    id100 = np.where(id100 == 0, ind0 + 1, id100)

    id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
        specparm > 0.875, ind0 + 9, 0
    )
    id110 = np.where(id110 == 0, ind0 + 10, id110)

    id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
        specparm > 0.875, ind0 - 1, 0
    )
    id200 = np.where(id200 == 0, ind0, id200)

    id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
        specparm > 0.875, ind0 + 8, 0
    )
    id210 = np.where(id210 == 0, ind0, id210)

    fac000 = fk00 * fac00[:laytrop]
    fac100 = fk10 * fac00[:laytrop]
    fac200 = fk20 * fac00[:laytrop]
    fac010 = fk00 * fac10[:laytrop]
    fac110 = fk10 * fac10[:laytrop]
    fac210 = fk20 * fac10[:laytrop]

    p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
        specparm1 > 0.875, -fs1, 0
    )
    p1 = np.where(p1 == 0, 0, p1)

    p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
        specparm1 > 0.875, p1 ** 4, 0
    )
    p41 = np.where(p41 == 0, 0, p41)

    fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(specparm1 > 0.875, p1 ** 4, 0)
    fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

    fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
        specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
    )
    fk11 = np.where(fk11 == 0, fs1, fk11)

    fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
        specparm1 > 0.875, p1 + p41, 0
    )
    fk21 = np.where(fk21 == 0, 0.0, fk21)

    id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
        specparm1 > 0.875, ind1 + 1, 0
    )
    id001 = np.where(id001 == 0, ind1, id001)

    id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
        specparm1 > 0.875, ind1 + 10, 0
    )
    id011 = np.where(id011 == 0, ind1 + 9, id011)

    id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
        specparm1 > 0.875, ind1, 0
    )
    id101 = np.where(id101 == 0, ind1 + 1, id101)

    id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
        specparm1 > 0.875, ind1 + 9, 0
    )
    id111 = np.where(id111 == 0, ind1 + 10, id111)

    id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
        specparm1 > 0.875, ind1 - 1, 0
    )
    id201 = np.where(id201 == 0, ind1, id201)

    id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
        specparm1 > 0.875, ind1 + 8, 0
    )
    id211 = np.where(id211 == 0, ind1, id211)

    fac001 = fk01 * fac01[:laytrop]
    fac101 = fk11 * fac01[:laytrop]
    fac201 = fk21 * fac01[:laytrop]
    fac011 = fk01 * fac11[:laytrop]
    fac111 = fk11 * fac11[:laytrop]
    fac211 = fk21 * fac11[:laytrop]

    for ig in range(ng09):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )
        n2om1 = ka_mn2o[ig, jmn2o, indm] + fmn2o * (
            ka_mn2o[ig, jmn2op, indm] - ka_mn2o[ig, jmn2o, indm]
        )
        n2om2 = ka_mn2o[ig, jmn2o, indmp] + fmn2o * (
            ka_mn2o[ig, jmn2op, indmp] - ka_mn2o[ig, jmn2o, indmp]
        )
        absn2o = n2om1 + minorfrac[:laytrop] * (n2om2 - n2om1)

        taug[ns09 + ig, :laytrop] = (
            speccomb
            * (
                fac000 * absa[ig, id000]
                + fac010 * absa[ig, id010]
                + fac100 * absa[ig, id100]
                + fac110 * absa[ig, id110]
                + fac200 * absa[ig, id200]
                + fac210 * absa[ig, id210]
            )
            + speccomb1
            * (
                fac001 * absa[ig, id001]
                + fac011 * absa[ig, id011]
                + fac101 * absa[ig, id101]
                + fac111 * absa[ig, id111]
                + fac201 * absa[ig, id201]
                + fac211 * absa[ig, id211]
            )
            + tauself
            + taufor
            + adjcoln2o * absn2o
        )

        fracs[ns09 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
            fracrefa[ig, jplp] - fracrefa[ig, jpl]
        )

    #  --- ...  upper atmosphere loop
    ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * nspb[8]
    ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * nspb[8]

    indm = indminor[laytrop:nlay] - 1
    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indmp = indm + 1

    #  --- ...  in atmospheres where the amount of n2o is too great to be considered
    #           a minor species, adjust the column amount of n2o by an empirical factor
    #           to obtain the proper contribution.

    temp = coldry[laytrop:nlay] * chi_mls[3, jp[laytrop:nlay]]
    ratn2o = colamt[laytrop:nlay, 3] / temp
    adjcoln2o = np.where(
        ratn2o > 1.5, (0.5 + (ratn2o - 0.5) ** 0.65) * temp, colamt[laytrop:nlay, 3]
    )

    for ig in range(ng09):
        absn2o = kb_mn2o[ig, indm] + minorfrac[laytrop:nlay] * (
            kb_mn2o[ig, indmp] - kb_mn2o[ig, indm]
        )

        taug[ns09 + ig, laytrop:nlay] = (
            colamt[laytrop:nlay, 4]
            * (
                fac00[laytrop:nlay] * absb[ig, ind0]
                + fac10[laytrop:nlay] * absb[ig, ind0p]
                + fac01[laytrop:nlay] * absb[ig, ind1]
                + fac11[laytrop:nlay] * absb[ig, ind1p]
            )
            + adjcoln2o * absn2o
        )

        fracs[ns09 + ig, laytrop:nlay] = fracrefb[ig]

    return taug, fracs


# Band 10:  1390-1480 cm-1 (low key - h2o; high key - h2o)
def taugb10(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    selfref,
    forref,
    absa,
    absb,
    fracrefa,
    fracrefb,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 10:  1390-1480 cm-1 (low key - h2o; high key - h2o)         !
    #  ------------------------------------------------------------------  !

    #  --- ...  lower atmosphere loop
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[9]
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[9]

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indsp = inds + 1
    indfp = indf + 1

    for ig in range(ng10):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )

        taug[ns10 + ig, :laytrop] = (
            colamt[:laytrop, 0]
            * (
                fac00[:laytrop] * absa[ig, ind0]
                + fac10[:laytrop] * absa[ig, ind0p]
                + fac01[:laytrop] * absa[ig, ind1]
                + fac11[:laytrop] * absa[ig, ind1p]
            )
            + tauself
            + taufor
        )

        fracs[ns10 + ig, :laytrop] = fracrefa[ig]

    #  --- ...  upper atmosphere loop

    ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * nspb[9]
    ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * nspb[9]

    indf = indfor[laytrop:nlay] - 1
    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indfp = indf + 1

    for ig in range(ng10):
        taufor = forfac[laytrop:nlay] * (
            forref[ig, indf]
            + forfrac[laytrop:nlay] * (forref[ig, indfp] - forref[ig, indf])
        )

        taug[ns10 + ig, laytrop:nlay] = (
            colamt[laytrop:nlay, 0]
            * (
                fac00[laytrop:nlay] * absb[ig, ind0]
                + fac10[laytrop:nlay] * absb[ig, ind0p]
                + fac01[laytrop:nlay] * absb[ig, ind1]
                + fac11[laytrop:nlay] * absb[ig, ind1p]
            )
            + taufor
        )

        fracs[ns10 + ig, laytrop:nlay] = fracrefb[ig]

    return taug, fracs


# Band 11:  1480-1800 cm-1 (low - h2o; low minor - o2)
#                          (high key - h2o; high minor - o2)
def taugb11(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    selfref,
    forref,
    absa,
    absb,
    fracrefa,
    fracrefb,
    ka_mo2,
    kb_mo2,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 11:  1480-1800 cm-1 (low - h2o; low minor - o2)             !
    #                              (high key - h2o; high minor - o2)       !
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping level :
    #     lower - o2, p = 706.2720 mbar, t = 278.94 k
    #     upper - o2, p = 4.758820 mbarm t = 250.85 k

    #  --- ...  lower atmosphere loop
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[10]
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[10]

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indm = indminor[:laytrop] - 1
    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indsp = inds + 1
    indfp = indf + 1
    indmp = indm + 1

    scaleo2 = colamt[:laytrop, 5] * scaleminor[:laytrop]

    for ig in range(ng11):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )
        tauo2 = scaleo2 * (
            ka_mo2[ig, indm]
            + minorfrac[:laytrop] * (ka_mo2[ig, indmp] - ka_mo2[ig, indm])
        )

        taug[ns11 + ig, :laytrop] = (
            colamt[:laytrop, 0]
            * (
                fac00[:laytrop] * absa[ig, ind0]
                + fac10[:laytrop] * absa[ig, ind0p]
                + fac01[:laytrop] * absa[ig, ind1]
                + fac11[:laytrop] * absa[ig, ind1p]
            )
            + tauself
            + taufor
            + tauo2
        )

        fracs[ns11 + ig, :laytrop] = fracrefa[ig]

    #  --- ...  upper atmosphere loop
    ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * nspb[10]
    ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * nspb[10]

    indf = indfor[laytrop:nlay] - 1
    indm = indminor[laytrop:nlay] - 1
    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indfp = indf + 1
    indmp = indm + 1

    scaleo2 = colamt[laytrop:nlay, 5] * scaleminor[laytrop:nlay]

    for ig in range(ng11):
        taufor = forfac[laytrop:nlay] * (
            forref[ig, indf]
            + forfrac[laytrop:nlay] * (forref[ig, indfp] - forref[ig, indf])
        )
        tauo2 = scaleo2 * (
            kb_mo2[ig, indm]
            + minorfrac[laytrop:nlay] * (kb_mo2[ig, indmp] - kb_mo2[ig, indm])
        )

        taug[ns11 + ig, laytrop:nlay] = (
            colamt[laytrop:nlay, 0]
            * (
                fac00[laytrop:nlay] * absb[ig, ind0]
                + fac10[laytrop:nlay] * absb[ig, ind0p]
                + fac01[laytrop:nlay] * absb[ig, ind1]
                + fac11[laytrop:nlay] * absb[ig, ind1p]
            )
            + taufor
            + tauo2
        )

        fracs[ns11 + ig, laytrop:nlay] = fracrefb[ig]

    return taug, fracs


# Band 12:  1800-2080 cm-1 (low - h2o,co2; high - nothing)
def taugb12(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    chi_mls,
    selfref,
    forref,
    absa,
    fracrefa,
    oneminus,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 12:  1800-2080 cm-1 (low - h2o,co2; high - nothing)         !
    #  ------------------------------------------------------------------  !

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower/upper atmosphere.

    refrat_planck_a = chi_mls[0, 9] / chi_mls[1, 9]  # P =   174.164 mb

    #  --- ...  lower atmosphere loop

    speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 0] * colamt[:laytrop, 1]
    specparm = colamt[:laytrop, 0] / speccomb
    specmult = 8.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[11] + js - 1

    speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 0, 1] * colamt[:laytrop, 1]
    specparm1 = colamt[:laytrop, 0] / speccomb1
    specmult1 = 8.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[11] + js1 - 1

    speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 1]
    specparm_planck = colamt[:laytrop, 0] / speccomb_planck
    specparm_planck = np.where(specparm_planck >= oneminus, oneminus, specparm_planck)
    specmult_planck = 8.0 * specparm_planck
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indsp = inds + 1
    indfp = indf + 1
    jplp = jpl + 1

    p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(specparm > 0.875, -fs, 0)
    p0 = np.where(p0 == 0, 0, p0)

    p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
        specparm > 0.875, p0 ** 4, 0
    )
    p40 = np.where(p40 == 0, 0, p40)

    fk00 = np.where(specparm < 0.125, p40, 0) + np.where(specparm > 0.875, p0 ** 4, 0)
    fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

    fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
        specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
    )
    fk10 = np.where(fk10 == 0, fs, fk10)

    fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
        specparm > 0.875, p0 + p40, 0
    )
    fk20 = np.where(fk20 == 0, 0.0, fk20)

    id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
        specparm > 0.875, ind0 + 1, 0
    )
    id000 = np.where(id000 == 0, ind0, id000)

    id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
        specparm > 0.875, ind0 + 10, 0
    )
    id010 = np.where(id010 == 0, ind0 + 9, id010)

    id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
        specparm > 0.875, ind0, 0
    )
    id100 = np.where(id100 == 0, ind0 + 1, id100)

    id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
        specparm > 0.875, ind0 + 9, 0
    )
    id110 = np.where(id110 == 0, ind0 + 10, id110)

    id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
        specparm > 0.875, ind0 - 1, 0
    )
    id200 = np.where(id200 == 0, ind0, id200)

    id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
        specparm > 0.875, ind0 + 8, 0
    )
    id210 = np.where(id210 == 0, ind0, id210)

    fac000 = fk00 * fac00[:laytrop]
    fac100 = fk10 * fac00[:laytrop]
    fac200 = fk20 * fac00[:laytrop]
    fac010 = fk00 * fac10[:laytrop]
    fac110 = fk10 * fac10[:laytrop]
    fac210 = fk20 * fac10[:laytrop]

    p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
        specparm1 > 0.875, -fs1, 0
    )
    p1 = np.where(p1 == 0, 0, p1)

    p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
        specparm1 > 0.875, p1 ** 4, 0
    )
    p41 = np.where(p41 == 0, 0, p41)

    fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(specparm1 > 0.875, p1 ** 4, 0)
    fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

    fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
        specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
    )
    fk11 = np.where(fk11 == 0, fs1, fk11)

    fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
        specparm1 > 0.875, p1 + p41, 0
    )
    fk21 = np.where(fk21 == 0, 0.0, fk21)

    id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
        specparm1 > 0.875, ind1 + 1, 0
    )
    id001 = np.where(id001 == 0, ind1, id001)

    id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
        specparm1 > 0.875, ind1 + 10, 0
    )
    id011 = np.where(id011 == 0, ind1 + 9, id011)

    id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
        specparm1 > 0.875, ind1, 0
    )
    id101 = np.where(id101 == 0, ind1 + 1, id101)

    id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
        specparm1 > 0.875, ind1 + 9, 0
    )
    id111 = np.where(id111 == 0, ind1 + 10, id111)

    id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
        specparm1 > 0.875, ind1 - 1, 0
    )
    id201 = np.where(id201 == 0, ind1, id201)

    id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
        specparm1 > 0.875, ind1 + 8, 0
    )
    id211 = np.where(id211 == 0, ind1, id211)

    fac001 = fk01 * fac01[:laytrop]
    fac101 = fk11 * fac01[:laytrop]
    fac201 = fk21 * fac01[:laytrop]
    fac011 = fk01 * fac11[:laytrop]
    fac111 = fk11 * fac11[:laytrop]
    fac211 = fk21 * fac11[:laytrop]

    for ig in range(ng12):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )

        taug[ns12 + ig, :laytrop] = (
            speccomb
            * (
                fac000 * absa[ig, id000]
                + fac010 * absa[ig, id010]
                + fac100 * absa[ig, id100]
                + fac110 * absa[ig, id110]
                + fac200 * absa[ig, id200]
                + fac210 * absa[ig, id210]
            )
            + speccomb1
            * (
                fac001 * absa[ig, id001]
                + fac011 * absa[ig, id011]
                + fac101 * absa[ig, id101]
                + fac111 * absa[ig, id111]
                + fac201 * absa[ig, id201]
                + fac211 * absa[ig, id211]
            )
            + tauself
            + taufor
        )

        fracs[ns12 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
            fracrefa[ig, jplp] - fracrefa[ig, jpl]
        )

    #  --- ...  upper atmosphere loop
    for ig in range(ng12):
        taug[ns12 + ig, laytrop:nlay] = 0.0
        fracs[ns12 + ig, laytrop:nlay] = 0.0

    return taug, fracs


# Band 13:  2080-2250 cm-1 (low key-h2o,n2o; high minor-o3 minor)
def taugb13(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    chi_mls,
    selfref,
    forref,
    absa,
    fracrefa,
    fracrefb,
    ka_mco2,
    ka_mco,
    kb_mo3,
    oneminus,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 13:  2080-2250 cm-1 (low key-h2o,n2o; high minor-o3 minor)  !
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping levels :
    #     lower - co2, p = 1053.63 mb, t = 294.2 k
    #     lower - co, p = 706 mb, t = 278.94 k
    #     upper - o3, p = 95.5835 mb, t = 215.7 k

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower/upper atmosphere.

    refrat_planck_a = chi_mls[0, 4] / chi_mls[3, 4]  # P = 473.420 mb (Level 5)
    refrat_m_a = chi_mls[0, 0] / chi_mls[3, 0]  # P = 1053. (Level 1)
    refrat_m_a3 = chi_mls[0, 2] / chi_mls[3, 2]  # P = 706. (Level 3)

    #  --- ...  lower atmosphere loop

    speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 2, 0] * colamt[:laytrop, 3]
    specparm = colamt[:laytrop, 0] / speccomb
    specmult = 8.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[12] + js - 1

    speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 2, 1] * colamt[:laytrop, 3]
    specparm1 = colamt[:laytrop, 0] / speccomb1
    specmult1 = 8.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[12] + js1 - 1

    speccomb_mco2 = colamt[:laytrop, 0] + refrat_m_a * colamt[:laytrop, 3]
    specparm_mco2 = colamt[:laytrop, 0] / speccomb_mco2
    specmult_mco2 = 8.0 * np.minimum(specparm_mco2, oneminus)
    jmco2 = 1 + specmult_mco2.astype(np.int32) - 1
    fmco2 = specmult_mco2 % 1.0

    #  --- ...  in atmospheres where the amount of co2 is too great to be considered
    #           a minor species, adjust the column amount of co2 by an empirical factor
    #           to obtain the proper contribution.

    speccomb_mco = colamt[:laytrop, 0] + refrat_m_a3 * colamt[:laytrop, 3]
    specparm_mco = colamt[:laytrop, 0] / speccomb_mco
    specmult_mco = 8.0 * np.minimum(specparm_mco, oneminus)
    jmco = 1 + specmult_mco.astype(np.int32) - 1
    fmco = specmult_mco % 1.0

    speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 3]
    specparm_planck = colamt[:laytrop, 0] / speccomb_planck
    specmult_planck = 8.0 * np.minimum(specparm_planck, oneminus)
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indm = indminor[:laytrop] - 1
    indsp = inds + 1
    indfp = indf + 1
    indmp = indm + 1
    jplp = jpl + 1
    jmco2p = jmco2 + 1
    jmcop = jmco + 1

    #  --- ...  in atmospheres where the amount of co2 is too great to be considered
    #           a minor species, adjust the column amount of co2 by an empirical factor
    #           to obtain the proper contribution.

    temp = coldry[:laytrop] * 3.55e-4
    ratco2 = colamt[:laytrop, 1] / temp
    adjcolco2 = np.where(
        ratco2 > 3.0, (2.0 + (ratco2 - 2.0) ** 0.68) * temp, colamt[:laytrop, 1]
    )

    p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(specparm > 0.875, -fs, 0)
    p0 = np.where(p0 == 0, 0, p0)

    p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
        specparm > 0.875, p0 ** 4, 0
    )
    p40 = np.where(p40 == 0, 0, p40)

    fk00 = np.where(specparm < 0.125, p40, 0) + np.where(specparm > 0.875, p0 ** 4, 0)
    fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

    fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
        specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
    )
    fk10 = np.where(fk10 == 0, fs, fk10)

    fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
        specparm > 0.875, p0 + p40, 0
    )
    fk20 = np.where(fk20 == 0, 0.0, fk20)

    id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
        specparm > 0.875, ind0 + 1, 0
    )
    id000 = np.where(id000 == 0, ind0, id000)

    id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
        specparm > 0.875, ind0 + 10, 0
    )
    id010 = np.where(id010 == 0, ind0 + 9, id010)

    id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
        specparm > 0.875, ind0, 0
    )
    id100 = np.where(id100 == 0, ind0 + 1, id100)

    id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
        specparm > 0.875, ind0 + 9, 0
    )
    id110 = np.where(id110 == 0, ind0 + 10, id110)

    id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
        specparm > 0.875, ind0 - 1, 0
    )
    id200 = np.where(id200 == 0, ind0, id200)

    id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
        specparm > 0.875, ind0 + 8, 0
    )
    id210 = np.where(id210 == 0, ind0, id210)

    fac000 = fk00 * fac00[:laytrop]
    fac100 = fk10 * fac00[:laytrop]
    fac200 = fk20 * fac00[:laytrop]
    fac010 = fk00 * fac10[:laytrop]
    fac110 = fk10 * fac10[:laytrop]
    fac210 = fk20 * fac10[:laytrop]

    p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
        specparm1 > 0.875, -fs1, 0
    )
    p1 = np.where(p1 == 0, 0, p1)

    p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
        specparm1 > 0.875, p1 ** 4, 0
    )
    p41 = np.where(p41 == 0, 0, p41)

    fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(specparm1 > 0.875, p1 ** 4, 0)
    fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

    fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
        specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
    )
    fk11 = np.where(fk11 == 0, fs1, fk11)

    fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
        specparm1 > 0.875, p1 + p41, 0
    )
    fk21 = np.where(fk21 == 0, 0.0, fk21)

    id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
        specparm1 > 0.875, ind1 + 1, 0
    )
    id001 = np.where(id001 == 0, ind1, id001)

    id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
        specparm1 > 0.875, ind1 + 10, 0
    )
    id011 = np.where(id011 == 0, ind1 + 9, id011)

    id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
        specparm1 > 0.875, ind1, 0
    )
    id101 = np.where(id101 == 0, ind1 + 1, id101)

    id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
        specparm1 > 0.875, ind1 + 9, 0
    )
    id111 = np.where(id111 == 0, ind1 + 10, id111)

    id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
        specparm1 > 0.875, ind1 - 1, 0
    )
    id201 = np.where(id201 == 0, ind1, id201)

    id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
        specparm1 > 0.875, ind1 + 8, 0
    )
    id211 = np.where(id211 == 0, ind1, id211)

    fac001 = fk01 * fac01[:laytrop]
    fac101 = fk11 * fac01[:laytrop]
    fac201 = fk21 * fac01[:laytrop]
    fac011 = fk01 * fac11[:laytrop]
    fac111 = fk11 * fac11[:laytrop]
    fac211 = fk21 * fac11[:laytrop]

    for ig in range(ng13):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )
        co2m1 = ka_mco2[ig, jmco2, indm] + fmco2 * (
            ka_mco2[ig, jmco2p, indm] - ka_mco2[ig, jmco2, indm]
        )
        co2m2 = ka_mco2[ig, jmco2, indmp] + fmco2 * (
            ka_mco2[ig, jmco2p, indmp] - ka_mco2[ig, jmco2, indmp]
        )
        absco2 = co2m1 + minorfrac[:laytrop] * (co2m2 - co2m1)
        com1 = ka_mco[ig, jmco, indm] + fmco * (
            ka_mco[ig, jmcop, indm] - ka_mco[ig, jmco, indm]
        )
        com2 = ka_mco[ig, jmco, indmp] + fmco * (
            ka_mco[ig, jmcop, indmp] - ka_mco[ig, jmco, indmp]
        )
        absco = com1 + minorfrac[:laytrop] * (com2 - com1)

        taug[ns13 + ig, :laytrop] = (
            speccomb
            * (
                fac000 * absa[ig, id000]
                + fac010 * absa[ig, id010]
                + fac100 * absa[ig, id100]
                + fac110 * absa[ig, id110]
                + fac200 * absa[ig, id200]
                + fac210 * absa[ig, id210]
            )
            + speccomb1
            * (
                fac001 * absa[ig, id001]
                + fac011 * absa[ig, id011]
                + fac101 * absa[ig, id101]
                + fac111 * absa[ig, id111]
                + fac201 * absa[ig, id201]
                + fac211 * absa[ig, id211]
            )
            + tauself
            + taufor
            + adjcolco2 * absco2
            + colamt[:laytrop, 6] * absco
        )

        fracs[ns13 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
            fracrefa[ig, jplp] - fracrefa[ig, jpl]
        )

    #  --- ...  upper atmosphere loop
    indm = indminor[laytrop:nlay] - 1
    indmp = indm + 1

    for ig in range(ng13):
        abso3 = kb_mo3[ig, indm] + minorfrac[laytrop:nlay] * (
            kb_mo3[ig, indmp] - kb_mo3[ig, indm]
        )

        taug[ns13 + ig, laytrop:nlay] = colamt[laytrop:nlay, 2] * abso3

        fracs[ns13 + ig, laytrop:nlay] = fracrefb[ig]

    return taug, fracs, taufor


# Band 14:  2250-2380 cm-1 (low - co2; high - co2)
def taugb14(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    taufor,
    selfref,
    forref,
    absa,
    absb,
    fracrefa,
    fracrefb,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 14:  2250-2380 cm-1 (low - co2; high - co2)                 !
    #  ------------------------------------------------------------------  !

    #  --- ...  lower atmosphere loop

    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[13]
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[13]

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    ind0p = ind0 + 1
    ind1p = ind1 + 1
    indsp = inds + 1
    indfp = indf + 1

    for ig in range(ng14):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )

        taug[ns14 + ig, :laytrop] = (
            colamt[:laytrop, 1]
            * (
                fac00[:laytrop] * absa[ig, ind0]
                + fac10[:laytrop] * absa[ig, ind0p]
                + fac01[:laytrop] * absa[ig, ind1]
                + fac11[:laytrop] * absa[ig, ind1p]
            )
            + tauself
            + taufor
        )

        fracs[ns14 + ig, :laytrop] = fracrefa[ig]

    #  --- ...  upper atmosphere loop

    ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * nspb[13]
    ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * nspb[13]

    ind0p = ind0 + 1
    ind1p = ind1 + 1

    for ig in range(ng14):
        taug[ns14 + ig, laytrop:nlay] = colamt[laytrop:nlay, 1] * (
            fac00[laytrop:nlay] * absb[ig, ind0]
            + fac10[laytrop:nlay] * absb[ig, ind0p]
            + fac01[laytrop:nlay] * absb[ig, ind1]
            + fac11[laytrop:nlay] * absb[ig, ind1p]
        )

        fracs[ns14 + ig, laytrop:nlay] = fracrefb[ig]

    return taug, fracs


# Band 15:  2380-2600 cm-1 (low - n2o,co2; low minor - n2)
#                          (high - nothing)
def taugb15(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    chi_mls,
    selfref,
    forref,
    absa,
    fracrefa,
    ka_mn2,
    oneminus,
    nspa,
):
    #  ------------------------------------------------------------------  !
    #     band 15:  2380-2600 cm-1 (low - n2o,co2; low minor - n2)         !
    #                              (high - nothing)                        !
    #  ------------------------------------------------------------------  !

    #  --- ...  minor gas mapping level :
    #     lower - nitrogen continuum, P = 1053., T = 294.

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower atmosphere.

    refrat_planck_a = chi_mls[3, 0] / chi_mls[1, 0]  # P = 1053. mb (Level 1)
    refrat_m_a = chi_mls[3, 0] / chi_mls[1, 0]  # P = 1053. mb

    #  --- ...  lower atmosphere loop
    speccomb = colamt[:laytrop, 3] + rfrate[:laytrop, 4, 0] * colamt[:laytrop, 1]
    specparm = colamt[:laytrop, 3] / speccomb
    specmult = 8.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[14] + js - 1

    speccomb1 = colamt[:laytrop, 3] + rfrate[:laytrop, 4, 1] * colamt[:laytrop, 1]
    specparm1 = colamt[:laytrop, 3] / speccomb1
    specmult1 = 8.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[14] + js1 - 1

    speccomb_mn2 = colamt[:laytrop, 3] + refrat_m_a * colamt[:laytrop, 1]
    specparm_mn2 = colamt[:laytrop, 3] / speccomb_mn2
    specmult_mn2 = 8.0 * np.minimum(specparm_mn2, oneminus)
    jmn2 = 1 + specmult_mn2.astype(np.int32) - 1
    fmn2 = specmult_mn2 % 1.0

    speccomb_planck = colamt[:laytrop, 3] + refrat_planck_a * colamt[:laytrop, 1]
    specparm_planck = colamt[:laytrop, 3] / speccomb_planck
    specmult_planck = 8.0 * np.minimum(specparm_planck, oneminus)
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0

    scalen2 = colbrd[:laytrop] * scaleminor[:laytrop]

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indm = indminor[:laytrop] - 1
    indsp = inds + 1
    indfp = indf + 1
    indmp = indm + 1
    jplp = jpl + 1
    jmn2p = jmn2 + 1

    p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(specparm > 0.875, -fs, 0)
    p0 = np.where(p0 == 0, 0, p0)

    p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
        specparm > 0.875, p0 ** 4, 0
    )
    p40 = np.where(p40 == 0, 0, p40)

    fk00 = np.where(specparm < 0.125, p40, 0) + np.where(specparm > 0.875, p0 ** 4, 0)
    fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

    fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
        specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
    )
    fk10 = np.where(fk10 == 0, fs, fk10)

    fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
        specparm > 0.875, p0 + p40, 0
    )
    fk20 = np.where(fk20 == 0, 0.0, fk20)

    id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
        specparm > 0.875, ind0 + 1, 0
    )
    id000 = np.where(id000 == 0, ind0, id000)

    id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
        specparm > 0.875, ind0 + 10, 0
    )
    id010 = np.where(id010 == 0, ind0 + 9, id010)

    id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
        specparm > 0.875, ind0, 0
    )
    id100 = np.where(id100 == 0, ind0 + 1, id100)

    id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
        specparm > 0.875, ind0 + 9, 0
    )
    id110 = np.where(id110 == 0, ind0 + 10, id110)

    id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
        specparm > 0.875, ind0 - 1, 0
    )
    id200 = np.where(id200 == 0, ind0, id200)

    id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
        specparm > 0.875, ind0 + 8, 0
    )
    id210 = np.where(id210 == 0, ind0, id210)

    fac000 = fk00 * fac00[:laytrop]
    fac100 = fk10 * fac00[:laytrop]
    fac200 = fk20 * fac00[:laytrop]
    fac010 = fk00 * fac10[:laytrop]
    fac110 = fk10 * fac10[:laytrop]
    fac210 = fk20 * fac10[:laytrop]

    p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
        specparm1 > 0.875, -fs1, 0
    )
    p1 = np.where(p1 == 0, 0, p1)

    p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
        specparm1 > 0.875, p1 ** 4, 0
    )
    p41 = np.where(p41 == 0, 0, p41)

    fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(specparm1 > 0.875, p1 ** 4, 0)
    fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

    fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
        specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
    )
    fk11 = np.where(fk11 == 0, fs1, fk11)

    fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
        specparm1 > 0.875, p1 + p41, 0
    )
    fk21 = np.where(fk21 == 0, 0.0, fk21)

    id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
        specparm1 > 0.875, ind1 + 1, 0
    )
    id001 = np.where(id001 == 0, ind1, id001)

    id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
        specparm1 > 0.875, ind1 + 10, 0
    )
    id011 = np.where(id011 == 0, ind1 + 9, id011)

    id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
        specparm1 > 0.875, ind1, 0
    )
    id101 = np.where(id101 == 0, ind1 + 1, id101)

    id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
        specparm1 > 0.875, ind1 + 9, 0
    )
    id111 = np.where(id111 == 0, ind1 + 10, id111)

    id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
        specparm1 > 0.875, ind1 - 1, 0
    )
    id201 = np.where(id201 == 0, ind1, id201)

    id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
        specparm1 > 0.875, ind1 + 8, 0
    )
    id211 = np.where(id211 == 0, ind1, id211)

    fac001 = fk01 * fac01[:laytrop]
    fac101 = fk11 * fac01[:laytrop]
    fac201 = fk21 * fac01[:laytrop]
    fac011 = fk01 * fac11[:laytrop]
    fac111 = fk11 * fac11[:laytrop]
    fac211 = fk21 * fac11[:laytrop]

    for ig in range(ng15):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )
        n2m1 = ka_mn2[ig, jmn2, indm] + fmn2 * (
            ka_mn2[ig, jmn2p, indm] - ka_mn2[ig, jmn2, indm]
        )
        n2m2 = ka_mn2[ig, jmn2, indmp] + fmn2 * (
            ka_mn2[ig, jmn2p, indmp] - ka_mn2[ig, jmn2, indmp]
        )
        taun2 = scalen2 * (n2m1 + minorfrac[:laytrop] * (n2m2 - n2m1))

        taug[ns15 + ig, :laytrop] = (
            speccomb
            * (
                fac000 * absa[ig, id000]
                + fac010 * absa[ig, id010]
                + fac100 * absa[ig, id100]
                + fac110 * absa[ig, id110]
                + fac200 * absa[ig, id200]
                + fac210 * absa[ig, id210]
            )
            + speccomb1
            * (
                fac001 * absa[ig, id001]
                + fac011 * absa[ig, id011]
                + fac101 * absa[ig, id101]
                + fac111 * absa[ig, id111]
                + fac201 * absa[ig, id201]
                + fac211 * absa[ig, id211]
            )
            + tauself
            + taufor
            + taun2
        )

        fracs[ns15 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
            fracrefa[ig, jplp] - fracrefa[ig, jpl]
        )

    #  --- ...  upper atmosphere loop
    for ig in range(ng15):
        taug[ns15 + ig, laytrop:nlay] = 0.0

        fracs[ns15 + ig, laytrop:nlay] = 0.0

    return taug, fracs


# Band 16:  2600-3250 cm-1 (low key- h2o,ch4; high key - ch4)
def taugb16(
    laytrop,
    pavel,
    coldry,
    colamt,
    colbrd,
    wx,
    tauaer,
    rfrate,
    fac00,
    fac01,
    fac10,
    fac11,
    jp,
    jt,
    jt1,
    selffac,
    selffrac,
    indself,
    forfac,
    forfrac,
    indfor,
    minorfrac,
    scaleminor,
    scaleminorn2,
    indminor,
    nlay,
    taug,
    fracs,
    chi_mls,
    selfref,
    forref,
    absa,
    absb,
    fracrefa,
    fracrefb,
    oneminus,
    nspa,
    nspb,
):
    #  ------------------------------------------------------------------  !
    #     band 16:  2600-3250 cm-1 (low key- h2o,ch4; high key - ch4)      !
    #  ------------------------------------------------------------------  !

    #  --- ...  calculate reference ratio to be used in calculation of Planck
    #           fraction in lower atmosphere.

    refrat_planck_a = chi_mls[0, 5] / chi_mls[5, 5]  # P = 387. mb (Level 6)

    #  --- ...  lower atmosphere loop
    speccomb = colamt[:laytrop, 0] + rfrate[:laytrop, 3, 0] * colamt[:laytrop, 4]
    specparm = colamt[:laytrop, 0] / speccomb
    specmult = 8.0 * np.minimum(specparm, oneminus)
    js = 1 + specmult.astype(np.int32)
    fs = specmult % 1.0
    ind0 = ((jp[:laytrop] - 1) * 5 + (jt[:laytrop] - 1)) * nspa[15] + js - 1

    speccomb1 = colamt[:laytrop, 0] + rfrate[:laytrop, 3, 1] * colamt[:laytrop, 4]
    specparm1 = colamt[:laytrop, 0] / speccomb1
    specmult1 = 8.0 * np.minimum(specparm1, oneminus)
    js1 = 1 + specmult1.astype(np.int32)
    fs1 = specmult1 % 1.0
    ind1 = (jp[:laytrop] * 5 + (jt1[:laytrop] - 1)) * nspa[15] + js1 - 1

    speccomb_planck = colamt[:laytrop, 0] + refrat_planck_a * colamt[:laytrop, 4]
    specparm_planck = colamt[:laytrop, 0] / speccomb_planck
    specmult_planck = 8.0 * np.minimum(specparm_planck, oneminus)
    jpl = 1 + specmult_planck.astype(np.int32) - 1
    fpl = specmult_planck % 1.0

    inds = indself[:laytrop] - 1
    indf = indfor[:laytrop] - 1
    indsp = inds + 1
    indfp = indf + 1
    jplp = jpl + 1

    p0 = np.where(specparm < 0.125, fs - 1.0, 0) + np.where(specparm > 0.875, -fs, 0)
    p0 = np.where(p0 == 0, 0, p0)

    p40 = np.where(specparm < 0.125, p0 ** 4, 0) + np.where(
        specparm > 0.875, p0 ** 4, 0
    )
    p40 = np.where(p40 == 0, 0, p40)

    fk00 = np.where(specparm < 0.125, p40, 0) + np.where(specparm > 0.875, p0 ** 4, 0)
    fk00 = np.where(fk00 == 0, 1.0 - fs, fk00)

    fk10 = np.where(specparm < 0.125, 1.0 - p0 - 2.0 * p40, 0) + np.where(
        specparm > 0.875, 1.0 - p0 - 2.0 * p40, 0
    )
    fk10 = np.where(fk10 == 0, fs, fk10)

    fk20 = np.where(specparm < 0.125, p0 + p40, 0) + np.where(
        specparm > 0.875, p0 + p40, 0
    )
    fk20 = np.where(fk20 == 0, 0.0, fk20)

    id000 = np.where(specparm < 0.125, ind0, 0) + np.where(
        specparm > 0.875, ind0 + 1, 0
    )
    id000 = np.where(id000 == 0, ind0, id000)

    id010 = np.where(specparm < 0.125, ind0 + 9, 0) + np.where(
        specparm > 0.875, ind0 + 10, 0
    )
    id010 = np.where(id010 == 0, ind0 + 9, id010)

    id100 = np.where(specparm < 0.125, ind0 + 1, 0) + np.where(
        specparm > 0.875, ind0, 0
    )
    id100 = np.where(id100 == 0, ind0 + 1, id100)

    id110 = np.where(specparm < 0.125, ind0 + 10, 0) + np.where(
        specparm > 0.875, ind0 + 9, 0
    )
    id110 = np.where(id110 == 0, ind0 + 10, id110)

    id200 = np.where(specparm < 0.125, ind0 + 2, 0) + np.where(
        specparm > 0.875, ind0 - 1, 0
    )
    id200 = np.where(id200 == 0, ind0, id200)

    id210 = np.where(specparm < 0.125, ind0 + 11, 0) + np.where(
        specparm > 0.875, ind0 + 8, 0
    )
    id210 = np.where(id210 == 0, ind0, id210)

    fac000 = fk00 * fac00[:laytrop]
    fac100 = fk10 * fac00[:laytrop]
    fac200 = fk20 * fac00[:laytrop]
    fac010 = fk00 * fac10[:laytrop]
    fac110 = fk10 * fac10[:laytrop]
    fac210 = fk20 * fac10[:laytrop]

    p1 = np.where(specparm1 < 0.125, fs1 - 1.0, 0) + np.where(
        specparm1 > 0.875, -fs1, 0
    )
    p1 = np.where(p1 == 0, 0, p1)

    p41 = np.where(specparm1 < 0.125, p1 ** 4, 0) + np.where(
        specparm1 > 0.875, p1 ** 4, 0
    )
    p41 = np.where(p41 == 0, 0, p41)

    fk01 = np.where(specparm1 < 0.125, p41, 0) + np.where(specparm1 > 0.875, p1 ** 4, 0)
    fk01 = np.where(fk01 == 0, 1.0 - fs1, fk01)

    fk11 = np.where(specparm1 < 0.125, 1.0 - p1 - 2.0 * p41, 0) + np.where(
        specparm1 > 0.875, 1.0 - p1 - 2.0 * p41, 0
    )
    fk11 = np.where(fk11 == 0, fs1, fk11)

    fk21 = np.where(specparm1 < 0.125, p1 + p41, 0) + np.where(
        specparm1 > 0.875, p1 + p41, 0
    )
    fk21 = np.where(fk21 == 0, 0.0, fk21)

    id001 = np.where(specparm1 < 0.125, ind1, 0) + np.where(
        specparm1 > 0.875, ind1 + 1, 0
    )
    id001 = np.where(id001 == 0, ind1, id001)

    id011 = np.where(specparm1 < 0.125, ind1 + 9, 0) + np.where(
        specparm1 > 0.875, ind1 + 10, 0
    )
    id011 = np.where(id011 == 0, ind1 + 9, id011)

    id101 = np.where(specparm1 < 0.125, ind1 + 1, 0) + np.where(
        specparm1 > 0.875, ind1, 0
    )
    id101 = np.where(id101 == 0, ind1 + 1, id101)

    id111 = np.where(specparm1 < 0.125, ind1 + 10, 0) + np.where(
        specparm1 > 0.875, ind1 + 9, 0
    )
    id111 = np.where(id111 == 0, ind1 + 10, id111)

    id201 = np.where(specparm1 < 0.125, ind1 + 2, 0) + np.where(
        specparm1 > 0.875, ind1 - 1, 0
    )
    id201 = np.where(id201 == 0, ind1, id201)

    id211 = np.where(specparm1 < 0.125, ind1 + 11, 0) + np.where(
        specparm1 > 0.875, ind1 + 8, 0
    )
    id211 = np.where(id211 == 0, ind1, id211)

    fac001 = fk01 * fac01[:laytrop]
    fac101 = fk11 * fac01[:laytrop]
    fac201 = fk21 * fac01[:laytrop]
    fac011 = fk01 * fac11[:laytrop]
    fac111 = fk11 * fac11[:laytrop]
    fac211 = fk21 * fac11[:laytrop]

    for ig in range(ng16):
        tauself = selffac[:laytrop] * (
            selfref[ig, inds]
            + selffrac[:laytrop] * (selfref[ig, indsp] - selfref[ig, inds])
        )
        taufor = forfac[:laytrop] * (
            forref[ig, indf]
            + forfrac[:laytrop] * (forref[ig, indfp] - forref[ig, indf])
        )

        taug[ns16 + ig, :laytrop] = (
            speccomb
            * (
                fac000 * absa[ig, id000]
                + fac010 * absa[ig, id010]
                + fac100 * absa[ig, id100]
                + fac110 * absa[ig, id110]
                + fac200 * absa[ig, id200]
                + fac210 * absa[ig, id210]
            )
            + speccomb1
            * (
                fac001 * absa[ig, id001]
                + fac011 * absa[ig, id011]
                + fac101 * absa[ig, id101]
                + fac111 * absa[ig, id111]
                + fac201 * absa[ig, id201]
                + fac211 * absa[ig, id211]
            )
            + tauself
            + taufor
        )

        fracs[ns16 + ig, :laytrop] = fracrefa[ig, jpl] + fpl * (
            fracrefa[ig, jplp] - fracrefa[ig, jpl]
        )

    #  --- ...  upper atmosphere loop
    ind0 = ((jp[laytrop:nlay] - 13) * 5 + (jt[laytrop:nlay] - 1)) * nspb[15]
    ind1 = ((jp[laytrop:nlay] - 12) * 5 + (jt1[laytrop:nlay] - 1)) * nspb[15]

    ind0p = ind0 + 1
    ind1p = ind1 + 1

    for ig in range(ng16):
        taug[ns16 + ig, laytrop:nlay] = colamt[laytrop:nlay, 4] * (
            fac00[laytrop:nlay] * absb[ig, ind0]
            + fac10[laytrop:nlay] * absb[ig, ind0p]
            + fac01[laytrop:nlay] * absb[ig, ind1]
            + fac11[laytrop:nlay] * absb[ig, ind1p]
        )

        fracs[ns16 + ig, laytrop:nlay] = fracrefb[ig]

    return taug, fracs
