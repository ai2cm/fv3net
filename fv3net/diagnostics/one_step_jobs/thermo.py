from vcm.constants import (
    SPECIFIC_HEAT_CONST_PRESSURE,
    GRAVITY,
    kg_m2s_to_mm_day
)
from vcm.calc import mass_integrate

def psurf_from_delp(delp, p_toa = 300):
    psurf = (delp.sum(dim='z') + p_toa)
    psurf = psurf.assign_attrs({'long_name': 'surface pressure', 'units': 'Pa'})
    return psurf


def total_water(sphum, ice_wat, liq_wat, rainwat, snowwat, graupel, delp):
    total_water = ((delp/GRAVITY)*(sphum + ice_wat + liq_wat + rainwat + snowwat + graupel)).sum('z')
    total_water = total_water.assign_attrs({'long_name': 'total water species', 'units': 'mm'})
    return total_water


def precipitable_water(sphum, delp):
    pw = ((delp/GRAVITY)*sphum).sum('z')
    pw = pw.assign_attrs({'long_name' : 'precipitable water', 'units': 'mm'})
    return pw


def total_heat(T, delp):
    total_heat =  SPECIFIC_HEAT_CONST_PRESSURE*(delp/GRAVITY)*T.sum("z")
    total_heat = total_heat.assign_attrs({'long_name' : "column integrated heat", "units": "J/m**2"})
    return total_heat


def column_integrated_heating(dT_dt, delp):
    dT_dt_integrated = SPECIFIC_HEAT_CONST_PRESSURE * mass_integrate(
        dT_dt, delp, dim = 'z')
    dT_dt_integrated = dT_dt_integrated.assign_attrs({
        'long_name': 'column integrated heating',
        'units': "W/m^2"
    })
    return dT_dt_integrated


def column_integrated_moistening(dsphum_dt, delp):
    dsphum_dt_integrated = kg_m2s_to_mm_day*mass_integrate(
        dsphum_dt, delp, dim='z')
    dsphum_dt_integrated = dsphum_dt_integrated.assign_attrs({
        'long_name': 'column integrated moistening',
        'units': "mm/d"
    })
    return dsphum_dt_integrated



