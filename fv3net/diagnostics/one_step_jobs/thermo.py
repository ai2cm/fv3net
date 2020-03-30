from vcm.constants import (
    SPECIFIC_HEAT_CONST_PRESSURE,
    GRAVITY,
    kg_m2s_to_mm_day
)

def _psurf_from_delp(delp, p_toa = 300):
    psurf = (delp.sum(dim='z') + p_toa)
    psurf = psurf.assign_attrs({'long_name': 'surface pressure', 'units': 'Pa'})
    return psurf


def _total_water(sphum, ice_wat, liq_wat, rainwat, snowwat, graupel, delp):
    total_water = ((delp/GRAVITY)*(sphum + ice_wat + liq_wat + rainwat + snowwat + graupel)).sum('z')
    total_water = total_water.assign_attrs({'long_name': 'total water species', 'units': 'mm'})
    return total_water


def _precipitable_water(sphum, delp):
    pw = ((delp/GRAVITY)*sphum).sum('z')
    pw = pw.assign_attrs({'long_name' : 'precipitable water', 'units': 'mm'})
    return pw


def _total_heat(T, delp):
    total_heat =  SPECIFIC_HEAT_CONST_PRESSURE*(delp/GRAVITY)*T.sum("z")
    total_heat = total_heat.assign_attrs({'long_name' : "column integrated heat", "units": "J/m**2"})
    return total_heat