gravity = 9.81
specific_heat = 1004


def mass_integrate(phi, dp, dim='pfull'):
    return (phi * dp / gravity).sum(dim)


def apparent_heating(dtemp_dt, w):
    return dtemp_dt + w * gravity / specific_heat
