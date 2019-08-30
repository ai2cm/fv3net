gravity = 9.81


def mass_integrate(phi, dp):
    return (phi * dp / gravity).sum("pfull")
