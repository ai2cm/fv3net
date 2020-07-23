VARNAMES = {
    "delp": "pressure_thickness_of_atmospheric_layer",
    "surface_type": "land_sea_mask",
    "time_dim": "time",
}

SURFACE_TYPE_ENUMERATION = {"sea": 0, "land": 1, "seaice": 2}

# information to calculate regridded pressure midpoints
TOA_PRESSURE = 300.0
# corresponds to the pressure grid in vcm.cubedsphere.constants, starting at 300 Pa
REGRIDDED_DELP = [
    200.0,
    200.0,
    300.0,
    1000.0,
    1000.0,
    2000.0,
    2000.0,
    3000.0,
    2500.0,
    2500.0,
    2500.0,
    2500.0,
    2500.0,
    2500.0,
    5000.0,
    5000.0,
    5000.0,
    5000.0,
    5000.0,
    5000.0,
    5000.0,
    5000.0,
    5000.0,
    5000.0,
    2500.0,
    2500.0,
    2500.0,
    2500.0,
    2500.0,
    2500.0,
    2500.0,
    2500.0,
    2500.0,
    2500.0,
]
