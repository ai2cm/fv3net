VARNAMES = {
    "delp": "pressure_thickness_of_atmospheric_layer",
    "surface_type": "land_sea_mask",
    "time_dim": "time",
}

SURFACE_TYPE_ENUMERATION = {"sea": 0, "land": 1, "seaice": 2}

# information to calculate regridded pressure midpoints
TOA_PRESSURE = 300.
REGRIDDED_DELP = [ 200.,  200.,  300., 1000., 1000., 2000., 2000., 3000., 2500.,
       2500., 2500., 2500., 2500., 2500., 5000., 5000., 5000., 5000.,
       5000., 5000., 5000., 5000., 5000., 5000., 2500., 2500., 2500.,
       2500., 2500., 2500., 2500., 2500., 2500., 2500.]