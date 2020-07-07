import numpy as np

VARNAMES = {
    "delp": "pressure_thickness_of_atmospheric_layer",
    "surface_type": "land_sea_mask",
    "time_dim": "time",
}

SURFACE_TYPE_ENUMERATION = {"sea": 0, "land": 1, "seaice": 2}

NET_PRECIPITATION_ENUMERATION = {
    "negative_net_precipitation": -np.inf,
    "positive_net_precipitation": 0,
}
DOMAINS = [
    "land",
    "sea",
    "global",
    "positive_net_precipitation",
    "negative_net_precipitation",
]

PRIMARY_VARS = ["dQ1", "pQ1", "dQ2", "pQ2"]
