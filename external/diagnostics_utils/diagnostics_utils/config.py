VARNAMES = {
    "delp": "pressure_thickness_of_atmospheric_layer",
    "surface_type": "land_sea_mask",
    "time_dim": "time",
}
# for domain averaging, sea ice (2) is counted as sea
SURFACE_TYPE_ENUMERATION = {0: "sea", 1: "land", 2: "sea"}
DOMAINS = (
    "land",
    "sea",
    "global",
    "positive_net_precipitation",
    "negative_net_precipitation",
)

PRIMARY_VARS = ("dQ1", "pQ1", "dQ2", "pQ2", "Q1", "Q2")
