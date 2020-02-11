COORD_X_CENTER = "grid_xt"
COORD_X_OUTER = "grid_x"
COORD_Y_CENTER = "grid_yt"
COORD_Y_OUTER = "grid_y"
COORD_Z_CENTER = "pfull"
COORD_Z_OUTER = "phalf"
COORD_Z_SOIL = "soil_layer"
FV_CORE_X_CENTER = "xaxis_1"
FV_CORE_Y_CENTER = "yaxis_2"
FV_CORE_X_OUTER = "xaxis_2"
FV_CORE_Y_OUTER = "yaxis_1"
FV_SRF_WND_X_CENTER = "xaxis_1"
FV_SRF_WND_Y_CENTER = "yaxis_1"
FV_TRACER_X_CENTER = "xaxis_1"
FV_TRACER_Y_CENTER = "yaxis_1"
RESTART_Z_CENTER = "zaxis_1"
RESTART_Z_OUTER = "zaxis_2"
SFC_DATA_X_CENTER = "xaxis_1"
SFC_DATA_Y_CENTER = "yaxis_1"
VAR_LON_CENTER = "lon"
VAR_LAT_CENTER = "lat"
VAR_LON_OUTER = "lonb"
VAR_LAT_OUTER = "latb"
VAR_GRID_LON_CENTER = "grid_lont"
VAR_GRID_LAT_CENTER = "grid_latt"
VAR_GRID_LON_OUTER = "grid_lon"
VAR_GRID_LAT_OUTER = "grid_lat"
INIT_TIME_DIM = "initialization_time"
FORECAST_TIME_DIM = "forecast_time"
TIME_FMT = "%Y%m%d.%H%M%S"
RESTART_CATEGORIES = ["fv_core.res", "sfc_data", "fv_tracer.res", "fv_srf_wnd.res"]
GRID_VARS = [VAR_LAT_CENTER, VAR_LAT_OUTER, VAR_LON_CENTER, VAR_LON_OUTER, "area"]
INIT_TIME_DIM = "initialization_time"
FORECAST_TIME_DIM = "forecast_time"
TILE_COORDS = range(6)
TILE_COORDS_FILENAMES = range(1, 7)  # tile numbering in model output filenames

# for use in regridding values to the same vertical grid [Pa]
PRESSURE_GRID = [
    100000.0,
    97500.0,
    95000.0,
    92500.0,
    90000.0,
    87500.0,
    85000.0,
    82500.0,
    80000.0,
    77500.0,
    75000.0,
    70000.0,
    65000.0,
    60000.0,
    55000.0,
    50000.0,
    45000.0,
    40000.0,
    35000.0,
    30000.0,
    25000.0,
    22500.0,
    20000.0,
    17500.0,
    15000.0,
    12500.0,
    10000.0,
    7000.0,
    5000.0,
    3000.0,
    2000.0,
    1000.0,
    700.0,
    500.0,
    300.0,
    200.0,
    100.0,
]
