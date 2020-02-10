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
    0.0,
    700.0,
    1100.0,
    1500.0,
    2000.0,
    2400.0,
    2800.0,
    3300.0,
    3800.0,
    4300.0,
    4900.0,
    5500.0,
    6200.0,
    6900.0,
    7700.0,
    8600.0,
    9400.0,
    10400.0,
    11400.0,
    12400.0,
    13600.0,
    14700.0,
    16000.0,
    17200.0,
    18600.0,
    19900.0,
    21400.0,
    22900.0,
    24400.0,
    26000.0,
    27600.0,
    29200.0,
    30900.0,
    32600.0,
    34400.0,
    36200.0,
    38000.0,
    39800.0,
    41600.0,
    43500.0,
    45400.0,
    47300.0,
    49200.0,
    51100.0,
    53000.0,
    54900.0,
    56900.0,
    58800.0,
    60700.0,
    62600.0,
    64500.0,
    66400.0,
    68300.0,
    70100.0,
    72000.0,
    73800.0,
    75600.0,
    77400.0,
    79100.0,
    80800.0,
    82500.0,
    84100.0,
    85700.0,
    87200.0,
    88700.0,
    90100.0,
    91400.0,
    92700.0,
    93900.0,
    95000.0,
    96000.0,
    97000.0,
    97900.0,
    98700.0,
    99400.0,
    100000.0,
    100600.0,
    101000.0,
    101400.0,
]
