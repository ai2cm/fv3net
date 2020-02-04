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
FV_TRACER_X_CENTER = "xaxis_1"
FV_TRACER_Y_CENTER = "yaxis_1"
RESTART_Z_CENTER = "zaxis_1"
RESTART_Z_OUTER = "zaxis_2"
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
RESTART_CATEGORIES = ["fv_core.res", "sfc_data", "fv_tracer", "fv_srf_wnd.res"]
GRID_VARS = [VAR_LAT_CENTER, VAR_LAT_OUTER, VAR_LON_CENTER, VAR_LON_OUTER, "area"]
INIT_TIME_DIM = "initialization_time"
FORECAST_TIME_DIM = "forecast_time"
TILE_COORDS = range(
    6
)  # note that we changed to start index at 0, so some older data might start at 1
