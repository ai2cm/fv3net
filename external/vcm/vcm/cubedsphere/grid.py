import dataclasses


@dataclasses.dataclass
class Grid:
    x: str
    y: str
    x_interface: str
    y_interface: str
    tile: str = "tile"
    lon: str = "lon"
    lonb: str = "lonb"
    lat: str = "lat"
    latb: str = "latb"


GFDL_GRID = Grid("grid_xt", "grid_yt", "grid_x", "grid_y")
PYTHON_DIAGS_GRID = Grid("x", "y", "x_interface", "y_interface")
