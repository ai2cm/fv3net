import dataclasses


@dataclasses.dataclass
class GridMetadata:
    x: str
    y: str
    x_interface: str
    y_interface: str
    tile: str = "tile"
    lon: str = "lon"
    lonb: str = "lonb"
    lat: str = "lat"
    latb: str = "latb"
