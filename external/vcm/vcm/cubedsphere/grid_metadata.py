import dataclasses
import abc


class GridMetadata(abc.ABC):
    @property
    @abc.abstractmethod
    def coord_vars(self) -> dict:
        ...


@dataclasses.dataclass
class GridMetadataFV3(GridMetadata):
    x: str = "x"
    y: str = "y"
    x_interface: str = "x_interface"
    y_interface: str = "y_interface"
    tile: str = "tile"
    lon: str = "lon"
    lonb: str = "lonb"
    lat: str = "lat"
    latb: str = "latb"

    @property
    def coord_vars(self):
        coord_vars = {
            self.lonb: [self.y_interface, self.x_interface, self.tile],
            self.latb: [self.y_interface, self.x_interface, self.tile],
            self.lon: [self.y, self.x, self.tile],
            self.lat: [self.y, self.x, self.tile],
        }
        return coord_vars


@dataclasses.dataclass
class GridMetadataScream(GridMetadata):
    x: str = "x"
    y: str = "y"
    tile: str = "tile"
    lon: str = "lon"
    lat: str = "lat"

    @property
    def coord_vars(self):
        coord_vars = {
            self.lon: [self.y, self.x, self.tile],
            self.lat: [self.y, self.x, self.tile],
        }
        return coord_vars
