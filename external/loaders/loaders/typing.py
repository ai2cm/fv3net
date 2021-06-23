from typing import Mapping, Sequence
from typing_extensions import Protocol
import xarray as xr


Mapper = Mapping[str, xr.Dataset]
Batches = Sequence[xr.Dataset]


class MapperFunction(Protocol):
    def __call__(self, data_path: str, *args, **kwargs) -> Mapper:
        pass


class BatchesFunction(Protocol):
    def __call__(self, data_path: str, *args, **kwargs) -> Batches:
        pass
