from typing import Callable, Mapping, Sequence
import xarray as xr


Mapper = Mapping[str, xr.Dataset]
Batches = Sequence[xr.Dataset]

# to strictly type the call signature, we would need to refactor these types
# to have consistent signatures, e.g. by combining the flexible arguments
# in a dataclass similarly to what is done in fv3fit for models
MapperFunction = Callable[..., Mapper]

# call signature uses "data_path" as first arg
BatchesFunction = Callable[..., Batches]

# call signature uses "mapper" as first arg
BatchesFromMapperFunction = Callable[..., Batches]

# if we refactor, the protocols would look something like this:
#
# class MapperFunction(Protocol):
#
#     __name__: str
#
#     def __call__(self, data_path: str, *args, **kwargs) -> Mapper:
#         pass
