from collections import defaultdict
from typing import Any, Callable, Mapping, Union

import xarray as xr
from toolz import curry


class Registry:
    def __init__(
        self, merge_func: Callable[[Mapping[str, Union[xr.DataArray, xr.Dataset]]], Any]
    ):
        self._funcs = defaultdict()
        self.merge_func = merge_func

    @curry
    def register(
        self, name: str, func: Callable[[Any], Union[xr.Dataset, xr.DataArray]]
    ):
        if name in self._funcs:
            raise ValueError(f"Function {name} has already been added to registry.")
        self._funcs[name] = func

    def compute(self, *args, **kwargs) -> Any:
        computed_outputs = {}
        for name, func in self._funcs.items():
            computed_outputs[name] = func(*args, **kwargs)
        return self.merge_func(computed_outputs)
