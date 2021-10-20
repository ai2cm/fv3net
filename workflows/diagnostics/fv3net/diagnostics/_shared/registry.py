from collections import defaultdict
import logging
from typing import Any, Callable, Mapping, Union, Dict

from joblib import Parallel, delayed
import xarray as xr
from toolz import curry

logger = logging.getLogger(__name__)


class Registry:
    def __init__(
        self, merge: Callable[[Mapping[str, Union[xr.DataArray, xr.Dataset]]], Any]
    ):
        self.funcs = defaultdict()
        self.merge = merge

    @curry
    def register(
        self, name: str, func: Callable[[Any], Union[xr.Dataset, xr.DataArray]]
    ):
        if name in self.funcs:
            raise ValueError(f"Function {name} has already been added to registry.")
        self.funcs[name] = func

    def compute(self, *args, n_jobs=-1, **kwargs) -> Any:
        computed_outputs = Parallel(n_jobs=n_jobs, verbose=True)(
            delayed(self.load)(name, func, *args, **kwargs)
            for name, func in self.funcs.items()
        )
        return self.merge(computed_outputs)

    @staticmethod
    def load(name, func, *args, **kwargs):
        _start_logger_if_necessary()
        return name, func(*args, **kwargs).load()


def _start_logger_if_necessary():
    # workaround for joblib.Parallel logging from
    # https://github.com/joblib/joblib/issues/1017
    logger = logging.getLogger("SaveDiags")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        fh = logging.FileHandler("out.log", mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


def prepare_diag_dict(suffix: str, ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    """
    Take a diagnostic dataset and add a suffix to all variable names and return as dict.
    Useful in multiple merge functions passed to registries.
    """

    diags = {}
    for variable in ds:
        lower = str(variable).lower()
        da = ds[variable]
        diags[f"{lower}_{suffix}"] = da

    return diags
