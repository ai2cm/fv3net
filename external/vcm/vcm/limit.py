import xarray as xr
from .safe import get_variables
from typing import Optional, Sequence, Mapping


class DatasetQuantileLimiter:
    """Transformer to reduce the extremity of outliers of a dataset to quantile limits.

    Limiting is done on a variable by variable basis and along specified dimensions.

    Args:
        upper_quantile_limit: specifies the upper quantile, values above which will
            be reduced to that quantile
        lower_quantile_limit: specifies the lower quantile, values below which will
            be raised to that quantile
        limit_only: variables in the dataset to be limited; if not specified all
            variables are limited

    """

    def __init__(
        self,
        upper_quantile_limit: float,
        lower_quantile_limit: float,
        limit_only: Optional[Sequence[str]] = None,
    ):
        self._upper_quantile_limit: float = upper_quantile_limit
        self._lower_quantile_limit: float = lower_quantile_limit
        self._limit_only: Optional[Sequence[str]] = limit_only
        self._upper: xr.Dataset = None
        self._lower: xr.Dataset = None

    def fit(
        self, ds: xr.Dataset, feature_dims: Optional[Sequence[str]] = None,
    ) -> "DatasetQuantileLimiter":
        """Fit the limiter on a dataset.

        Args:
            ds: Dataset to be used to fit the limits.
            feature_dims: Dimensions along which the fitted quantile limits will
                vary, i.e., quantiles are computed over the dimensions not specified
                in this list. For example, set `feature_dims=['lon']` to apply a
                different limit for each longitude.

        Returns: Fitted limiter

        """
        limit_vars = self._limit_only if self._limit_only is not None else ds.data_vars
        limit_ds = get_variables(ds, limit_vars).load()
        sample_dims = (
            set(limit_ds.dims) - set(feature_dims)
            if feature_dims is not None
            else limit_ds.dims
        )
        self._lower = limit_ds.quantile(
            self._lower_quantile_limit, dim=sample_dims
        ).drop_vars("quantile")
        self._upper = limit_ds.quantile(
            self._upper_quantile_limit, dim=sample_dims
        ).drop_vars("quantile")
        return self

    def transform(self, ds: xr.Dataset, deepcopy: bool = False) -> xr.Dataset:
        """Limit data.

        Args:
            ds: Dataset to be limited
            deepcopy: Whether to make a new copy of ds before applying limits; if
                false the original dataset may be modified

        Returns: Limited dataset

        """
        if self._lower is None and self._upper is None:
            raise ValueError("Limiter method .fit must be called before .transform")
        limited = ds.copy(deep=deepcopy)
        for var in self._limit_only if self._limit_only is not None else ds.data_vars:
            limited[var] = (
                ds[var]
                .where(ds[var] < self._upper[var], self._upper[var])
                .where(ds[var] > self._lower[var], self._lower[var])
            )
        return limited

    @property
    def limits(self) -> Mapping[str, xr.Dataset]:
        """The fitted quantile limits which are applied by the transform method."""
        if self._lower is not None and self._upper is not None:
            return {"lower": self._lower, "upper": self._upper}
        else:
            raise ValueError(
                "Limiter method .fit must be called before accessing limits."
            )
