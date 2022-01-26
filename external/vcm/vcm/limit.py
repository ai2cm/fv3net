import xarray as xr
from typing import Sequence, Mapping, Union, Optional


def _limit_extremes(data: xr.DataArray, limits: xr.DataArray) -> xr.DataArray:
    """Limit values beyond prescribed extremes in a data array
    
    Args:
        data: Data to be limited
        limits: variable containing limits of the data, which must include a 'bounds'
            dimension containing 'upper' and 'lower' coordinates. Other dimensions
            should be a subset of the data dimensions.
            
    Returns: a limited data array
    
    """
    upper = limits.sel(bounds="upper")
    lower = limits.sel(bounds="lower")
    return data.where(data < upper, upper).where(data > lower, lower)


class LimitedDataset(xr.Dataset):
    """A dataset with outlier values limited in magnitude.
    
    This wraps xr.Dataset, specifying at what percentile values should be limited
    in the resulting dataset and over what dimensions and subsets the percentiles
    should be computed.

    Args:
        ds: dataset to be limited
        alpha: two-tailed alpha for computing extrema quantiles, values beyond
            which will be reduced to that quantile; e.g., alpha=0.10 will result in
            all values of greater than 95th or less than 5th percentile being
            changed to the 95th and 5th percentile values, respectively.
        limit_only: sequence of variable names to limit; if provided other
            variables in the dataset will not be limited
        feature_dims: dimensions along which distinct quantiles should be used; i.e.,
            dimensions along which the quantiles will not be computed
        fit_indexers: indexers to a subset of sample dimensions that will be used
            to compute the quantiles; useful because the quantile function forces
            loading the data

    """

    def __init__(
        self,
        ds: xr.Dataset,
        alpha: float,
        limit_only: Optional[Sequence[str]] = None,
        feature_dims: Optional[Sequence[str]] = None,
        fit_indexers: Optional[Mapping[str, int]] = None,
    ):
        super().__init__(ds)
        self._alpha: float = alpha
        self._limit_only: Optional[Sequence[str]] = limit_only
        self._feature_dims: Optional[Sequence[str]] = feature_dims
        self._fit_indexers: Optional[Mapping[str, int]] = fit_indexers
        self._limits: xr.Dataset = self._fit_limits(ds)

    def __getitem__(self, key):
        return self._limit_extremes(super().__getitem__(key))

    def _fit_limits(self, ds: xr.Dataset):
        sample_ds = (
            ds.isel(**self._fit_indexers) if self._fit_indexers is not None else ds
        ).load()
        sample_dims = (
            set(sample_ds.dims) - set(self._feature_dims)
            if self._feature_dims is not None
            else sample_ds.dims
        )
        lower = sample_ds.quantile(self._alpha / 2.0, dim=sample_dims)
        upper = sample_ds.quantile(1.0 - self._alpha / 2.0, dim=sample_dims)
        return xr.concat(
            [lower, upper],
            dim=xr.DataArray(["lower", "upper"], dims=["bounds"], name="bounds"),
        )

    def _limit_extremes(self, data: Union[xr.Dataset, xr.DataArray]):
        if isinstance(data, xr.Dataset):
            limited = data
            vars_ = self._limit_only if self._limit_only is not None else data.data_vars
            for var in vars_:
                limited[var] = _limit_extremes(data[var], self._limits[var])
        elif isinstance(data, xr.DataArray):
            if (self._limit_only is None) or (data.name in self._limit_only):
                limited = _limit_extremes(data, self._limits[data.name])
            else:
                limited = data
        return limited
