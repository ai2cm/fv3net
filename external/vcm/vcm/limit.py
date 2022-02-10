import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Sequence, Mapping, Union


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


class DatasetQuantileLimiter(BaseEstimator, TransformerMixin):
    """Transformer to reduce the extremity of outliers of a dataset to quantile limits
    
    Limiting is done on a variable by variable basis and along specified dimensions.
        Limits are optionally computed on a configurable subset of the dataset to avoid
        loading the entire dataset.
    
    Args:
        alpha: two-tailed alpha for computing extrema quantiles, values beyond which
            will be reduced to the quantile
        limit_only: variables in the dataset to be limited; if not specified all
            variables are limited

    """

    def __init__(
        self, alpha: float, limit_only: Optional[Sequence[str]] = None,
    ):
        self._alpha: float = alpha
        self._limit_only: Optional[Sequence[str]] = limit_only
        self._limits: xr.Dataset = None

    def fit(
        self,
        ds: xr.Dataset,
        feature_dims: Optional[Sequence[str]] = None,
        fit_indexers: Optional[Mapping[str, int]] = None,
    ) -> "DatasetQuantileLimiter":
        """Fit the limiter on a dataset.
        
        Args:
            ds: Dataset to be used to fit the limits.
            feature_dims: Dimensions along which quantile limits should NOT be
                computed, i.e., the resulting limits will be computed separately
                for each coordinate along the feature_dims.
            fit_indexer: Integer-based indexers that select a subset of `ds` on
                which to fit limits
            
        Returns: Fitted limiter
        
        """
        sample_ds = (ds.isel(**fit_indexers) if fit_indexers is not None else ds).load()
        sample_dims = (
            set(sample_ds.dims) - set(feature_dims)
            if feature_dims is not None
            else sample_ds.dims
        )
        lower = sample_ds.quantile(self._alpha / 2.0, dim=sample_dims)
        upper = sample_ds.quantile(1.0 - self._alpha / 2.0, dim=sample_dims)
        self._limits = xr.concat(
            [lower, upper],
            dim=xr.DataArray(["lower", "upper"], dims=["bounds"], name="bounds"),
        )
        return self

    def transform(
        self, data: Union[xr.Dataset, xr.DataArray]
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Limit data.
        
        Args:
            data: Dataset or dataarray to be limited
            
        Returns: Limited dataset or dataarray
        
        """
        if self._limits is None:
            raise ValueError("Limiter method .fit must be called before .transform")
        if isinstance(data, xr.Dataset):
            limited = data.copy()
            vars_ = self._limit_only if self._limit_only is not None else data.data_vars
            for var in vars_:
                limited[var] = _limit_extremes(data[var], self._limits[var])
        elif isinstance(data, xr.DataArray):
            if (self._limit_only is None) or (data.name in self._limit_only):
                limited = _limit_extremes(data, self._limits[data.name])
            else:
                limited = data
        return limited

    @property
    def limits(self) -> xr.Dataset:
        """The fitted quantile limits which are applied by the transform method."""
        return self._limits
