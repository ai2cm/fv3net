import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Sequence, Mapping


class DatasetQuantileLimiter(BaseEstimator, TransformerMixin):
    """Transformer to reduce the extremity of outliers of a dataset to quantile limits.
    
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
        self._upper: xr.Dataset = None
        self._lower: xr.Dataset = None

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
                which to fit limits. Indexed dimensions must not be in `feature_dims`.
            
        Returns: Fitted limiter
        
        """
        if fit_indexers is not None and feature_dims is not None:
            for index_dim in fit_indexers.keys():
                if index_dim in feature_dims:
                    raise ValueError(
                        f"Indexer dim {index_dim} may not be in feature_dims."
                    )

        sample_ds = (ds.isel(**fit_indexers) if fit_indexers is not None else ds).load()
        sample_dims = (
            set(sample_ds.dims) - set(feature_dims)
            if feature_dims is not None
            else sample_ds.dims
        )
        self._lower = sample_ds.quantile(self._alpha / 2.0, dim=sample_dims)
        self._upper = sample_ds.quantile(1.0 - self._alpha / 2.0, dim=sample_dims)
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
        vars_ = self._limit_only if self._limit_only is not None else ds.data_vars
        for var in vars_:
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
