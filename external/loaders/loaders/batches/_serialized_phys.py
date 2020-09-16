import logging
import os
import fsspec
import xarray as xr
import numpy as np

from typing import Sequence, Union


logger = logging.getLogger(__name__)


class SerializedSequence(Sequence[xr.Dataset]):
    """
    Create a sequence over savepoints for serialized physics data.
    """

    def __init__(self, xr_data: xr.Dataset, item_dim: str = "savepoint"):
        """
        Args
        ----
        xr_data: Xarray data source
        item_dim: data dimension to use as the "get" index
        """

        self.data = xr_data
        self.item_dim = item_dim

    def __getitem__(self, item: Union[int, slice]):

        return self.data.isel({self.item_dim: item}).load()

    def __len__(self):
        return self.data.sizes[self.item_dim]


class FlattenDims(Sequence[xr.Dataset]):
    """
    Flatten dimensions of sequence dataset item after retrieval
    """

    def __init__(
        self,
        serialized_seq: SerializedSequence,
        dims_to_stack: Sequence[str],
        dim_name: str = "sample",
    ):
        self.seq = serialized_seq
        self.dims_to_stack = dims_to_stack
        self.sample_dim_name = dim_name

    def __getitem__(self, item: Union[int, slice]):
        ds = self.seq[item]
        return self._flatten(ds)

    def __len__(self):
        return len(self.seq)

    def _flatten(self, ds: xr.Dataset):
        # selection of single item removes sample dim from pool
        dims_to_stack = [dim for dim in self.dims_to_stack if dim in ds.dims]
        logger.debug(
            f"Stacking dimensions into {self.sample_dim_name}: {dims_to_stack}"
        )

        stacked_sample_only = ds.stack({self.sample_dim_name: dims_to_stack})
        flat_2d = _separate_by_extra_feature_dim(stacked_sample_only)
        flat_2d = _check_sample_first(flat_2d, self.sample_dim_name)

        return flat_2d


def _separate_by_extra_feature_dim(ds):

    """
    Look for data variables with an extra feature dimension after the
    sample dimension has been stacked.  If there is one, it is a tracer
    field, and separate each tracer out into an individual data variable.
    """

    for var, da in ds.items():
        if da.ndim > 3:
            raise ValueError(
                "Expects at most 3 dimensions (sample, vertical, tracer_no)."
            )
        elif da.ndim == 3:
            tracer_dim = _find_tracer_dim(da.dims)
            logger.debug(
                f"Separating variable ({var}) with extra feature dim ({tracer_dim})"
            )
            tracer_vars = {
                f"{var}_{i}": da.isel({tracer_dim: i})
                for i in range(da.sizes[tracer_dim])
            }
            ds = ds.drop(var)
            ds.update(tracer_vars)

    return ds


def _find_tracer_dim(dims):
    tracer_dims = [dim for dim in dims if "tracer" in dim]
    if len(tracer_dims) > 1:
        raise ValueError(
            "Multiple tracer dimensions detected. Selection of single key is ambiguous."
        )
    else:
        return tracer_dims[0]


def _stack_extra_features(ds, sample_dim_name):

    """
    Need 2D variables for ML, but serialized turbulence had a tracer dim after
    the vertical dimension.  This works for more than a single extra dimension.
    Keeping around in case I want stack instead of
    separation for these multi-feature variables
    """

    combined_feature_dims = {}
    for var, da in ds.items():
        if da.ndim > 2:
            dims_to_stack = tuple(
                sorted([dim for dim in da.dims if dim != sample_dim_name])
            )
            dim_name = combined_feature_dims.get(dims_to_stack, None)
            if dim_name is None:
                dim_name = f"stacked_feature{len(combined_feature_dims) + 1}"
                combined_feature_dims[dims_to_stack] = dim_name

            stack_map = {dim_name: dims_to_stack}
            feature_stacked = da.stack(stack_map)
            ds[var] = feature_stacked.drop(dim_name)

    return ds


def _check_sample_first(ds, sample_dim):
    for var, da in ds.items():
        if da.dims.index(sample_dim) != 0:
            logger.debug(f"Transposing {var} to lead with sampling dimension")
            da = da.transpose()
            ds[var] = da

    return ds


def open_serialized_physics_data(
    path, zarr_prefix: str = "phys", drop_const: bool = True, consolidated=True
) -> xr.Dataset:

    """
    Open xarray dataset of serialized physics component inputs and outputs from a
    zarr file.  Constant variables are dropped by default.

    Args:
    -----
        path: Directory (remote or local) where zarr files are located
        zarr_prefix: Prefix for the two expected zarr files <prefix>_in.zarr and
                     <prefix>_out.zarr
        drop_const: Drop variables from the dataset that appear to be constant
                    from sampling a loaded chunk
    """

    z_phys_in = f"{zarr_prefix}_in.zarr"
    z_phys_out = f"{zarr_prefix}_out.zarr"

    ds_phys_in = xr.open_zarr(
        fsspec.get_mapper(os.path.join(path, z_phys_in)), consolidated=consolidated
    )
    ds_phys_out = xr.open_zarr(
        fsspec.get_mapper(os.path.join(path, z_phys_out)), consolidated=consolidated
    )

    ds_phys = _merge_phys_ds(ds_phys_in, ds_phys_out)

    if drop_const:
        ds_phys = _drop_const_vars(ds_phys)

    return ds_phys


def _merge_phys_ds(in_ds: xr.Dataset, out_ds: xr.Dataset) -> xr.Dataset:

    in_ds = _add_var_suffix(in_ds, "input")
    out_ds = _add_var_suffix(out_ds, "output")
    merged = in_ds.merge(out_ds, join="outer")

    return merged


def _add_var_suffix(ds: xr.Dataset, suffix: str) -> xr.Dataset:

    rename_map = {var: f"{var}_{suffix}" for var in ds.data_vars}
    return ds.rename(rename_map)


def _drop_const_vars(ds: xr.Dataset) -> xr.Dataset:

    for var, da in ds.items():
        logger.debug(f"Checking for constant values in {var}")
        sample = _load_sample_using_chunk(da)
        dtype = sample.dtype

        if not np.issubdtype(dtype, np.number):
            logger.info(f"Removing non-numeric dtype variable {var}")
            ds = ds.drop(var)
            continue

        # Grab zero threshold if numeric
        if np.issubdtype(dtype, np.inexact):
            eps = np.finfo(dtype).eps
        else:
            eps = 0

        # Drop if approximately constant
        selection = {dim: 0 for dim in da.dims}
        item = sample.isel(**selection)
        const = abs(sample - item) < eps
        if const.all():
            logger.info(f"Removing constant-valued variable {var} from dataset.")
            ds = ds.drop(var)

    return ds


def _load_sample_using_chunk(da: xr.DataArray) -> xr.DataArray:

    try:
        chunk = da.data.chunksize
        select = {dim: slice(0, chunk[i]) for i, dim in enumerate(da.dims)}
        sample = da.isel(**select).load()
    except AttributeError:
        logger.debug("No chunksize attribute. Data already loaded.")
        sample = da

    return sample
