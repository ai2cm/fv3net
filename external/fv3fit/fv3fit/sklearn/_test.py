from vcm.cloud import fsspec
import joblib
import xarray as xr
import os

from vcm.convenience import round_time
from vcm import safe
import logging

MODEL_FILENAME = "sklearn_model.pkl"
SAMPLE_DIM = "sample"

logger = logging.getLogger(__name__)


def load_test_dataset(
    test_data_path, init_time_dim="initial_time", coord_z_center="z",
):
    """

    Args:
        test_data_path: path to dir that contains test data, which assumed to be saved
        in multiple batches in zarrs
        num_files_to_load: number of files to concat into final test dataset

    Returns:
        xarray dataset created by concatenating test data zarrs
    """
    logger.info("Loading test dataset")
    # TODO I/O buried very deep in the call stack. This should happen at the entrypoint.
    fs = fsspec.get_fs(test_data_path)
    test_data_urls = sorted(fs.ls(test_data_path))
    zarrs_in_test_dir = sorted([file for file in test_data_urls if ".zarr" in file])

    if len(zarrs_in_test_dir) < 1:
        raise ValueError(f"No .zarr files found in  {test_data_path}.")

    ds_test = xr.concat(
        [xr.open_zarr(fs.get_mapper(file_path)) for file_path in zarrs_in_test_dir],
        init_time_dim,
    )
    ds_test = ds_test.assign_coords(
        {init_time_dim: [round_time(t) for t in ds_test[init_time_dim].values]}
    )

    # TODO (noah) dropping these variables is needed to avoid out of memory errors
    # when stack broadcasts x_interface and y_interface
    # Ideally this woudn't be a hard-code, but this code will likely be refactored
    # substantially soon, so not worth parameterizing.  IMO stacking is the
    # responsibility of the xarray wrapping object, not user code like this.
    ds_test = ds_test.drop(["lonb", "latb"])
    ds_stacked = _stack_and_drop_nan_samples(ds_test, coord_z_center)
    return ds_stacked


def _stack_and_drop_nan_samples(ds, coord_z_center):
    """

    Args:
        ds: xarray dataset

    Returns:
        xr dataset stacked into sample dimension and with NaN elements dropped
         (the masked out land/sea type)
    """
    # TODO delete this function
    ds = (
        safe.stack_once(
            ds, SAMPLE_DIM, [dim for dim in ds.dims if dim != coord_z_center]
        )
        .transpose(SAMPLE_DIM, coord_z_center)
        .dropna(SAMPLE_DIM)
    )
    return ds


def predict_dataset(sk_wrapped_model, ds_stacked, vars_to_keep):
    """

    Args:
        sk_wrapped_model_path: location of trained model wrapped in the SklearnWrapper
        ds_stacked: dataset with features and targets data variables, stacked in a
        single sample dimension
        vars_to_keep: features and other variables to keep (e.g. delp) with prediction
        stack_dim: dimension to stack along

    Returns:
        Unstacked prediction dataset
    """
    # TODO implement in sklearn_wrapper
    ds_keep_vars = ds_stacked[vars_to_keep]
    ds_pred = sk_wrapped_model.predict(
        ds_stacked[sk_wrapped_model.input_vars_], SAMPLE_DIM
    )
    return xr.merge([ds_pred, ds_keep_vars]).unstack()


def load_model(model_path):
    protocol = fsspec.get_protocol(model_path)
    if protocol == "gs":
        fs = fsspec.get_fs(model_path)
        fs.get(os.path.join(model_path, MODEL_FILENAME), "temp_model.pkl")
        return joblib.load("temp_model.pkl")
    else:
        return joblib.load(model_path)
