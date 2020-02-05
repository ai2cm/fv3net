import fsspec
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..dataset_handler import stack_and_drop_nan_samples
from vcm.calc import mass_integrate, r2_score
from vcm.cloud import gsutil
from vcm.fv3_restarts import _split_url
from vcm.cubedsphere.constants import INIT_TIME_DIM

SAMPLE_DIM = "sample"
kg_m2s_to_mm_day = (1e3 * 86400) / 997.


def load_test_dataset(test_data_path, num_files_to_load=50):
    """

    Args:
        test_data_path: path to dir that contains test data, which assumed to be saved
        in multiple batches in zarrs
        num_files_to_load: number of files to concat into final test dataset

    Returns:
        xarray dataset created by concatenating test data zarrs
    """
    protocol, _ = _split_url(test_data_path)
    fs = fsspec.filesystem(protocol)
    zarrs_in_test_dir = [file for file in fs.ls(test_data_path) if ".zarr" in file]
    if len(zarrs_in_test_dir) < 1:
        raise ValueError(f"No .zarr files found in  {test_data_path}.")
    np.random.shuffle(zarrs_in_test_dir)

    ds_test = xr.concat(
            map(xr.open_zarr,
                [fs.get_mapper(file_path)
                 for file_path in zarrs_in_test_dir[:num_files_to_load]]),
        INIT_TIME_DIM)
    ds_stacked = stack_and_drop_nan_samples(ds_test)
    return ds_stacked


def predict_dataset(sk_wrapped_model_path, ds_stacked):
    """

    Args:
        sk_wrapped_model_path: location of trained model wrapped in the SklearnWrapper
        ds_stacked: dataset with features and targets data variables, stacked in a
        single sample dimension
        stack_dim: dimension to stack along

    Returns:

    """
    sk_wrapped_model = _load_model(sk_wrapped_model_path)
    ds_pred = sk_wrapped_model.predict(
        ds_stacked[sk_wrapped_model.input_vars_], SAMPLE_DIM)
    return ds_pred


def merge_comparison_datasets(
        ds_pred, ds_data, ds_hires, grid):
    """

    Args:
        ds_pred_unstacked:
        ds_data_unstacked:
        ds_hires_unstacked:
        grid:

    Returns:
        Dataset with new dataset dimension to denote the target vs predicted
        quantities. It is unstacked into the original x,y, time dimensions.
    """
    src_dim_index = pd.Index(
        ["coarsened high res", "C48 run", "prediction"],
        name='dataset')
    ds_data = ds_data[["Q1", "Q2"]].unstack()
    ds_comparison = xr.merge(
        [xr.concat([ds_hires, ds_data, ds_pred], src_dim_index), grid])
    return ds_comparison


def _load_model(model_path):
    protocol, _ = _split_url(model_path)
    if protocol == "gs":
        gsutil.copy(model_path, "temp_model.pkl")
        return joblib.load("temp_model.pkl")
    else:
        return joblib.load(model_path)


def make_r2_plot(
        ds_pred,
        ds_target,
        vars,
        output_dir,
        sample_dim=SAMPLE_DIM
):
    if isinstance(vars, str):
        vars = [vars]
    x = ds_pred["pfull"].values
    for var in vars:
        y = r2_score(ds_target[var], ds_pred[var], sample_dim).values
        plt.plot(x, y, label=var)
    plt.legend()
    plt.xlabel("pressure level")
    plt.ylabel("$R^2$")
    plt.savefig(f"{output_dir}/r2_vs_pressure_level.png")


