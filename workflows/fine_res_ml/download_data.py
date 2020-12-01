import logging
import joblib
import loaders.batches
import loaders.mappers
import numpy as np
import vcm
import xarray
from budget.data import open_merged
from loaders.mappers._fine_resolution_budget import convergence
from vcm import metadata

logging.basicConfig(level=logging.INFO)


def fine_res_to_zarr(url, output, seed=0):
    np.random.seed(seed)
    mapper = loaders.mappers.open_fine_resolution_budget(url=url)
    train = list(np.random.choice(list(mapper), 130, replace=False))
    template = mapper[train[0]].drop("time")
    output_mapper = vcm.ZarrMapping.from_schema(
        store=output, schema=template, dims=["time"], coords={"time": train}
    )

    def process(time):
        logging.info(f"Processing {time}")
        output_mapper[[time]] = mapper[time]

    joblib.Parallel(6)(joblib.delayed(process)(time) for time in train)


def _compute_targets(dataset, vertical_dim="pfull"):
    output = xarray.Dataset({})

    eddy_temp = (
        dataset.T_vulcan_omega_coarse
        - dataset["T"] * dataset["vulcan_omega_coarse"]
        + dataset.eddy_flux_vulcan_omega_temp
    )
    output["dQ1"] = (
        dataset.t_dt_fv_sat_adj_coarse
        + dataset.dt3dt_mp_coarse
        - convergence(eddy_temp, dataset.delp, dim=vertical_dim)
    )

    eddy_sphum = (
        dataset.sphum_vulcan_omega_coarse
        - dataset["sphum"] * dataset["vulcan_omega_coarse"]
        + dataset.eddy_flux_vulcan_omega_sphum
    )
    output["dQ2"] = (
        dataset.qv_dt_fv_sat_adj_coarse
        + dataset.dq3dt_mp_coarse
        - convergence(eddy_sphum, dataset.delp, dim=vertical_dim)
    )

    return output


def _compute_inputs(dataset):
    output = {}
    output["air_temperature"] = dataset["T"]
    output["specific_humidity"] = dataset["sphum"]
    return xarray.Dataset(output)


def save_fine_res_to_zarr(url, output):
    dataset = xarray.open_zarr(url)
    normalize = metadata.gfdl_to_standard(dataset)
    targets = _compute_targets(normalize, vertical_dim="z")
    inputs = _compute_inputs(normalize)

    ml_data = xarray.merge([inputs, targets])
    ml_data = ml_data.assign_coords(
        time=np.vectorize(vcm.parse_datetime_from_str)(ml_data.time)
    )
    ml_data.to_zarr(output, mode='w')
