import logging
import joblib
import loaders.batches
import loaders.mappers
import numpy as np
import vcm
import xarray
from loaders.mappers._fine_resolution_budget import convergence
from vcm import metadata
import vcm.catalog
import datetime
import fv3post

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
    output["surface_precipitation_rate"] = dataset["PRATEsfc_coarse"]
    output["convective_surface_precipitation_rate"] = dataset["CPRATEsfc_coarse"]
    output["latent_heat_flux"] = dataset["LHTFLsfc_coarse"]
    output["sensible_heat_flux"] = dataset["SHTFLsfc_coarse"]

    one_g_per_kg_per_day = 1 / 86400 / 1000
    one_k_per_day = 1 / 86400
    no_pbl = (np.abs(dataset.dq3dt_pbl_coarse) < one_g_per_kg_per_day) | (
        np.abs(dataset.dt3dt_pbl_coarse) < one_k_per_day
    )

    eddy_temp = (
        dataset.T_vulcan_omega_coarse
        - dataset["T"] * dataset["vulcan_omega_coarse"]
        + dataset.eddy_flux_vulcan_omega_temp
    )
    output["dQ1"] = (
        dataset.t_dt_fv_sat_adj_coarse
        + dataset.dt3dt_mp_coarse
        - convergence(eddy_temp, dataset.delp, dim=vertical_dim).where(no_pbl, 0)
    )

    eddy_sphum = (
        dataset.sphum_vulcan_omega_coarse
        - dataset["sphum"] * dataset["vulcan_omega_coarse"]
        + dataset.eddy_flux_vulcan_omega_sphum
    )
    output["dQ2"] = (
        dataset.qv_dt_fv_sat_adj_coarse
        + dataset.dq3dt_mp_coarse
        - convergence(eddy_sphum, dataset.delp, dim=vertical_dim).where(no_pbl, 0)
    )

    return output


def _compute_inputs(dataset):
    output = {}
    output["air_temperature"] = dataset["T"]
    output["specific_humidity"] = dataset["sphum"]
    output["pressure_thickness_of_atmospheric_layer"] = dataset["delp"]
    return xarray.Dataset(output)


def save_fine_res_to_zarr(url, output):
    # open 3d
    data_3d = xarray.open_zarr(url)
    data_3d["time"] = time = np.vectorize(vcm.parse_datetime_from_str)(data_3d.time)
    # open 2d data
    ds_2d = vcm.catalog.catalog["40day_c48_gfsphysics_15min_may2020"].to_dask()
    time = vcm.convenience.round_time(ds_2d.time) - datetime.timedelta(
        minutes=7, seconds=30
    )
    ds_2d["time"] = time
    merged = xarray.merge([ds_2d, data_3d], join="inner")
    normalized = metadata.gfdl_to_standard(merged)
    targets = _compute_targets(normalized, vertical_dim="z")
    inputs = _compute_inputs(normalized)
    ml_data = xarray.merge([inputs, targets])
    chunked = ml_data.chunk({"x": -1, "y": -1, "z": -1, "tile": -1, "time": 1})
    fv3post.clear_encoding(chunked)
    chunked.to_zarr(output, mode="w")
