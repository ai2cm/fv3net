import fv3fit.sklearn._train
import metadata
import xarray
from loaders.mappers._fine_resolution_budget import convergence


def compute_targets(dataset, vertical_dim="pfull"):
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


def compute_inputs(dataset):
    output = {}
    output["air_temperature"] = dataset["T"]
    output["specific_humidity"] = dataset["sphum"]
    return xarray.Dataset(output)


def train(url, output):
    dataset = xarray.open_zarr(url)
    normalize = metadata.gfdl_to_standard(dataset)
    targets = compute_targets(normalize, vertical_dim="z")
    inputs = compute_inputs(normalize)

    ml_data = xarray.merge([inputs, targets])

    assert False
