import argparse
import cftime
import copy
import fsspec
from joblib import delayed, Parallel
import os
import xarray as xr
import yaml
import warnings

import fv3fit
from fv3fit._shared import get_dir, put_dir
from fv3fit._shared.halos import append_halos

warnings.filterwarnings("ignore")

RENAME_MAP = {
    "ocean_surface_temperature_rc_in": "sst",
    "ocean_surface_temperature_rc_out": "sst_out",
    "air_temperature_at_2m_hyb_in": "t2m_at_next_timestep",
    "eastward_wind_at_10m_hyb_in": "u10_at_next_timestep",
    "northward_wind_at_10m_hyb_in": "v10_at_next_timestep",
    "air_temperature_at_2m_rc_in": "t2m_at_next_timestep",
    "eastward_wind_at_10m_rc_in": "u10_at_next_timestep",
    "northward_wind_at_10m_rc_in": "v10_at_next_timestep",
}


def _rename(ds):
    filtered_rename = {k: v for k, v in RENAME_MAP.items() if k in ds}
    return ds.rename(filtered_rename)


def get_synchronization_data(fv3_output_path, overlap, nonhybrid_inputs):

    increment_path = os.path.join(fv3_output_path, "reservoir_incrementer_diags.zarr")
    predict_path = os.path.join(fv3_output_path, "reservoir_predictor_diags.zarr")
    increment = xr.open_zarr(increment_path)
    predict = xr.open_zarr(predict_path)

    sync_data = _rename(increment.drop_vars("time"))
    sync_data = sync_data.merge(_rename(predict).drop_vars(["time", "sst_out"]))

    for_increment = sync_data[nonhybrid_inputs]
    if overlap > 0:
        for_increment = append_halos(for_increment, overlap)

    return for_increment


def sync_model(model, data):
    for i in range(len(data.time)):
        current = data.isel(time=i)
        model.increment_state(current)


def _load_sync_save(model_path, output_path, sync_data):
    with get_dir(model_path) as f:
        model = fv3fit.load(f)
    model.reset_state()
    sync_model(model, sync_data)
    with put_dir(output_path) as f:
        model.dump(f)
        if model.is_hybrid:
            model.model.reservoir.dump_state(f"{f}/hybrid_reservoir_model/reservoir")
        else:
            model.model.reservoir.dump_state(f"{f}/reservoir_model/reservoir")


def get_new_initial_time(ic_dir):
    coupler_file = os.path.join(ic_dir, "coupler.res")
    with fsspec.open(coupler_file, "r") as f:
        lines = f.readlines()
    current_time = [item for item in lines[-1].split(" ") if item][:6]
    return [int(item) for item in current_time]


def sync_models(fv3_output_path, model_map, sync_data):
    model_output_path = os.path.join(fv3_output_path, "artifacts", "synced_models")
    new_model_map = {}
    jobs = []
    for rank, model_path in model_map.items():
        _output_path = os.path.join(model_output_path, f"model_tile{rank}")
        jobs.append(
            delayed(_load_sync_save)(
                model_path, _output_path, sync_data.isel(tile=rank)
            )
        )
        new_model_map[rank] = _output_path

    Parallel(n_jobs=6)(jobs)
    return new_model_map


def print_new_config(fv3_output_path, config, new_model_map):
    init_time = cftime.DatetimeJulian(
        *config["namelist"]["coupler_nml"]["current_date"]
    )
    time_string = init_time.strftime("%Y%m%d.%H%M%S")
    ic_dir = os.path.join(fv3_output_path, "artifacts", time_string, "RESTART")

    new_config = copy.deepcopy(config)
    new_config["initial_conditions"] = ic_dir
    new_config["namelist"]["coupler_nml"]["current_date"] = get_new_initial_time(ic_dir)
    new_config["reservoir_corrector"]["models"] = new_model_map
    del new_config["diag_table"]

    print(yaml.dump(new_config))


def main(fv3_output_path):
    # load config yaml
    with fsspec.open(os.path.join(fv3_output_path, "fv3config.yml"), "r") as f:
        config = yaml.safe_load(f)

    reservoir_config = config["reservoir_corrector"]
    models = reservoir_config["models"]
    with get_dir(models[0]) as f:
        model = fv3fit.load(f)

    sync_data = get_synchronization_data(
        fv3_output_path, model.input_overlap, model.nonhybrid_input_variables,
    )

    if len(models) != 6:
        raise NotImplementedError("Only 6 models supported for now")

    new_model_map = sync_models(fv3_output_path, models, sync_data)

    print_new_config(fv3_output_path, config, new_model_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process fv3 output path")
    parser.add_argument("fv3_output_path", type=str, help="Path to fv3 output")
    args = parser.parse_args()

    main(args.fv3_output_path)
