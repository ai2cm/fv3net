import argparse
import os
from collections import defaultdict

import fv3config
import tensorflow as tf
import wandb
import xarray
import xarray as xr
from fv3fit.emulation.thermobasis.batch import batch_to_specific_humidity_basis, to_dict
from prognostic_diagnostics import open_run, regional_average

ft = "tendency_of_air_temperature_due_to_dynamics"
temperature = "air_temperature"
fqc = "tendency_of_cloud_water_mixing_ratio_due_to_dynamics"
qc = "cloud_water_mixing_ratio"
fu = "tendency_of_eastward_wind_due_to_dynamics"
u = "eastward_wind"
fv = "tendency_of_northward_wind_due_to_dynamics"
v = "northward_wind"
fqv = "tendency_of_specific_humidity_due_to_dynamics"
qv = "specific_humidity"

forcing_variables = [ft, fqc, fqv, ft, fu, fv]


def run_steps(
    ic, forcings, model=lambda x: x, nsteps=100, timestep=900, evolve_winds=True
):
    state = ic.copy()
    yield 0, state
    for i in range(nsteps):
        # dynamics step
        state[temperature] += timestep * forcings[ft]
        state[qv] += timestep * forcings[fqv]
        state[qc] += timestep * forcings[fqc]

        if evolve_winds:
            state[u] += timestep * forcings[fu]
            state[v] += timestep * forcings[fv]

        step = model(state)

        state[temperature] = step[temperature]
        state[qv] = step[qv]
        state[qc] = step[qc]

        if evolve_winds:
            state[u] = step[u]
            state[v] = step[v]

        yield (i + 1) * timestep, state


def collect(states):
    out = defaultdict(list)
    for time, state in states:
        for key in state:
            out[key].append(state[key])
        out["time"].append(time)

    return {key: tf.stack(out[key]) for key in out}


def tf_simulation_to_dataset(simulation) -> xarray.Dataset:
    """

    Args:
        simulation: dictionary of (time, sample, z) shaped tensors
    """
    dims = ["time", "batch", "z"]
    data_vars = {}
    for key in [qv, qc, u, v, temperature]:
        data_vars[key] = (dims, simulation[key].numpy())
    coords = {"time": simulation["time"].numpy().ravel()}
    return xarray.Dataset(data_vars, coords=coords)


def step_emulator(emu, x):
    """
    Args:
        emu: Trainer object
        x: dictionary of inputs
    """
    in_ = batch_to_specific_humidity_basis(x, emu.extra_inputs)
    out = emu.model(in_)
    return to_dict(out)


def generate_forcing_data():
    def open_emulator(url):
        path = os.path.join(url, "fv3config.yml")
        with open(path) as f:
            config = fv3config.load(f)

        emu_path = config["online_emulator"]["emulator"]
        return emu_path

    def strip_prefix(ds, prefix):
        rename = {}
        for key in ds:
            new_key = key[len(prefix) :] if key.startswith(prefix) else key
            rename[key] = new_key
        return ds.rename(rename)

    offline_id = "13cn84im"
    offline_url = f"/scratch/runs/prognostic-runs/{offline_id}/"
    ds = strip_prefix(open_run(offline_url), "emulator_")
    return regional_average(ds)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--no-evolve-winds", action="store_true")
    parser.add_argument("--no-evolve-qc", action="store_true")
    parser.add_argument("--region", type=str, default="tropical ocean")

    args = parser.parse_args()

    job = wandb.init(
        job_type="single-column-run", project="emulator-noah", entity="ai2cm"
    )
    job.config.update(args)

    evolve_winds = not args.no_evolve_winds
    evolve_qc = not args.no_evolve_qc
    model_id = args.model_id
    region = args.region

    artifact = job.use_artifact(model_id, type="model")
    artifact_dir = artifact.download()
    from fv3fit.emulation.thermobasis.emulator import Trainer

    emu = Trainer.load(artifact_dir)
    model = emu.model

    from functools import partial

    import xarray
    from fv3fit.emulation.thermobasis.batch import to_tensors

    try:
        ds = xr.open_dataset("forcings.nc")
    except FileNotFoundError:
        ds = generate_forcing_data().mean("time")
        ds.to_netcdf("forcings.nc")

    time_mean = ds.sel(region=region)
    # prepare input conditions
    time_mean["is_sea"] = xarray.full_like(time_mean["cos_zenith_angle"], True)
    time_mean["is_sea_ice"] = xarray.full_like(time_mean["cos_zenith_angle"], False)
    time_mean["is_land"] = xarray.full_like(time_mean["cos_zenith_angle"], False)
    ic = to_tensors(time_mean[emu.input_variables].expand_dims("sample"))
    forcings = to_tensors(time_mean[forcing_variables])

    # run single column model
    simu = collect(
        run_steps(
            ic,
            forcings,
            model=partial(step_emulator, emu),
            nsteps=1000,
            evolve_winds=evolve_winds,
        )
    )
    simu_ds = tf_simulation_to_dataset(simu)

    # In[7]:

    import matplotlib.pyplot as plt
    import numpy as np
    from fv3fit.tensorboard import plot_to_image

    def to_image(fig: plt.Figure):
        return wandb.Image(plot_to_image(fig))

    def plot(simu, qv, **kwargs):
        hour = 3600
        z = np.arange(79)
        plt.figure()
        plt.pcolormesh(simu["time"] / hour, z, simu[qv].numpy()[:, 0, :].T, **kwargs)
        plt.xlabel("Hours")
        plt.gca().invert_yaxis()
        plt.title(qv)
        plt.colorbar()
        wandb.log({qv: to_image(plt.gcf())})

    def plot_simulation(simu):
        plot(simu, qv, vmin=0, vmax=0.04)
        plot(simu, qc, vmin=0, vmax=1e-4)
        plot(simu, temperature, vmin=200, vmax=320)
        plot(simu, u, vmin=-50, vmax=50)
        plot(simu, v, vmin=-50, vmax=50)

    plot_simulation(simu)
    wandb.finish()
