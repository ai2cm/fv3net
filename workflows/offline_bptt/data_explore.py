import xarray as xr
import fsspec
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "gs://vcm-ml-experiments/2020-10-09-physics-on-nudge-to-fine"  # 1d timescale

print("loading zarrs")

before_dynamics = xr.open_zarr(
    fsspec.get_mapper(os.path.join(DATA_DIR, "before_dynamics.zarr"))
)
print(".")
after_physics = xr.open_zarr(
    fsspec.get_mapper(os.path.join(DATA_DIR, "after_physics.zarr"))
)
print(".")
after_nudging = xr.open_zarr(
    fsspec.get_mapper(os.path.join(DATA_DIR, "after_nudging.zarr"))
)
print(".")

print("zarrs loaded, computing plot values")

n_steps = 4 * 24 * 1
before_dynamics_plot = before_dynamics.isel(time=slice(0, n_steps), tile=0, x=24, y=24)
after_physics_plot = after_physics.isel(time=slice(0, n_steps), tile=0, x=24, y=24)
after_nudging_plot = after_nudging.isel(time=slice(0, n_steps), tile=0, x=24, y=24)

name = "specific_humidity"

after_physics_diff = after_physics_plot[name].values - before_dynamics_plot[name].values
after_nudging_diff = after_nudging_plot[name].values - after_physics_plot[name].values
plot_values = [after_physics_diff, after_nudging_diff]

step_diff = np.diff(after_physics_plot[name].values[::4], axis=0)
plt.figure()
im = plt.pcolormesh(after_physics_diff.T * 4)
plt.colorbar(im)
plt.figure()
print(after_physics_plot[name])  # confirm that dimensions are [time, height]
im = plt.pcolormesh(step_diff.T)
plt.colorbar(im)

# plot_values = [
#     before_dynamics_plot[name].values,
#     after_physics_plot[name].values,
#     after_nudging_plot[name].values
# ]

vmin = min(val.min() for val in plot_values)
vmax = max(val.max() for val in plot_values)

print(vmin, vmax)

fig, ax = plt.subplots(len(plot_values), 1, figsize=(8, 5), sharex=True, sharey=True)
for i, plot_value in enumerate(plot_values):
    im = ax[i].pcolormesh(plot_value.T, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax[i])

plt.tight_layout()
plt.show()
