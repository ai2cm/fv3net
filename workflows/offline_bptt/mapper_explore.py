import matplotlib.pyplot as plt
import loaders

DATA_DIR = "gs://vcm-ml-experiments/2020-06-17-triad-round-1/nudging/nudging/outdir-3h"
NUDGE_DATA_DIR = (
    "gs://vcm-ml-experiments/2020-10-09-physics-on-nudge-to-fine"  # 1d timescale
)

before_dynamics = loaders.mappers.open_merged_nudged(
    NUDGE_DATA_DIR, merge_files=("before_dynamics.zarr", "nudging_tendencies.zarr"),
)
after_physics = loaders.mappers.open_merged_nudged(
    NUDGE_DATA_DIR, merge_files=("after_physics.zarr", "nudging_tendencies.zarr"),
)
reference = loaders.mappers.open_merged_nudged(
    NUDGE_DATA_DIR, merge_files=("reference.zarr", "nudging_tendencies.zarr"),
)


shared_keys = sorted(
    list(
        set(before_dynamics.keys()).union(reference.keys().union(after_physics.keys()))
    )
)


n_steps = 4 * 24 * 5
before_dynamics_plot = before_dynamics[shared_keys[0]].isel(tile=0, x=24, y=24)
after_physics_plot = after_physics[shared_keys[0]].isel(tile=0, x=24, y=24)
after_nudging = None
after_nudging_plot = after_nudging[shared_keys[0]].isel(tile=0, x=24, y=24)

name = "specific_humidity"

after_physics_diff = after_physics_plot[name].values - before_dynamics_plot[name].values
after_nudging_diff = after_nudging_plot[name].values - after_physics_plot[name].values
plot_values = [after_physics_diff, after_nudging_diff]

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
