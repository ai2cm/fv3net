# flake8: noqa
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3.8.12 ('fv3net')
#     language: python
#     name: python3
# ---

# %%
import xarray
import fv3viz
import vcm.fv3
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-colorblind")

# %% [markdown]
# Analyze a gscond-only simulation.

# %%
import fsspec
import vcm.catalog

url = "gs://vcm-ml-experiments/microphysics-emulation/2022-05-12/rnn-b0d7d8-v3-prog-v6-online"
m = fsspec.get_mapper(url + "/atmos_dt_atmos.zarr")
ds = xarray.open_zarr(m)

m = fsspec.get_mapper(url + "/piggy.zarr")
ds_p = xarray.open_zarr(m)

m = fsspec.get_mapper(url + "/state_after_timestep.zarr")
ds_s = xarray.open_zarr(m)


ds_g = vcm.fv3.metadata.gfdl_to_standard(ds)
ds_p = vcm.fv3.metadata.gfdl_to_standard(ds_p)
ds_ds = vcm.fv3.metadata.gfdl_to_standard(ds_s)
grid = vcm.catalog.catalog["grid/c48"].to_dask()
merged = grid.merge(ds_g.drop(list(grid))).merge(ds_p, join="inner").merge(ds_s)


# %%
fv3viz.plot_cube(merged.isel(time=0), "PWAT", vmax=80)

# %%
fv3viz.plot_cube(merged.isel(time=-2), "PWAT", vmax=80)

# %%
def plot_cloud(merged):
    QC = "cloud_water_mixing_ratio"
    subset = merged.isel(time=-1, z=0)
    cloud = subset[QC]
    subset[QC] = cloud.where(cloud > 1e-15, 1e-15)
    fv3viz.plot_cube(subset, QC, norm=matplotlib.colors.LogNorm())
    plt.title(subset.time.item().isoformat())


plot_cloud(merged)

# %%
def plot_humidity_tendency(merged):
    field = "tendency_of_specific_humidity_due_to_gscond_emulator"
    subset = merged.isel(time=0).sel(z=200, method="nearest")
    subset[field] = subset[field] * 86400 * 1000
    subset[field].attrs["units"] = "g/kg/d"
    fv3viz.plot_cube(subset, field)
    plt.title(subset.time.item().isoformat())


plot_humidity_tendency(merged)

# %% [markdown]
# The drifts are everywhere, but especially near antarctica. What are the piggy tends doing?

# %%
def plot():
    #         dt_g_450 = merged.tendency_of_air_temperature_due_to_zhao_carr_physics.sel(z=450, method='nearest')
    field = "air_temperature"
    ds = xarray.Dataset(
        {
            "physics": merged[f"tendency_of_{field}_due_to_gscond_physics"],
            "emulator": merged[f"tendency_of_air_temperature_due_to_gscond_emulator"],
        }
    )
    array = ds.to_array() / 2.51e6 * 1004
    f = vcm.zonal_average_approximate(merged.lat, array.sel(z=450, method="nearest"))
    f.plot(col="variable", figsize=(12, 8), vmax=1e-8)


plot()

# %% [markdown]
# There is some lack of heat conservation.

# %%
sample = merged.isel(time=slice(0, None, 20))

# %%
import re

subset = sample[
    [v for v in sample if re.match(r".*(tendency|.*temperature.*|humidity.*|lat)$", v)]
]
dq = subset.to_dataframe()

# %%
import matplotlib.colors


# %%
def plot_1_1(x, y, l=1):
    a = min(x.min(), y.min())
    b = max(x.max(), y.max())

    lim = [-l, l]
    plt.plot(lim, lim[::-1], c="1.0", ls="--", alpha=0.3)
    plt.hexbin(x, y, norm=matplotlib.colors.LogNorm(), extent=lim + lim, cmap="viridis")


# %%
df = dq

# %% [markdown]
# Emulator doesn't conserve heat well:

# %%
mis_match = lambda dT, dQ: np.mean(np.abs((dT + dQ)))


def plot_conservation(dQ, dT):
    dT = dT * 86400
    dQ = dQ * 2.51e6 / 1004 * 86400

    plot_1_1(dT, dQ, 10)
    plt.title("E|Lv/cp dQ+dT| = {:.2f} (K/d)".format(mis_match(dQ, dT)))
    plt.xlabel(dT.name)
    plt.ylabel(dQ.name)


def plot_conservation_emulator(df):
    return plot_conservation(
        df.tendency_of_specific_humidity_due_to_gscond_emulator,
        df.tendency_of_air_temperature_due_to_gscond_emulator,
    )


# %%
plot_conservation(
    df.tendency_of_specific_humidity_due_to_gscond_emulator,
    df.tendency_of_air_temperature_due_to_gscond_emulator,
)

# %% [markdown]
# But the physics do:

# %%
plot_conservation(
    dT=df.tendency_of_air_temperature_due_to_gscond_physics,
    dQ=df.tendency_of_specific_humidity_due_to_gscond_physics,
)

# %% [markdown]
# # Accuracy

# %%
import numpy as np

# %%
plot_1_1(
    df.tendency_of_air_temperature_due_to_gscond_physics * 1004,
    df.tendency_of_air_temperature_due_to_gscond_emulator * 1004,
    0.01,
)
plt.title("condensation tendency temperature")
plt.xlabel("Physics (W/kg)")
plt.ylabel("Emulator (W/kg)")

# %%
plot_1_1(
    df.tendency_of_specific_humidity_due_to_gscond_physics * 2.51e6,
    df.tendency_of_specific_humidity_due_to_gscond_emulator * 2.51e6,
    0.01,
)
plt.title("condensation tendency humidity")
plt.xlabel("Physics (W/kg)")
plt.ylabel("Emulator (W/kg)")

# %% [markdown]
# # Reason

# %%
small = np.abs(df.tendency_of_air_temperature_due_to_gscond_emulator * 86400) < 2

# %%
(df.tendency_of_air_temperature_due_to_gscond_emulator[small] * 86400).mean()

# %%
(
    df.tendency_of_specific_humidity_due_to_gscond_emulator[small]
    * 86400
    * 2.51e6
    / 1004
).mean()

# %% [markdown]
# The non-conservation is most evident above 200 mb.

# %%


def plot(df):
    df = df.reset_index()
    plot_conservation_emulator(df[df.z < 200])


plot(df)


# %%
def plot(df):
    df = df.reset_index()
    plot_conservation_emulator(df[df.z > 200])


plot(df)


# %%
def match_vars(reg, da):
    variables = [v for v in da if re.match(reg, v)]
    return da[variables]


zonalvg = vcm.zonal_average_approximate(
    merged.lat, match_vars(r"tendency", merged).isel(time=0)
)

(86400 * 2.51e3 * zonalvg.tendency_of_specific_humidity_due_to_gscond_emulator).plot(
    vmax=3, y="z", yincrease=False
)

# %%
(86400 * zonalvg.tendency_of_air_temperature_due_to_gscond_emulator).plot(
    vmax=3, y="z", yincrease=False
)

# %%
(86400 * zonalvg.tendency_of_air_temperature_due_to_gscond_physics).plot(
    vmax=3, y="z", yincrease=False
)

# %% [markdown]
# Why is the temperature tendency so much less noisy in the upper atmosphere?

# %%
(86400 * zonalvg.tendency_of_air_temperature_due_to_gscond_emulator).plot(
    vmax=3, y="z", yincrease=False
)

# %%
total_tend = zonalvg.tendency_of_specific_humidity_due_to_zhao_carr_physics
precpd_tend = (
    zonalvg.tendency_of_specific_humidity_due_to_zhao_carr_physics
    - zonalvg.tendency_of_specific_humidity_due_to_gscond_emulator
)
(86400 * 2.51e3 * precpd_tend).plot(y="z", yincrease=False)

# %%
(86400 * 2.51e3 * total_tend).plot(y="z", yincrease=False, vmax=3)
