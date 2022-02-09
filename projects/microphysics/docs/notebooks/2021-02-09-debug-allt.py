# flake8: noqa
# %%
import sys

sys.path.insert(0, "..")
import cartopy.crs
import config
import fsspec
import fv3viz
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import vcm
import vcm.catalog
import vcm.fv3
import xarray

# url = "gs://vcm-ml-experiments/microphysics-emulation/2022-02-08/rnn-alltdep-47ad5b-login-limited-10d-online"
url = "gs://vcm-ml-experiments/microphysics-emulation/2022-02-09/rnn-alltdep-47ad5b-login-limited-10d-gsout-online"


def open_url(url):
    fs = fsspec.filesystem("gs", use_listings_cache=False)
    mapper = fs.get_mapper(url + "/atmos_dt_atmos.zarr")
    ds = xarray.open_zarr(mapper, consolidated=True)
    piggy = xarray.open_zarr(fsspec.get_mapper(url + "/piggy.zarr"), consolidated=True)
    grid = vcm.catalog.catalog["grid/c48"].to_dask()

    ds = vcm.fv3.gfdl_to_standard(
        xarray.merge([ds, piggy], compat="override", join="inner")
    ).assign(**grid)
    return ds, grid


def gscond_tendency(data, field, source):
    if field == "cloud_water" and source == "emulator":
        return -data[f"tendency_of_specific_humidity_due_to_gscond_{source}"]
    else:
        return data[f"tendency_of_{field}_due_to_gscond_{source}"]


def total_tendency(data, field, source):
    return data[f"tendency_of_{field}_due_to_zhao_carr_{source}"]


def precpd_tendency(data, field, source):
    return total_tendency(data, field, source) - gscond_tendency(data, field, source)


def assoc_precpd_tendencies(ds):

    ds_with_gscond = ds.copy()

    for source in ["emulator", "physics"]:
        for field in ["specific_humidity", "air_temperature", "cloud_water"]:
            ds_with_gscond[
                f"tendency_of_{field}_due_to_precpd_{source}"
            ] = precpd_tendency(ds, field, source)

    ds_with_gscond["gscond"] = ds_with_gscond[
        "tendency_of_air_temperature_due_to_gscond_physics"
    ]

    ds_with_gscond["precpd"] = (
        ds_with_gscond["tendency_of_air_temperature_due_to_zhao_carr_physics"]
        - ds_with_gscond["tendency_of_air_temperature_due_to_gscond_physics"]
    )

    ds_with_gscond["t_precpd_emu"] = (
        ds_with_gscond["tendency_of_air_temperature_due_to_zhao_carr_emulator"]
        - ds_with_gscond["tendency_of_air_temperature_due_to_gscond_emulator"]
    )

    return ds_with_gscond


def combine_dict(heatings, labels):
    heating = vcm.combine_array_sequence(
        [("heating", label, val) for label, val in heatings.items()], labels
    )
    return heating.heating


def compute_heat_budget(ds, compute_tendency):
    """Compute the heat budget for a given process
    """

    T = "air_temperature"
    q = "specific_humidity"
    qc = "cloud_water"

    emulator = "emulator"
    physics = "physics"

    heatings = {}

    for source in [physics, emulator]:

        heatings[("dT", source)] = compute_tendency(ds, T, source)
        heatings[("-L/cp dqv", source)] = (
            -compute_tendency(ds, q, source) * 2.51e6 / 1004
        )
        heatings[("L/cp dqc", source)] = (
            compute_tendency(ds, qc, source) * 2.51e6 / 1004
        )

    return combine_dict(heatings, ["name", "source"])


def plot_heatings(grid, heatings):
    heatings.attrs["long_name"] = ""

    crs = cartopy.crs.Orthographic(central_latitude=-90)
    fig = fv3viz.plot_cube(
        grid.assign(plotme=heatings),
        "plotme",
        col="name",
        row="source",
        norm=matplotlib.colors.SymLogNorm(1e-6),
        projection=crs,
    )[0]
    fig.suptitle(
        "Ref Pressure: {} (mb)\nLast time saved before crash {}\nRun: {}".format(
            plotme.z.item(), plotme.time.item(), url
        )
    )
    plt.subplots_adjust(top=0.8)


ds, grid = open_url(url)
# %%
plotme = ds.isel(time=-1).sel(z=823, method="nearest")
phys = plotme["tendency_of_air_temperature_due_to_zhao_carr_physics"]
pred = plotme["tendency_of_air_temperature_due_to_zhao_carr_emulator"]

bins = np.logspace(-30, 0, 100)

plt.hist(np.abs(np.ravel(phys)), bins=bins, histtype="step", label="physics")
plt.hist(np.abs(np.ravel(pred)), bins=bins, histtype="step", label="emulator")
plt.xscale("log")
fraction_zero = float((np.abs(phys) < bins[0]).mean())
plt.title(fraction_zero)
plt.legend()
# %%
plotme = ds.isel(time=-1).sel(z=512, method="nearest")
t_emu = "tendency_of_air_temperature_due_to_zhao_carr_emulator"
q_emu = "tendency_of_specific_humidity_due_to_zhao_carr_emulator"
qc_emu = "tendency_of_cloud_water_due_to_zhao_carr_emulator"
qc_true = "tendency_of_cloud_water_due_to_zhao_carr_physics"
t_true = "tendency_of_air_temperature_due_to_zhao_carr_physics"
q_true = "tendency_of_specific_humidity_due_to_zhao_carr_physics"
pred = plotme[t_emu]
truth = plotme[t_true]


heatings = {}

heatings[("dT", "physics")] = plotme[t_true]
heatings[("-L/cp dqv", "physics")] = -plotme[q_true] * 2.51e6 / 1004
heatings[("L/cp dqc", "physics")] = plotme[qc_true] * 2.51e6 / 1004

heatings[("dT", "emulation")] = plotme[t_emu]
heatings[("-L/cp dqv", "emulation")] = -plotme[q_emu] * 2.51e6 / 1004
heatings[("L/cp dqc", "emulation")] = plotme[qc_emu] * 2.51e6 / 1004


plot_heatings(grid, combine_dict(heatings, ["name", "source"]))

# %%
def compute_dqt(ds):
    ds = ds.copy()
    ds["dq_error"] = ds[q_emu] - ds[q_true]
    ds["dqc_error"] = ds[qc_emu] - ds[qc_true]
    ds["dqt_error"] = ds["dq_error"] + ds["dqc_error"]
    ds["dqt"] = ds[q_true] + ds[qc_true]
    return ds


# %%
ds_with_error = compute_dqt(ds)
dims = ["x", "y", "tile"]
mss = vcm.weighted_average(ds_with_error.dqt ** 2, grid.area, dims)
mse = vcm.weighted_average(ds_with_error.dqt_error ** 2, grid.area, dims)
skill = 1 - mse / mss
skill.plot(y="z", vmin=-1, vmax=1, yincrease=False)

# %%
ds_with_gscond = assoc_precpd_tendencies(ds)


# %%
plotme = ds_with_gscond.sel(z=400, method="nearest").isel(time=-1)
crs = cartopy.crs.Orthographic(central_latitude=-90)
fv3viz.plot_cube(
    plotme, "precpd", projection=crs,
)

fv3viz.plot_cube(
    plotme, "gscond", projection=crs,
)

fv3viz.plot_cube(
    plotme, "t_precpd_emu", projection=crs,
)
fv3viz.plot_cube(
    plotme, "tendency_of_air_temperature_due_to_gscond_emulator", projection=crs,
)

fv3viz.plot_cube(
    plotme, "tendency_of_specific_humidity_due_to_gscond_emulator", projection=crs,
)
# %%


def global_skill(truth, pred, grid):
    dims = ["x", "y", "tile"]
    mss = vcm.weighted_average(truth ** 2, grid.area, dims)
    mse = vcm.weighted_average((pred - truth) ** 2, grid.area, dims)
    return 1 - mse / mss


def global_skill_plot(truth, pred, grid):
    skill = global_skill(truth, pred, grid)
    skill.plot(y="z", vmin=-1, yincrease=False)


global_skill_plot(ds_with_gscond.precpd, ds_with_gscond.t_precpd_emu, grid)
# %%
global_skill_plot(
    ds_with_gscond["tendency_of_air_temperature_due_to_gscond_physics"],
    ds_with_gscond["tendency_of_air_temperature_due_to_gscond_emulator"],
    grid,
)

# %%
gscond_skill = global_skill(
    ds_with_gscond["tendency_of_air_temperature_due_to_gscond_physics"],
    ds_with_gscond["tendency_of_air_temperature_due_to_gscond_emulator"],
    grid,
)
# %%
def plot_cube(arr, figsize=(10, 6), **kw):
    fig = fv3viz.plot_cube(grid.assign(plotme=arr), "plotme", **kw)[0]
    fig.set_size_inches(*figsize)


# %%
neg_skill = ds_with_gscond.assign(
    gscond_err=ds_with_gscond["tendency_of_air_temperature_due_to_gscond_emulator"]
    - ds_with_gscond["tendency_of_air_temperature_due_to_gscond_physics"],
    precpd_err=ds_with_gscond["precpd"] - ds_with_gscond["t_precpd_emu"],
)


plotme = neg_skill.isel(time=-1).interp(z=500)


fig = fv3viz.plot_cube(
    plotme,
    "gscond_err",
    # projection=crs,
)[0]

fig.set_size_inches(10, 6)

fig = fv3viz.plot_cube(
    plotme,
    "tendency_of_air_temperature_due_to_gscond_physics",
    # projection=crs,
)[0]

fig.set_size_inches(10, 6)

fig = fv3viz.plot_cube(
    plotme,
    "tendency_of_air_temperature_due_to_gscond_emulator",
    # projection=crs,
)[0]

fig.set_size_inches(10, 6)
# %%
plotme = neg_skill[
    [
        "gscond_err",
        "tendency_of_air_temperature_due_to_gscond_physics",
        "precpd_err",
        "precpd",
    ]
]
plt.figure(figsize=(10, 4))
bins = np.logspace(-12, -3, 100)
for key in plotme:
    plt.hist(np.ravel(np.abs(plotme[key])), bins, histtype="step", label=key)
plt.legend(loc="upper left")
plt.xscale("log")


# %%
def plot_heatings(grid, heatings, label):
    heatings.attrs["long_name"] = ""

    crs = cartopy.crs.Orthographic(central_latitude=-90)
    fig, axes, hs, cbar, facet_grid = fv3viz.plot_cube(
        grid.assign(plotme=heatings),
        "plotme",
        col="name",
        row="source",
        norm=matplotlib.colors.SymLogNorm(1e-6, base=10),
        colorbar=False,
        projection=crs,
    )

    fig.suptitle(
        "{}\nRef Pressure: {} (mb)\nLast time saved before crash {}\nRun: {}".format(
            label, heatings.z.item(), heatings.time.item(), url
        )
    )
    fig.set_size_inches(12, 12 * 0.6)
    plt.subplots_adjust(top=0.8)
    plt.colorbar(hs[0], ax=axes.tolist())


heating = compute_heat_budget(ds, gscond_tendency)
plot_heatings(
    grid, heating.isel(time=-1).interp(z=512), "Condensation Tendency (gscond)"
)

# %%
heating = compute_heat_budget(ds, precpd_tendency)
plot_heatings(
    grid, heating.isel(time=-1).interp(z=512), "Precipitation Tendency (precpd)"
)
