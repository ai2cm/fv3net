# flake8: noqa
# %%
from fv3net.diagnostics.prognostic_run.emulation.single_run import open_rundir
import vcm
import fv3viz
import matplotlib.pyplot as plt
import xarray

# %%

url = "gs://vcm-ml-experiments/microphysics-emulation/2022-06-27/gscond-only-classifier-zcloud-online-10d-v2-online"
ds = open_rundir(url)

url = "gs://vcm-ml-experiments/microphysics-emulation/2022-06-27/gscond-only-classifier-zcloud-online-10d-v2-offline"
ds_offline = open_rundir(url)

# %%
left = ds_offline.assign_coords(mode="offline").expand_dims("mode")
right = ds.assign_coords(mode="online").expand_dims("mode")
diff = ds - ds_offline
merged = xarray.combine_by_coords(
    [left, right, diff.assign_coords(mode="diff").expand_dims("mode")]
)
# %%
def precpd_tend(ds):

    qv_tend = ds["tendency_of_specific_humidity_due_to_gscond_emulator"]
    qv_tend_tot = ds["tendency_of_specific_humidity_due_to_zhao_carr_emulator"]
    qv_tend_p = qv_tend_tot - qv_tend
    return qv_tend_p.assign_attrs(long_name="Qv Precpd tend", units="kg/kg/s")


def gscond_tend(ds):
    return ds["tendency_of_specific_humidity_due_to_gscond_emulator"].assign_attrs(
        units="kg/kg/s", long_name="Qv Gscond tend"
    )


def gscond_tend_physics(ds):
    return ds["tendency_of_specific_humidity_due_to_gscond_physics"].assign_attrs(
        units="kg/kg/s", long_name="Qv Gscond tend (physics)"
    )


def humidity(ds):
    return ds["specific_humidity"].assign_attrs(units="kg/kg")


# %%
end = merged.mean("time")
z = humidity(end)
z = z.assign_coords(lat=ds.lat, lon=ds.lon)
z = z.load()

# %%
interp = vcm.interpolate_unstructured(z, coords=vcm.select.meridional_ring())
interp = interp.swap_dims({"sample": "lat"})
interp.plot(yincrease=False, col="mode")
plt.subplots_adjust(top=0.8, right=0.80)
plt.suptitle("Meridional slice (lon=0)")
plt.savefig("lon0slice.png")

# %%
interp = vcm.interpolate_unstructured(z, coords=vcm.select.zonal_ring(lat=30))
interp = interp.swap_dims({"sample": "lon"})
interp.plot(yincrease=False, col="mode")
plt.subplots_adjust(top=0.8, right=0.80)
plt.suptitle("Zonal Slice (lat=30)")
plt.savefig("lat30slice.png")

# %%
interp = vcm.zonal_average_approximate(z.lat, z)
interp.plot(yincrease=False, col="mode")
plt.subplots_adjust(top=0.8, right=0.80)
plt.suptitle("Zonal average")
plt.savefig("zonal_average.png")
