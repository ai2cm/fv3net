# %%
from fv3fit.emulation.data.load import nc_dir_to_tfdataset
from fv3fit.train_microphysics import TransformConfig
import fv3fit.emulation.transforms.zhao_carr as zhao_carr
import tensorflow as tf


def open_data(url: str) -> tf.data.Dataset:
    variables = [
        "latitude",
        "longitude",
        "pressure_thickness_of_atmospheric_layer",
        "air_pressure",
        "surface_air_pressure",
        "air_temperature_input",
        "specific_humidity_input",
        "cloud_water_mixing_ratio_input",
        "air_temperature_after_last_gscond",
        "specific_humidity_after_last_gscond",
        "surface_air_pressure_after_last_gscond",
        "specific_humidity_after_gscond",
        "cloud_water_mixing_ratio_after_gscond",
        # "air_temperature_after_gscond",
        # "air_temperature_after_precpd",
        # "specific_humidity_after_precpd",
        # "cloud_water_mixing_ratio_after_precpd",
        "total_precipitation",
        "ratio_of_snowfall_to_rainfall",
        "tendency_of_rain_water_mixing_ratio_due_to_microphysics",
        "time",
        zhao_carr.CLOUD_PRECPD,
    ]
    data_transform = TransformConfig(
        antarctic_only=False,
        use_tensors=True,
        vertical_subselections=None,
        derived_microphys_timestep=900,
    )
    # TODO change default of get_pipeline to get all the variables
    # will allow deleting the code above
    return nc_dir_to_tfdataset(url, data_transform.get_pipeline(variables))


url = "gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/microphysics-training-data-v4/test"
tfds = open_data(url).unbatch().batch(10_000)
# %%
for batch in tfds.as_numpy_iterator():
    break


# %%
x = batch[zhao_carr.CLOUD_INPUT]
y = batch[zhao_carr.CLOUD_GSCOND]
z = batch[zhao_carr.CLOUD_PRECPD]

# %%
import numpy as np
import matplotlib.pyplot as plt

xbins = np.logspace(-12, 0, 100)
bins = [xbins, xbins]
count, xbin, ybin = np.histogram2d(x.ravel(), z.ravel(), bins)
plt.imshow(count)


# %%
plt.plot(count[60])
plt.xlim([60 - 5, 60 + 5])


# %%
xbin = np.logspace(-7, np.log10(np.quantile(x, 0.995)), 32)
out = (z - x) / x
mask = (z != x) & (z > 1e-12)
# m = np.quantile(out, 0.995)
# ybin = np.linspace(-m, m, 32)
q = np.linspace(0, 1, 34)
ybin = np.quantile(np.unique(out), q)
bins = [xbin, ybin]
count, xbin, ybin = np.histogram2d(x[mask], out[mask], bins)
plt.pcolor(xbin, q, count.T / count.sum(1))
plt.xscale("log")
# %%
