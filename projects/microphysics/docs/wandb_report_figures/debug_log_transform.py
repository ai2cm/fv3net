# %%

from fv3fit.emulation.zhao_carr_fields import Field
from fv3fit.train_microphysics import nc_dir_to_tf_dataset, TrainConfig
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import tensorflow as tf
from toolz import memoize
import wandb
from fv3fit.tensorboard import plot_to_image

# %%
wandb.init(entity="ai2cm", project="microphysics-emulation", job_type="figures")

# %%
load_model = memoize(tf.keras.models.load_model)


wong_palette = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


plt.rcParams["axes.prop_cycle"] = cycler("color", wong_palette)
# %%
runs = {
    "log-cloud-dense-9b3e1a": "gs://vcm-ml-experiments/microphysics-emulation/2022-01-04/log-cloud-dense-9b3e1a/model.tf",  # noqa
    "rnn-v1-optimize-batch-512-1c5100": "gs://vcm-ml-experiments/microphysics-emulation/2021-12-18/rnn-v1-optimize-batch-512-1c5100/model.tf",  # noqa
    "log-cloud-gscond-inputs-rnn-2644d3": "gs://vcm-ml-experiments/microphysics-emulation/2022-01-06/log-cloud-gscond-inputs-rnn-2644d3/model.tf",  # noqa
}

# %%
# open data
config = TrainConfig.from_yaml_path(
    "gs://vcm-ml-experiments/microphysics-emulation/2022-01-04/"
    "log-cloud-dense-9b3e1a/config.yaml"
)

required_data = [
    "specific_humidity_input",
    "specific_humidity_after_precpd",
    "cloud_water_mixing_ratio_input",
    "cloud_water_mixing_ratio_after_precpd",
    "pressure_thickness_of_atmospheric_layer",
    "air_temperature_after_precpd",
    "air_temperature_input",
    "total_precipitation",
    "air_temperature_after_last_gscond",
    "air_temperature_input",
    "cloud_water_mixing_ratio_input",
    "pressure_thickness_of_atmospheric_layer",
    "specific_humidity_after_last_gscond",
    "specific_humidity_input",
]

train_ds = nc_dir_to_tf_dataset(
    config.train_url, config.transform.get_pipeline(required_data), nfiles=config.nfiles
)
train_set = next(iter(train_ds.batch(10000)))
train_set = config.get_transform().forward(train_set)

# %%
field = "cloud_water_mixing_ratio_after_precpd"
bins = 10.0 ** np.arange(-40, 10, 1)
truth = train_set[field].numpy()
plt.hist(truth.ravel(), bins, histtype="step", label="truth")

for display_name, model_url in runs.items():
    model = load_model(model_url)
    predictions = model(train_set)
    pred = predictions[field].numpy()
    plt.hist(pred.ravel(), bins, histtype="step", label=display_name)
plt.ylabel("count")
plt.xlabel(r"$q_c$ output (kg/kg)")
plt.xscale("log")
plt.yscale("log")
plt.ylim(top=10e8)
plt.title("Histogram of cloud output")
plt.legend()
wandb.log({"qc-histogram": wandb.Image(plot_to_image(plt.gcf()))})

# %%
plt.figure()
i = 0
field = "cloud_water_mixing_ratio_after_precpd"
plt.semilogy(truth[i], label="truth")
for display_name, model_url in runs.items():
    model = load_model(model_url)
    predictions = model(train_set)
    pred = predictions[field].numpy()
    truth = train_set[field].numpy()
    plt.semilogy(pred[i], label=display_name)
plt.xlabel("vertical index")
plt.ylabel(r"$q_c$ output (kg/kg)")
plt.legend()
wandb.log({"qc-profile": wandb.Image(plot_to_image(plt.gcf()))})

# %% [markdown]
# The skill is very poor for cloud water mixing ratio despite being good for log
# cloud
# %% skill
r2 = {}
for field in config._model.fields + [
    Field("cloud_water_mixing_ratio_after_precpd", "cloud_water_mixing_ratio_input")
]:
    if field.output_name and field.input_name:

        pred = predictions[field.output_name].numpy()
        truth = train_set[field.output_name].numpy()
        input_ = train_set[field.input_name].numpy()

        sse = ((pred - truth) ** 2).mean()
        ss = ((truth - input_) ** 2).mean()
        r2[field.output_name] = 1 - sse / ss

print("Skill versus null tendency:")
print()
for key in r2:
    print(f"{key}: {r2[key]:0.02f}")


# %%
wandb.finish()

# %%
