# %%

from fv3fit.emulation.zhao_carr_fields import Field
from fv3fit.train_microphysics import nc_dir_to_tf_dataset, TrainConfig
import matplotlib.pyplot as plt
from cycler import cycler
import tensorflow as tf

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
# open data
config = TrainConfig.from_yaml_path(
    "gs://vcm-ml-experiments/microphysics-emulation/2022-01-04/"
    "log-cloud-dense-9b3e1a/config.yaml"
)

train_ds = nc_dir_to_tf_dataset(
    config.train_url, config.get_dataset_convertor(), nfiles=config.nfiles
)
train_set = next(iter(train_ds.batch(10000)))

# %%
train_set = config.get_transform().forward(train_set)

# open model
model = tf.keras.models.load_model(config.out_url + "/model.tf")
# %%
predictions = model(train_set)

# %%
log_cloud = "log_cloud_output"
bins = list(range(-80, 10, 5))
pred = predictions[log_cloud].numpy()
truth = train_set[log_cloud].numpy()
plt.hist(pred.ravel(), bins, histtype="step", label="prediction")
plt.hist(truth.ravel(), bins, histtype="step", label="truth")
plt.ylabel("count")
plt.xlabel(r"$q_c$ output (kg/kg)")
plt.annotate(
    "large values\nin prediction",
    (1, 10000),
    xytext=(-20, 150000),
    arrowprops=dict(facecolor="black"),
)
plt.legend()

# %%
i = 0
field = "cloud_water_mixing_ratio_after_precpd"
bins = None

pred = predictions[field].numpy()
truth = train_set[field].numpy()
plt.semilogy(pred[i], label="prediction")
plt.semilogy(truth[i], label="truth")
plt.xlabel("vertical index")
plt.ylabel(r"$q_c$ output (kg/kg)")
plt.annotate("Large predicted value", (40, 10e10))
plt.legend()

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
