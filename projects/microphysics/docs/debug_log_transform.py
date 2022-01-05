# %%

from fv3fit.train_microphysics import nc_dir_to_tf_dataset, TrainConfig
import matplotlib.pyplot as plt
import tensorflow as tf

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
