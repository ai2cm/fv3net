"""
In the last report, "bias_categories.py" I showed that offline bias of the
limited RNN cannot be improved by thresholding large relative decreases of cloud
to 0. Doing so increased the bias.

What about the non-limited RNNS? It turns out that these RNNs have approximately
10x smaller bias.  However, they produce negative clouds very quickly, which is
not good. Perhaps we can limit negative clouds online by using a post-hoc
classifier. Even though it is posthoc, the bias may be smaller than an RNN with
a limiter applied during training.

"""
# %%
import xarray as xr
import numpy as np
from fv3fit.emulation.data.load import nc_dir_to_tfdataset
from fv3fit.train_microphysics import TransformConfig
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.


    from https://stackoverflow.com/a/45696768
    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        "Return the locations of the ticks"
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError(
            "Cannot get tick locations for a " "%s type." % type(self)
        )


def tensordict_to_dataset(x):
    """convert a tensor dict into a xarray dataset and flip the vertical coordinate"""

    def _get_dims(val):
        n, feat = val.shape
        if feat == 1:
            return (["sample"], val[:, 0].numpy())
        else:
            return (["sample", "z"], val[:, ::-1].numpy())

    return xr.Dataset({key: _get_dims(val) for key, val in x.items()})


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
        "air_temperature_after_gscond",
        "air_temperature_after_precpd",
        "specific_humidity_after_precpd",
        "cloud_water_mixing_ratio_after_precpd",
        "total_precipitation",
        "ratio_of_snowfall_to_rainfall",
        "tendency_of_rain_water_mixing_ratio_due_to_microphysics",
        "time",
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


test_url = "/Users/noahb/data/vcm-ml-experiments/microphysics-emulation/2021-11-24/microphysics-training-data-v3-training_netcdfs/test"  # noqa
model_path = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-02/limit-tests-all-loss-rnn-7ef273/model.tf"  # noqa

model = tf.keras.models.load_model(model_path)
tfds = open_data(test_url)
truth_pred = tfds.map(lambda x: (x, model(x)))
truth_dict, pred_dict = next(iter(truth_pred.unbatch().batch(50_000)))
truth = tensordict_to_dataset(truth_dict)
pred = tensordict_to_dataset(pred_dict)
# %%
truth_tend = (
    truth.cloud_water_mixing_ratio_after_precpd - truth.cloud_water_mixing_ratio_input
)
pred_tend = (
    pred.cloud_water_mixing_ratio_after_precpd - truth.cloud_water_mixing_ratio_input
)

x = truth.cloud_water_mixing_ratio_input
x_0 = np.abs(x) <= 1e-20
y = pred.cloud_water_mixing_ratio_after_precpd
y_true = truth.cloud_water_mixing_ratio_after_precpd


# %%
# ROC of cloud after precpd = 0
y = np.ravel(pred.cloud_water_mixing_ratio_after_precpd)
y_true = np.ravel(truth.cloud_water_mixing_ratio_after_precpd)
class_true = y_true >= 1e-20

fpr, tpr, t = roc_curve(class_true, y)
fig, (ax, ax_tresh) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
(p,) = ax.plot(fpr, tpr, label="true positive rate")
ax.tick_params(axis="y", colors=p.get_color())
ax.yaxis.label.set_color(p.get_color())
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title(
    "ROC Analysis (cloud after precpd >= 0): AUC={:.2f}".format(
        roc_auc_score(class_true, y)
    )
)

ax.set_xscale("log")


fpr_regular = np.logspace(-7, 0, 100)
thresholds = np.interp(fpr_regular, fpr, t)
bias = [np.mean(np.where(y > thresh, y, 0) - y_true) / 900 for thresh in thresholds]

ax1 = ax.twinx()
(p2,) = ax1.plot(fpr_regular, bias, label="bias", color="red")
ax1.tick_params(axis="y", colors=p2.get_color())
ax1.yaxis.label.set_color(p2.get_color())
ax1.set_ylabel("bias (kg/kg/s)")
ax.grid()
ax1.set_yscale("symlog", linthreshy=1e-11)
ax1.set_xscale("log")
ax1.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=1e-11))

ax_tresh.loglog(fpr_regular[:-1], thresholds[:-1], "-")
ax_tresh.set_xlabel("false positive rate")
ax_tresh.set_ylabel("threshold (kg/kg)")
ax_tresh.grid()


# %% ROC cloud in = glcoud gscond out
y_true = -np.ravel(truth.specific_humidity_after_gscond - truth.specific_humidity_input)
y = -np.ravel(pred.specific_humidity_after_gscond - truth.specific_humidity_input)

fpr, tpr, t = roc_curve(np.abs(y_true) >= 1e-20, np.abs(y))
auc = roc_auc_score(np.abs(y_true) >= 1e-20, np.abs(y))
ax = plt.subplot()
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot(ax=ax)
ax.set_title("ROC (cloud after gscond = cloud in)")
