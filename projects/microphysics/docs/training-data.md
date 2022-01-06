---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst,ipynb
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: fv3net
  language: python
  name: fv3net
---


# Training Data

The training data is generated from several simulations with the FV3GFS atmospheric model

```{code-cell} ipython3
from fv3fit.emulation.zhao_carr_fields import Field, ZhaoCarrFields
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
```

```{code-cell} ipython3
# open data
config = TrainConfig.from_yaml_path(
    "gs://vcm-ml-experiments/microphysics-emulation/2022-01-04/"
    "log-cloud-dense-9b3e1a/config.yaml"
)

train_ds = nc_dir_to_tf_dataset(
    config.train_url, config.get_dataset_convertor(), nfiles=config.nfiles
)
train_set = next(iter(train_ds.batch(40000)))
train_set = config.get_transform().forward(train_set)

# open model
model = tf.keras.models.load_model(config.out_url + "/model.tf")
```

The following conditional averaged plots show that a residual model works well for specific humidity and air temperature, but much less so for cloud water mixing ratio ($q_c$).
The cloud water mixing ratio likely depends strongly on the relative humidity tendency by the non-grid-scale condensation processes.

```{code-cell} ipython3
predictions = model(train_set)

fields = ZhaoCarrFields()
qc = fields.cloud_water

import numpy as np
def conditional(y, x, bins=100):
    """condtional pdf p(y|x)"""
    y = np.asarray(y).ravel()
    x = np.asarray(x).ravel()
    f, xe, ye = np.histogram2d(x, y, bins=bins)
    conditional_pdf = f / f.sum(0)
    
    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1,  height_ratios=(7, 2),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[1, 0], sharex=ax)
    
    ax.pcolormesh(xe, ye, conditional_pdf, cmap='bone_r')
    lim = [x.min(), x.max()]
    ax.plot(lim, lim, color='green', lw=2)
    
    ax_hist.hist(x, bins=bins)
    return ax, ax_hist


def conditional_field(train_set, qc: Field):
    ax, ax_hist = conditional(train_set[qc.output_name], train_set[qc.input_name], bins=30)
    ax_hist.set_xlabel(qc.input_name)
    ax.set_ylabel(qc.output_name)
    ax.set_title(f"{qc.output_name} | {qc.input_name}")

plt.figure()
conditional_field(train_set, fields.air_temperature)

plt.figure()
conditional_field(train_set, fields.cloud_water)

plt.figure()
conditional_field(train_set, fields.specific_humidity)

log_qc = Field("log_cloud_output", "log_cloud_input")
plt.figure()
conditional_field(train_set, log_qc)
```
