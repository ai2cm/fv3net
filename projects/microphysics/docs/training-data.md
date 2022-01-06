---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---
# Training Data

The training data is generated from several simulations with the FV3GFS atmospheric model

```{code-cell} ipython3
---
tags: [remove-input]
---
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
---
tags: [remove-input, remove-stderr]
---
# open data
config = TrainConfig.from_yaml_path(
    "gs://vcm-ml-experiments/microphysics-emulation/2022-01-04/"
    "log-cloud-dense-9b3e1a/config.yaml"
)

train_ds = nc_dir_to_tf_dataset(
    config.train_url, config.get_dataset_convertor(), nfiles=config.nfiles
)
train_set = next(iter(train_ds.batch(10000)))

# open model
model = tf.keras.models.load_model(config.out_url + "/model.tf")
```

The following conditional averaged plots show that a residual model works well for specific humidity and air temperature, but much less so for cloud water mixing ratio ($q_c$).
The cloud water mixing ratio likely depends strongly on the relative humidity tendency by the non-grid-scale condensation processes.

```{code-cell} ipython3
---
tags: [remove-input, remove-stderr, remove-stdout]
---
train_set = next(iter(train_ds.batch(1000)))
predictions = model(train_set)

fields = ZhaoCarrFields()
qc = fields.cloud_water

import numpy as np
def conditional(y, x, bins=100):
    """condtional pdf p(y|x)"""
    y = np.asarray(y).ravel()
    x = np.asarray(x).ravel()
    f, x, y = np.histogram2d(x, y, bins=bins)
    conditional_pdf = f / f.sum(0)
    plt.pcolormesh(x, y, conditional_pdf, cmap='bone_r')
    lim = [x.min(), x.max()]
    plt.plot(lim, lim, color='green', lw=2)


def conditional_field(train_set, qc: Field):
    conditional(train_set[qc.output_name], train_set[qc.input_name], bins=30)
    plt.xlabel(qc.input_name)
    plt.ylabel(qc.output_name)
    plt.title(f"{qc.output_name} | {qc.input_name}")

plt.figure()
conditional_field(train_set, fields.air_temperature)

plt.figure()
conditional_field(train_set, fields.cloud_water)

plt.figure()
conditional_field(train_set, fields.specific_humidity)
```
