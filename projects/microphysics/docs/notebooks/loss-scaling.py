# %%

from fv3fit.train_microphysics import get_default_config, nc_dir_to_tf_dataset

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


config = get_default_config()

train_ds = (
    nc_dir_to_tf_dataset(config.train_url, config.transform, nfiles=config.nfiles)
    .shuffle(100_000)
    .take(100_000)
    .cache()
)
test_ds = nc_dir_to_tf_dataset(
    config.test_url, config.transform, nfiles=config.nfiles_valid
)

# %%
train_set = next(iter(train_ds.batch(10048)))

# %%
model = config.build(train_set)

# %%
forward = tf.function(model)

loss = config.loss
loss.build(train_set)

prediction = model(train_set)


def sum_jacobian(jac):
    return sum(tf.reduce_mean(x ** 2) for x in jac)


def g(x):
    return tf.math.pow(x, 1 / 3)


with tf.GradientTape() as tape1:
    tape1.watch(train_set)
    with tf.GradientTape(persistent=True) as tape:
        prediction = forward(train_set)
        loss = sum(
            tf.reduce_mean((g(prediction[key]) - g(train_set[key])) ** 2)
            for key in ["air_temperature_output", "specific_humidity_output"]
        )
    gradient_mag = sum_jacobian(tape.gradient(loss, model.trainable_weights))
    out = tape1.gradient(gradient_mag, train_set)

# %%

v = out["specific_humidity_output"].numpy().ravel()
v = out["air_temperature_output"].numpy().ravel()
plt.hist(v, 31)


# %%

loss = (
    train_set["specific_humidity_output"] - prediction["specific_humidity_output"]
) ** 2
df = pd.DataFrame(
    dict(
        temperature=train_set["air_temperature_input"].numpy().ravel(),
        loss_grad_mag=out["specific_humidity_output"].numpy().ravel(),
        loss_grad_mag_temp=out["air_temperature_output"].numpy().ravel(),
        humidity=train_set["specific_humidity_output"].numpy().ravel(),
        loss=loss.numpy().ravel(),
    )
)
means = df.groupby(pd.cut(df.temperature, 50)).mean()
sigs = df.groupby(pd.cut(df.temperature, 50)).std()
(means.loss_grad_mag.abs() * sigs.humidity).plot(logy=True, label="|dl/dq| * |dq|")
(means.loss_grad_mag_temp.abs() * sigs.temperature).plot(
    logy=True, label="|dl/dT| * |dT|"
)
sigs.loss.plot(label="loss")
plt.legend()
# %%

# %%
