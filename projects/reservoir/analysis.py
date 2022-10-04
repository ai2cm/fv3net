import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def predict_time_series(
    predictor, test_data, nburn, nsteps,
):
    leadup_inputs = test_data[:nburn]
    ic = test_data[nburn]
    input_dim = test_data[0].shape

    predictor.reset_reservoir_state()
    for input in leadup_inputs:
        predictor.increment_reservoir_state(input)

    time_series = [ic, predictor.predict(ic)]

    for n in range(nsteps):
        current_state = time_series[-1]
        time_series.append(predictor.predict(current_state.reshape(input_dim)))

    return np.vstack(time_series)


def plot_time_series_comparison(
    truth,
    prediction,
    burnin=0,
    nplot=None,
    vmax=5,
    figsize=None,
    plot_kwargs={"aspect": "auto"},
):
    fig = plt.figure(figsize=figsize or (8, 4))
    vmax = 5

    input_dim = prediction[0].shape

    ax0 = fig.add_subplot(131)
    ax0.imshow(truth[:nplot], vmin=-vmax, vmax=vmax, **plot_kwargs)
    ax0.set_title("KS solution")
    ax0.set_ylabel("t")
    ax0.set_xlabel("x")

    prediction_reshaped = np.array([p.reshape(input_dim) for p in prediction])
    ax1 = fig.add_subplot(132)
    ax1.imshow(prediction_reshaped[:nplot], vmin=-vmax, vmax=vmax, **plot_kwargs)
    ax1.set_title("RC prediction")
    ax1.set_ylabel("t")
    ax1.set_xlabel("x")

    ax2 = fig.add_subplot(133)
    im = ax2.imshow(
        truth[:nplot] - np.array([p.reshape(input_dim) for p in prediction[:nplot]],),
        vmin=-vmax,
        vmax=vmax,
        **plot_kwargs
    )

    # cbar_ax = fig.add_axes([1.075, 0.3, 0.025, 0.4])
    # fig.colorbar(im, cax=cbar_ax)

    ax2.set_title("difference")
    ax2.set_ylabel("t")
    ax2.set_xlabel("x")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    return (
        truth[:nplot],
        prediction_reshaped[:nplot],
        prediction_reshaped[:nplot] - truth[:nplot],
    )
