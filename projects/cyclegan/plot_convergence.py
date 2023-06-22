import wandb
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    CHECKPOINT_PATH = "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/"
    EVALUATE_ON_TRAIN = False
    api = wandb.Api()
    run = api.run("ai2cm/cyclegan_c48_to_c384/3w471q4o")
    history = run.history()
    epoch = np.arange(1, len(history) + 1)
    fig, ax = plt.subplots(4, 1, figsize=(6, 10))
    ax[0].plot(epoch, history.train_loss.to_numpy(), label="training")
    ax[0].plot(epoch, history.val_train_loss.to_numpy(), label="validation")
    ax[0].set_ylabel("loss")
    ax[0].set_title("loss")
    ax[0].legend()
    ax[1].plot(epoch, history.fake_b_vs_real_b_bias_std_0.to_numpy(), label="training")
    ax[1].plot(
        epoch, history.val_fake_b_vs_real_b_bias_std_0.to_numpy(), label="validation"
    )
    ax[1].set_ylabel("Pattern bias\nstandard deviation (mm/day)")
    ax[1].set_title("pattern bias")
    ax[1].legend()
    ax[2].plot(epoch, history.fake_b_vs_real_b_bias_mean_0.to_numpy(), label="training")
    ax[2].plot(
        epoch, history.val_fake_b_vs_real_b_bias_mean_0.to_numpy(), label="validation"
    )
    ax[2].set_ylabel("Mean bias (mm/day)")
    ax[2].set_title("mean bias")
    ax[2].legend()
    ax[3].plot(
        epoch,
        history.get("val_percentile_0.990000_gen_error_0").to_numpy(),
        label="99%",
    )
    ax[3].plot(
        epoch,
        history.get("val_percentile_0.999000_gen_error_0").to_numpy(),
        label="99.9%",
    )
    ax[3].plot(
        epoch,
        history.get("val_percentile_0.999900_gen_error_0").to_numpy(),
        label="99.99%",
    )
    ax[3].set_ylabel("Error (mm/day)")
    ax[3].set_xlabel("Epoch")
    ax[3].set_title("validation percentile error")
    ax[3].legend()
    for i in range(4):
        ax[i].set_xlim(0, len(history) + 1)
    plt.tight_layout()
    plt.savefig("cyclegan_convergence.png", dpi=300)
