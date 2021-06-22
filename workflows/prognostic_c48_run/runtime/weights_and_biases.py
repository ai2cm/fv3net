import wandb
import matplotlib.pyplot as plt

from runtime.diagnostics.tensorboard import plot_to_image


class WandBLogger:
    def log_profiles(self, key, data, step):
        fig = plt.figure()
        plt.plot(data)
        wandb.log({key: wandb.Image(plot_to_image(fig))}, step=step)
        plt.close(fig)

    def log_dict(self, prefix, metrics, step):
        data = {}
        for key in metrics:
            name = prefix + "/" + key
            data[name] = metrics[key]
        data["epochs"] = step
        wandb.log(data, step=step)
