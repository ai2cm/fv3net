import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from fv3fit.tensorboard import plot_to_image


class TBLogger:
    def log_profiles(self, key, data, step):
        fig = plt.figure()
        plt.plot(data)
        tf.summary.image(key, plot_to_image(fig), step)

    def log_dict(self, prefix, metrics, step):
        for key in metrics:
            name = prefix + "/" + key
            tf.summary.scalar(name, metrics[key], step=step)


class ConsoleLogger:
    def log_profiles(self, key, data, step):
        pass

    def log_dict(self, prefix, metrics, step):
        print(f"step: {step}")
        for key in metrics:
            name = prefix + "/" + key
            print(f"{name}:", metrics[key])


class LoggerList:
    def __init__(self, loggers):
        self.loggers = loggers

    def log_profiles(self, key, data, step):
        for logger in self.loggers:
            logger.log_profiles(key, data, step)

    def log_dict(self, prefix, metrics, step):
        for logger in self.loggers:
            logger.log_dict(prefix, metrics, step)


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
