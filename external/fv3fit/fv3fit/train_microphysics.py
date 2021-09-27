import dataclasses
from typing import List
import wandb
import tensorflow as tf

from .emulation.microphysics import Config
from .emulation.data import TransformConfig
from .emulation.layers import MeanFeatureStdNormLayer
from fv3fit import set_random_seed
from fv3fit._shared.config import OptimizerConfig
from fv3fit.emulation.data import get_nc_files, nc_files_to_tf_dataset
from loaders.batches import shuffle

# TODO: need to assert that outputs from data are same
#       order as all outputs from the model

@dataclasses.dataclass
class TrainConfig:
    train_url: str
    test_url: str
    transform: TransformConfig
    model: Config
    use_wandb: bool = True
    epochs: int = 1
    batch_size: int = 64
    optimizer: OptimizerConfig = dataclasses.field(default_factory=lambda: OptimizerConfig("Adam"))
    loss_variables: List[str] = dataclasses.field(default_factory=list)
    metric_variables: List[str] = dataclasses.field(default_factory=list)


# TODO: update netcdf-> ds function
def _netcdf_url_to_dataset(url, transform, nfiles=None, do_shuffle=True):

    files = get_nc_files(url)
    if do_shuffle:
        files = shuffle(files)
    if nfiles is not None:
        files = files[:nfiles]

    return nc_files_to_tf_dataset(files, transform)


def get_out_samples(model_config: Config, samples, sample_names):

    direct_sample = []
    residual_sample = []
    for name, sample in zip(sample_names, samples):

        if name in model_config.direct_out_variables:
            direct_sample.append(sample)
        elif name in model_config.direct_out_variables:
            residual_sample.append(sample)
        else:
            print(f"Unrecognized: {name}")

    return direct_sample, residual_sample


class NormalizedMSE(tf.keras.losses.MeanSquaredError):
    
    def __init__(self, sample_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._normalize = MeanFeatureStdNormLayer()
        self._normalize.fit(sample_data)
        
    def call(self, y_true, y_pred):
        return super().call(self._normalize(y_true), self._normalize(y_pred))


def main(config: TrainConfig):
    set_random_seed(0)

    callbacks = []
    if config.use_wandb:
        job = wandb.init(
            entity="ai2cm",
            project="microphysics-emulation",
            job_type="training",
        )

        # saves best model by validation every epoch
        callbacks.append(wandb.keras.WandbCallback())


    train_ds = _netcdf_url_to_dataset(config.train_url, config.transform)
    test_ds = _netcdf_url_to_dataset(config.test_url, config.transform)

    sample_in, sample_out = next(iter(train_ds.shuffle(100_000).batch(50_000)))
    direct_sample, resid_sample = get_out_samples(config.model, sample_out, config.transform.output_variables)
    model = config.model.build(sample_in, sample_direct_out=direct_sample, sample_residual_out=resid_sample)

    losses = {}
    metrics = {}
    for out_varname, sample in zip(config.transform.output_variables, sample_out):
        loss_func = NormalizedMSE(sample)
        if out_varname in config.loss_variables:
            losses[out_varname] = loss_func
        elif out_varname in config.metric_variables:
            metrics[out_varname] = loss_func
    
    optimizer = config.optimizer
    model.compile(loss=losses, metrics=metrics, optimizer=optimizer)

    history = model.fit(
        train_ds.shuffle(100_000).batch(config.batch_size),
        epochs=config.epochs,
        valid_freq=5,
        valid_data=test_ds.batch(config.batch_size),
        verbose=2,
        callbacks=callbacks
    )

    # scoring
    # un-normed MSE (avg, in adjusted units)
    # percent improve over null, use artifact
    # RMSE by level
    # bias by level

    # save to GCS as backup (remove losses/optimizer)








