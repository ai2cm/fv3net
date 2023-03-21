import numpy as np
from fv3fit.keras import TrainingLoopConfig
from fv3fit.tfdataset import tfdataset_from_batches
from fv3fit.reservoir.domain import concat_variables_along_feature_dim
from fv3fit.reservoir.autoencoder import (
    train_dense_autoencoder,
    DenseAutoencoderHyperparameters,
)
from tests.training.test_train import (
    get_dataset_default,
    get_uniform_sample_func,
)
import fv3fit


def test_train_autoencoder():
    n_sample = 10
    n_tile, nx, ny, nz = 1, 12, 12, 5
    sample_func = get_uniform_sample_func(
        size=(n_sample, n_tile, nx, ny, nz), low=0, high=1
    )
    input_variables, _, train_dataset = get_dataset_default(sample_func)
    _, _, test_dataset = get_dataset_default(sample_func)

    train_tfdataset = tfdataset_from_batches([train_dataset for _ in range(4)])
    val_tfdataset = tfdataset_from_batches([test_dataset])
    variables = ["var_in_3d", "var_in_2d"]
    sample_data = concat_variables_along_feature_dim(
        variables=variables, variable_tensors=next(iter(val_tfdataset))
    )

    hyperparameters = DenseAutoencoderHyperparameters(
        input_variables=input_variables,
        output_variables=input_variables,
        training_loop=TrainingLoopConfig(epochs=10),
    )
    model = train_dense_autoencoder(hyperparameters, train_tfdataset, val_tfdataset)
    tol = 0.01
    assert np.mean(np.sqrt((model.predict(sample_data) - sample_data) ** 2)) < tol


def test_dump_load_trained_autoencoder(tmpdir):
    n_sample = 10
    n_tile, nx, ny, nz = 1, 12, 12, 5
    sample_func = get_uniform_sample_func(size=(n_sample, n_tile, nx, ny, nz),)
    input_variables, _, train_dataset = get_dataset_default(sample_func)
    _, _, test_dataset = get_dataset_default(sample_func)

    train_tfdataset = tfdataset_from_batches([train_dataset for _ in range(4)])
    val_tfdataset = tfdataset_from_batches([test_dataset])
    variables = ["var_in_3d", "var_in_2d"]
    sample_data = concat_variables_along_feature_dim(
        variables=variables, variable_tensors=next(iter(val_tfdataset))
    )
    hyperparameters = DenseAutoencoderHyperparameters(
        input_variables=input_variables,
        output_variables=input_variables,
        training_loop=TrainingLoopConfig(epochs=2),
    )
    model = train_dense_autoencoder(hyperparameters, train_tfdataset, val_tfdataset)

    output_path = f"{str(tmpdir)}/model"
    model.dump(output_path)
    loaded_model = fv3fit.load(output_path)
    np.testing.assert_array_equal(
        loaded_model.predict(sample_data), model.predict(sample_data)
    )
