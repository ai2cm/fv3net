import fv3fit
import pytest
import numpy as np


@pytest.mark.parametrize("scaling", ["standard", "standard_uniform"])
def test_mae_loss_increases_linearly(scaling):
    config = fv3fit.LossConfig(loss_type="mae", scaling=scaling)
    y = np.zeros([5])
    std = np.ones([5])
    loss = config.loss(std)
    assert loss(y, y + 2.0) == 2 * loss(y, y + 1)


@pytest.mark.parametrize("scaling", ["standard", "standard_uniform"])
def test_mae_loss_increases_quadratically(scaling):
    config = fv3fit.LossConfig(loss_type="mse", scaling=scaling)
    y = np.zeros([5])
    std = np.ones([5])
    loss = config.loss(std)
    assert loss(y, y + 2.0) == 4 * loss(y, y + 1)


@pytest.mark.parametrize("scaling", ["standard", "standard_uniform"])
@pytest.mark.parametrize("loss_type", ["mse", "mae"])
@pytest.mark.parametrize("weight_2", [2.0, 3.2])
def test_loss_increases_linearly_with_weight(loss_type, scaling, weight_2):
    config_1 = fv3fit.LossConfig(loss_type=loss_type, scaling=scaling, weight=1.0)
    config_2 = fv3fit.LossConfig(loss_type=loss_type, scaling=scaling, weight=weight_2)
    y = np.zeros([5])
    std = np.ones([5])
    loss_1 = config_1.loss(std)
    loss_2 = config_2.loss(std)
    output_1 = loss_1(y, y + 2)
    output_2 = loss_2(y, y + 2)
    assert output_2 == weight_2 * output_1


@pytest.mark.parametrize("loss_type", ["mse", "mae"])
def test_standard_uniform_loss_uses_uniform_scaling(loss_type):
    config = fv3fit.LossConfig(loss_type=loss_type, scaling="standard_uniform")
    y1 = np.zeros([5])
    std = np.random.uniform(low=1.0, high=2.0, size=(5,))
    loss = config.loss(std)
    y2 = np.zeros([5])
    y2[0] = 1.0
    expected = loss(y1, y2)
    for i in range(y1.shape[0]):
        y2 = np.zeros([5])
        y2[i] = 1.0
        assert loss(y1, y2) == expected


@pytest.mark.parametrize("loss_type", ["mse", "mae"])
def test_standard_loss_uses_nonuniform_scaling(loss_type):
    config = fv3fit.LossConfig(loss_type=loss_type, scaling="standard")
    y1 = np.zeros([5])
    std = np.random.uniform(low=1.0, high=2.0, size=(5,))
    loss = config.loss(std)
    loss_list = []
    for i in range(y1.shape[0]):
        y2 = np.zeros([5])
        y2[i] = 1.0
        loss_list.append(loss(y1, y2).numpy())
    losses_are_different = len(set(loss_list)) == len(loss_list)
    assert losses_are_different
