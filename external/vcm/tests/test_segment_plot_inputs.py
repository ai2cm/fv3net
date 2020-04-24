import pytest
import vcm
import numpy as np


@pytest.fixture(params=[(2, 2), (8, 10)])
def shape(request):
    return request.param


@pytest.fixture(params=["no_nans", "quarter_nans"])
def nan_fraction(request):
    if request.param == "no_nans":
        return 0
    elif request.param == "quarter_nans":
        return 0.25
    else:
        raise NotImplementedError()


@pytest.fixture
def array(shape, nan_fraction):
    array = np.zeros(shape)
    array_1d = array.ravel()
    n_values = array_1d.shape[0]
    array_1d[:] = np.arange(n_values)
    array_1d[np.random.choice(np.arange(n_values), size=min(n_values - 1, int(nan_fraction * n_values)))] = np.nan
    return array


@pytest.fixture
def x(shape):
    array = np.empty([shape[0] + 1, shape[1] + 1])
    array[:, :] = np.arange(shape[1] + 1)[None, :]
    return array


@pytest.fixture
def y(shape):
    array = np.empty([shape[0] + 1, shape[1] + 1])
    array[:, :] = np.arange(shape[0] + 1)[:, None]
    return array


def test_segment_plot_inputs_removes_nans(x, y, array):
    total_size = 0
    total_sum = 0
    for x, y, data in vcm.visualize._plot_cube._segment_plot_inputs(x, y, array):
        assert np.sum(np.isnan(data)) == 0
        assert np.product(data.shape) > 0
        total_size += np.product(data.shape)
        total_sum += np.sum(data)
    assert total_size == np.sum(~np.isnan(array))
    assert total_sum == np.nansum(array)


def test_segment_plot_inputs_returns_unique_values(x, y, array):
    values_seen = set()
    for x, y, data in vcm.visualize._plot_cube._segment_plot_inputs(x, y, array):
        for value in data.ravel():
            assert value not in values_seen
            values_seen.add(value)
