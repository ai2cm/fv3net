import numpy as np
import pytest

import fv3viz
from fv3viz._plot_helpers import _get_var_label


@pytest.mark.parametrize(
    "attrs,var_name,expected_label",
    [
        ({}, "temp", "temp"),
        ({"long_name": "air_temperature"}, "temp", "air_temperature"),
        ({"units": "degK"}, "temp", "temp [degK]"),
        (
            {"long_name": "air_temperature", "units": "degK"},
            "temp",
            "air_temperature [degK]",
        ),
    ],
)
def test__get_var_label(attrs, var_name, expected_label):
    assert _get_var_label(attrs, var_name) == expected_label


@pytest.mark.parametrize(
    "data, args, expected_result",
    [
        (np.array([0.0, 0.5, 1.0]), {}, (0.0, 1.0, "viridis")),
        (np.array([-0.5, 0, 1]), {}, (-1.0, 1.0, "RdBu_r")),
        (np.array([0.0, 0.5, 1.0]), {"robust": True}, (0.02, 0.98, "viridis")),
        (np.array([0.0, 0.5, 1.0]), {"cmap": "jet"}, (0.0, 1.0, "jet")),
        (np.array([0.0, 0.5, 1.0]), {"vmin": 0.2}, (0.2, 1.0, "viridis")),
        (np.array([-0.5, 0, 1]), {"vmin": -0.6}, (-0.6, 0.6, "RdBu_r")),
    ],
)
def test_auto_limits_cmap(data, args, expected_result):
    result = fv3viz.auto_limits_cmap(data, **args)
    assert result == expected_result
