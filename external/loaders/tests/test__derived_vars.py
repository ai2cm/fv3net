import pytest
from typing import Sequence
from loaders.batches._derived_vars import (
    nonderived_variable_names,
    _wind_rotation_needed
)

@pytest.mark.parametrize(
    "variables, nonderived_variables",
(
    [["dQ1", "dQ2",], ["dQ1", "dQ2"]],
    [["dQ1", "dQ2", "dQu", "dQv", "cos_zenith_angle"], ["dQ1", "dQ2", "dQx", "dQy"]]
)
)
def test_nonderived_variable_names(variables, nonderived_variables):
    assert set(nonderived_variable_names(variables)) == set(nonderived_variables)


@pytest.mark.parametrize(
    "available_data_vars, result",
    (
        [["dQu", "dQv", "dQx", "dQy"], False],
        [["dQ1", "dQ2", "dQx", "dQy"], True],
        [["dQ1", "dQ2"], KeyError],
    )
)
def test__wind_rotation_needed(available_data_vars, result):
    xy = ["dQx", "dQy"]
    latlon = ["dQu", "dQv"]
    if result in [True, False]:
        assert _wind_rotation_needed(available_data_vars, xy, latlon) == result
    else:
        with pytest.raises(KeyError):
            _wind_rotation_needed(available_data_vars, xy, latlon)
