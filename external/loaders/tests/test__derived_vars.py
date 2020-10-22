import pytest
from loaders.batches._derived_vars import (
    nonderived_variable_names,
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


