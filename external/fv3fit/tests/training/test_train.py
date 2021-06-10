import pytest
from fv3fit._shared.config import ESTIMATORS


@pytest.fixture(params=list(ESTIMATORS.keys()))
def model_type(request):
    return request.param
