import pytest
from runtime.config import get_model_urls


@pytest.mark.parametrize(
    "config, model_urls",
    [
        ({}, []),
        ({"scikit_learn": None, "prephysics": None}, []),
        ({"scikit_learn": None, "prephysics": [{"some_prescribed_data": 0}]}, []),
        (
            {
                "scikit_learn": {"model": ["ml_model_url"]},
                "prephysics": [{"some_prescribed_data": 0}],
            },
            ["ml_model_url"],
        ),
        (
            {
                "scikit_learn": {"model": ["ml_model_url"]},
                "prephysics": [
                    {"some_prescribed_data": 0},
                    {"model": ["prephysics_model_0", "prephysics_model_1"]},
                ],
            },
            ["ml_model_url", "prephysics_model_0", "prephysics_model_1"],
        ),
    ],
)
def test_get_model_urls(config, model_urls):
    assert set(get_model_urls(config)) == set(model_urls)
