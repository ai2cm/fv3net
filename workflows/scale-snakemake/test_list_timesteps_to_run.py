import pytest

from list_timesteps_to_run import timestep_from_url


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/one-step-run/C48/20160805.144500/",
            "20160805.144500",
        ),
        (
            "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/one-step-run/C48/20160805.144500",
            "20160805.144500",
        ),
    ],
)
def test_timestep_from_url(url, expected):
    assert timestep_from_url(url) == expected
