from loaders.mappers import open_fine_resolution_nudging_hybrid
import pytest
import synth


# timestep info
timestep1 = "20160801.000730"
timestep1_end = "20160801.001500"
timestep2 = "20160801.002230"
times_centered_str = [timestep1, timestep2]


@pytest.fixture
def nudging_url(tmpdir):
    nudging_url = str(tmpdir.mkdir("nudging"))
    synth.generate_nudging(nudging_url)
    return nudging_url


@pytest.fixture
def fine_url(tmpdir):
    fine_url = str(tmpdir.mkdir("fine_res"))
    synth.generate_fine_res(fine_url, times_centered_str)
    return fine_url


def test_open_fine_resolution_nudging_hybrid(nudging_url, fine_url):
    # test opener
    data = open_fine_resolution_nudging_hybrid(
        None, {"url": nudging_url}, {"fine_res_url": fine_url}
    )
    data[timestep1_end]


def test_open_fine_resolution_nudging_hybrid_data_path(nudging_url, fine_url):
    # passes the urls as data_paths
    data = open_fine_resolution_nudging_hybrid([nudging_url, fine_url], {}, {})
    data[timestep1_end]
