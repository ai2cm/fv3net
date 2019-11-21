import pytest

from fv3net.pipelines.extract_tars.transforms import not_finished_with_tar_extract


@pytest.mark.parametrize(
    'timestep, result',
    [('20160801.003000', False),
     ('20160801.004500', True),
     ('20160801.010000', True),
     ('20160801.011500', True)]
)
def test_not_finished_extract(timestep, result):

    """
    generated the test data on GCS using
    misc/upload_test_filter_check_data.py

    20160801.003000 - contains all files
    20160801.004500 - missing one surface file
    20160801.010000 - missing one fv_srf_wnd_coarse file
    20160801.011500 - empty directory with no files
    """

    output_prefix = 'tmp_dataflow/test_data_extract_check'
    fake_tstep_gcs_tar_url = f'gs://vcm-ml-data/{timestep}.tar'
    is_not_finished = not_finished_with_tar_extract(fake_tstep_gcs_tar_url,
                                                    output_prefix,
                                                    num_tiles=1,
                                                    num_subtiles=1)

    assert is_not_finished == result


@pytest.mark.parametrize(
    'num_tiles, num_subtiles',
    [(0, 1), (1, 0), (-1, -1)]
)
def test_not_finished_extract_non_positive_tiles(num_tiles, num_subtiles):
    timestep = '20160801.003000'
    output_prefix = 'tmp_dataflow/test_data_extract_check'
    fake_tstep_gcs_tar_url = f'gs://vcm-ml-data/{timestep}.tar'
    with pytest.raises(ValueError):
        not_finished_with_tar_extract(fake_tstep_gcs_tar_url, output_prefix,
                                      num_tiles=num_tiles,
                                      num_subtiles=num_subtiles)
