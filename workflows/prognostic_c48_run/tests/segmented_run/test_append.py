from runtime.segmented_run.append import read_last_segment


def test_read_last_segment(tmpdir):
    date1 = "20160101.000000"
    date2 = "20160102.000000"
    arts = tmpdir.mkdir("artifacts")
    arts.mkdir(date1)
    arts.mkdir(date2)
    ans = read_last_segment(str(tmpdir))
    assert f"file:/{str(tmpdir)}/artifacts/20160102.000000" == ans
