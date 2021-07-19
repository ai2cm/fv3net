import numpy as np

import emulation_slim.debug as debug


def test_dump_state(tmpdir, monkeypatch):

    monkeypatch.setenv("STATE_DUMP_PATH", str(tmpdir))

    state = {
        "rank": np.ones((10), dtype=np.float),
        "data": np.random.randn(10)
    }

    debug.dump_state(state)

    files = tmpdir.listdir()
    assert len(files) == 1
    filepath = str(files[0])
    assert "npz" in filepath

    reloaded = np.load(filepath)
    np.testing.assert_equal(reloaded["data"], state["data"])