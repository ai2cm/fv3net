import pytest
import numpy as np

from emulation.masks import RangeMask, LevelMask, compose_masks


def test_RangeMask():
    mask = RangeMask("foo", min=0, max=1)
    assert mask({}, {"foo": 0.5}) == {"foo": 0.5}
    assert mask({}, {"foo": 1.5}) == {"foo": 1.0}
    assert mask({}, {"foo": -1.5}) == {"foo": 0}


def test_compose_masks_no_action():
    mask = compose_masks([])
    out = {"a": 1}
    assert mask({}, out) == out


@pytest.mark.parametrize("start, stop", [(2, 3), (2, 5), (None, 2), (None, None)])
def test_LevelMask(start, stop):
    mask = LevelMask("foo", start, stop)
    ones = np.ones((4, 2))
    zeros = ones * 0
    result = mask(state={"foo": zeros}, emulator={"foo": ones})

    sl_ = slice(start, stop)
    num_fortran_elements = zeros[sl_].size
    np.testing.assert_array_equal(result["foo"][sl_], zeros[sl_])
    assert np.sum(result["foo"]) == result["foo"].size - num_fortran_elements
