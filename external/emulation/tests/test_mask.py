from emulation.masks import RangeMask, compose_masks


def test_RangeMask():
    mask = RangeMask("foo", min=0, max=1)
    assert mask({}, {"foo": 0.5}) == {"foo": 0.5}
    assert mask({}, {"foo": 1.5}) == {"foo": 1.0}
    assert mask({}, {"foo": -1.5}) == {"foo": 0}


def test_compose_masks_no_action():
    mask = compose_masks([])
    out = {"a": 1}
    assert mask({}, out) == out
