import fv3kube
import pytest


@pytest.mark.parametrize(
    "mappings, expected",
    (
        [[{"a": 1}, {"a": 2}], {"a": 2}],
        [[{"a": 1}, {"b": 3}], {"a": 1, "b": 3}],
        [[{"a": 1}, {"b": 3}, {"a": 2}], {"a": 2, "b": 3}],
        [[{"a": 1}, {"a": 2, "b": 3}], {"a": 2, "b": 3}],
        [[{"a": 1}, {"a": 2}], {"a": 2}],
        [[{"a": 1}, {"a": {"b": 2}}], {"a": {"b": 2}}],
        [[{"a": {"b": 1}}, {"a": {"b": 2}}], {"a": {"b": 2}}],
        [[{"a": 1}, {"a": 1, "b": 3}], {"a": 1, "b": 3}],
        [[{"a": 1}, {"a": 2, "b": 3}], {"a": 2, "b": 3}],
        [
            [{"patch_files": ["one"]}, {"patch_files": ["two"]}],
            {"patch_files": ["one", "two"]},
        ],
        [
            [
                {"diagnostics": ["temp"]},
                {"patch_files": ["one"], "diagnostics": ["sphum"]},
            ],
            {"patch_files": ["one"], "diagnostics": ["temp", "sphum"]},
        ],
    ),
)
def test_merge_fv3config_overlays(mappings, expected):
    output = fv3kube.merge_fv3config_overlays(*mappings)
    assert output == expected
