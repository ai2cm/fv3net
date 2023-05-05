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


def test_invalid_vertical_coordinate_file():

    base_url = "/some/path"
    timestep = "20160805.000000"
    vertical_coordinate_file = "/some/path/"

    with pytest.raises(
        AssertionError, match="Provided vertical coordinate file is a directory"
    ):
        fv3kube.c48_initial_conditions_overlay(
            base_url, timestep, vertical_coordinate_file
        )


@pytest.mark.parametrize("vertical_coordinate_file", [None, "/some/path"])
def test_c48_initial_conditions_overlay(regtest, vertical_coordinate_file):

    url = "some/url"
    timestep = "20160801.000000"

    if vertical_coordinate_file is None:
        ans = fv3kube.c48_initial_conditions_overlay(url, timestep)
    else:
        ans = fv3kube.c48_initial_conditions_overlay(
            url, timestep, vertical_coordinate_file
        )

    print(ans, file=regtest)
