import prepare_config
import pytest

MODEL_URL = "gs://ml-model"
IC_URL = "gs://ic-bucket"
IC_TIMESTAMP = "20160805.000000"
CONFIG_UPDATE = "prognostic_config.yml"
OTHER_FLAGS = ["--nudge-to-observations"]


def get_args():
    return [
        IC_URL,
        IC_TIMESTAMP,
        "--model_url",
        MODEL_URL,
        "--prog_config_yml",
        CONFIG_UPDATE,
    ] + OTHER_FLAGS


def test_prepare_config_regression(regtest):
    parser = prepare_config._create_arg_parser()
    args = parser.parse_args(get_args())
    with regtest:
        prepare_config.prepare_config(args)


@pytest.mark.parametrize(
    "mappings, expected",
    (
        [[{"a": 1}, {"a": 2}], {"a": 2}],
        [[{"a": 1}, {"b": 3}], {"a": 1, "b": 3}],
        [[{"a": 1}, {"b": 3}, {"a": 2}], {"a": 2, "b": 3}],
        [[{"a": 1}, {"a": 2, "b": 3}], {"a": 2, "b": 3}],
    ),
)
def test_merge_fv3config_overlays(mappings, expected):
    output = prepare_config.merge_fv3config_overlays(*mappings)
    assert output == expected


@pytest.mark.parametrize(
    "source, update, expected",
    (
        [{"a": 1}, {"a": 2}, {"a": 2}],
        [{"a": 1}, {"a": {"b": 2}}, {"a": {"b": 2}}],
        [{"a": 1}, {"a": 1, "b": 3}, {"a": 1, "b": 3}],
        [{"a": 1}, {"a": 2, "b": 3}, {"a": 2, "b": 3}],
        [
            {"patch_files": ["one"]},
            {"patch_files": ["two"]},
            {"patch_files": ["one", "two"]},
        ],
        [
            {"diagnostics": ["temp"]},
            {"patch_files": ["one"], "diagnostics": ["sphum"]},
            {"patch_files": ["one"], "diagnostics": ["temp", "sphum"]},
        ],
    ),
)
def test__merge_once(source, update, expected):
    output = prepare_config._merge_once(source, update)
    assert output == expected
