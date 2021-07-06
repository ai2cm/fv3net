import pytest

import data_transform_mock
from fv3fit.emulation.data import config


@pytest.fixture
def mocked_cfg_transforms(monkeypatch):

    monkeypatch.setattr(config, "transforms", data_transform_mock)


def test__TransformConfigItem():

    name = "transform_func"
    args = [1, 2, 3]
    kwargs = dict(kwarg1=1, kwarg2=2)

    item = config._TransformConfigItem(name, args, kwargs)

    assert item.name == name
    assert item.args == args
    assert item.kwargs == kwargs


def test__TransformConfigItem_args_mapping_to_sequence():

    name = "transform_func"
    args = dict(arg1=1, arg2=2, arg3=3)

    item = config._TransformConfigItem(name, args)

    assert item.args == [1, 2, 3]


def test__TransformConfigItem_from_dict():

    name = "transform_func"
    args = [1, 2, 3]
    kwargs = dict(kwarg1=1, kwarg2=2)
    item_dict = dict(name=name, args=args, kwargs=kwargs)
    item = config._TransformConfigItem.from_dict(item_dict)

    assert item.name == name
    assert item.args == args
    assert item.kwargs == kwargs


@pytest.mark.parametrize(
    "name, args, kwargs",
    [
        ("dummy_transform", [], {}),
        ("dummy_transform_w_kwarg", [], {"extra_kwarg": "dummy"}),
        ("dummy_transform_w_leading_arg", ["leading"], {}),
    ],
)
def test__TransformConfigItem_load_transform(name, args, kwargs, mocked_cfg_transforms):

    item = config._TransformConfigItem(name, args, kwargs)
    func = item.load_transform_func()
    assert callable(func)
    assert func(1) == 1


def test__TransformConfigItem_load_transform_func_overspecified(mocked_cfg_transforms):

    name = "dummy_transform"

    # dummy has no extra args/kwargs besides dataset
    overspecified = config._TransformConfigItem(name, [1])
    with pytest.raises(TypeError):
        overspecified.load_transform_func()


def test__load_transforms(mocked_cfg_transforms):

    item = config._TransformConfigItem("dummy_transform")
    composed = config._load_transforms([item, item, item])
    assert callable(composed)
    assert composed(1) == 1


def test_TransformConfig(mocked_cfg_transforms):

    config_item = config._TransformConfigItem("dummy_transform")
    transform_seq = [config_item] * 3

    base_config = config.TransformConfig(transforms=transform_seq)
    pipeline_func = base_config.get_transform_pipeline()

    assert callable(pipeline_func)
    assert pipeline_func(1) == 1


def test_TransformConfig_empty_transforms(mocked_cfg_transforms):
    base_config = config.TransformConfig()
    pipeline_func = base_config.get_transform_pipeline()

    assert callable(pipeline_func)
    assert pipeline_func(1) == 1


def test_TransformConfig__initialize_custom_transforms():
    custom_transforms = {"transforms": [{"name": "dummy_transform"}] * 3}
    result = config.TransformConfig._initialize_custom_transforms(custom_transforms)

    assert "transforms" in result
    for item in result["transforms"]:
        assert isinstance(item, config._TransformConfigItem)


def test_TransformConfig_from_dict(mocked_cfg_transforms):

    custom_transforms = {"transforms": [{"name": "dummy_transform"}] * 3}
    result = config.TransformConfig.from_dict(custom_transforms)

    assert isinstance(result, config.TransformConfig)
