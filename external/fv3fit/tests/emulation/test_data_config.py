from fv3fit.emulation.data import config


def test_InputTransformConfig():

    result = config.InputTransformConfig(
        input_variables=["a", "b"],
        output_variables=["c", "d"],
        antarctic_only=False,
        vertical_subselections={"a": slice(5, None)},
    )

    transform_func = result.get_transform_pipeline()
    assert callable(transform_func)


def test_InputTransformConfig_from_dict():

    result = config.InputTransformConfig.from_dict(
        dict(
            input_variables=["a", "b"],
            output_variables=["c", "d"],
            antarctic_only=False,
            vertical_subselections={"a": slice(5, None)},
        )
    )

    assert isinstance(result, config.InputTransformConfig)
    transform_func = result.get_transform_pipeline()
    assert callable(transform_func)
