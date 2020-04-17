import generate


def test_config_to_configmap():
    configs = {"a": "1", "b": "5"}
    generate._configs_to_configmap(configs)