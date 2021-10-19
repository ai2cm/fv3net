from io import StringIO
import yaml
import fv3fit
import os

DIRNAME = os.path.dirname(os.path.abspath(__file__))


def test_argo_yaml_has_valid_training_configs():
    with open(os.path.join(DIRNAME, "argo.yaml"), "r") as f:
        argo_data = yaml.safe_load(f)
    parameters_list = argo_data["spec"]["arguments"]["parameters"]
    parameters = {entry["name"]: entry["value"] for entry in parameters_list}
    all_configs = yaml.safe_load(StringIO(parameters["training-configs"]))
    assert len(all_configs) > 0
    for config in all_configs:
        fv3fit.TrainingConfig.from_dict(config["config"])
