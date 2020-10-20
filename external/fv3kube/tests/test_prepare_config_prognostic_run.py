import os
import fv3kube.prepare_config.prognostic_run as prepare_config

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_UPDATE = os.path.join(FILE_DIR, "prognostic_config.yml")


MODEL_URL = "gs://ml-model"
IC_URL = "gs://ic-bucket"
IC_TIMESTAMP = "20160805.000000"
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
