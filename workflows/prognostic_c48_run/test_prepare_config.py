import prepare_config

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
