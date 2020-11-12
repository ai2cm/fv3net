import prepare_config

MODEL_URL = "gs://ml-model"
IC_URL = "gs://ic-bucket"
IC_TIMESTAMP = "20160805.000000"
ML_CONFIG_UPDATE = "prognostic_config.yml"
NUDGING_CONFIG_UPDATE = "nudging/nudging_config.yaml"
OTHER_FLAGS = ["--nudge-to-observations"]


def get_ml_args():
    return [
        ML_CONFIG_UPDATE,
        IC_URL,
        IC_TIMESTAMP,
        "--model_url",
        MODEL_URL,
    ] + OTHER_FLAGS


def get_nudging_args():
    return [
        NUDGING_CONFIG_UPDATE,
        IC_URL,
        IC_TIMESTAMP,
    ]


def test_prepare_ml_config_regression(regtest):
    parser = prepare_config._create_arg_parser()
    args = parser.parse_args(get_ml_args())
    with regtest:
        prepare_config.prepare_config(args)


def test_prepare_nudging_config_regression(regtest):
    parser = prepare_config._create_arg_parser()
    args = parser.parse_args(get_nudging_args())
    with regtest:
        prepare_config.prepare_config(args)
