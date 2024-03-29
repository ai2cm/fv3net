import yaml
from scream_run import ScreamConfig
import os
import contextlib
import subprocess

config_file = os.getenv("TEST_SCREAM_RUN_CONFIG")


@contextlib.contextmanager
def cwd(path):
    cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(cwd)


def test_runtime_option():
    with open(config_file) as f:
        scream_config = ScreamConfig.from_dict(yaml.safe_load(f))
    case_scripts_dir = os.path.join(
        scream_config.CASE_ROOT,
        scream_config.CASE_NAME,
        f"{scream_config.number_of_processors}x1",
        "case_scripts",
    )
    scream_config.set_runtime_option(case_scripts_dir)
    with cwd(case_scripts_dir):
        for opt in [
            "STOP_OPTION",
            "STOP_N",
            "REST_OPTION",
            "REST_N",
            "HIST_OPTION",
            "HIST_N",
        ]:
            output = subprocess.run(
                f"./xmlquery {opt}", shell=True, stdout=subprocess.PIPE
            )
            assert (
                str(scream_config.RUNTIME.__dict__[opt])
                == output.stdout.decode("utf-8").strip().split(" ")[-1]
            )
