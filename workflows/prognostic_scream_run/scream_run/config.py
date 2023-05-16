import dataclasses
from typing import Any, Dict, Union, Optional
import vcm.cloud.gsutil
import os
import dacite
import sys
import contextlib
import subprocess
from dataclasses import asdict


@contextlib.contextmanager
def cwd(path):
    cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(cwd)


def gather_output_yaml(output_yaml: str, rundir: str):
    fs = vcm.cloud.get_fs(output_yaml)
    assert fs.exists(output_yaml), f"{output_yaml} does not exist"
    local_filename = os.path.join(rundir, os.path.basename(output_yaml))
    local_filename = os.path.abspath(local_filename)
    fs.get(output_yaml, local_filename)
    return local_filename


@dataclasses.dataclass
class RuntimeScreamConfig:
    upload_to_cloud_path: Optional[str] = None

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "RuntimeScreamConfig":
        return dacite.from_dict(
            data_class=cls, data=kwargs, config=dacite.Config(strict=True)
        )


@dataclasses.dataclass
class ScreamConfig:
    output_yaml: Union[str, list] = "gs://vcm-scream/config/output/default.yaml"
    initial_conditions_type: str = "local"
    create_newcase: bool = True
    case_setup: bool = True
    case_build: bool = True
    number_of_processers: int = 16
    CASE_ROOT: str = ""
    CASE_NAME: str = "scream_test"
    COMPSET: str = "F2010-SCREAMv1"
    RESOLUTION: str = "ne30pg2_ne30pg2"
    ATM_NCPL: int = 48
    STOP_OPTION: str = "nhours"
    STOP_N: int = 1
    REST_OPTION: str = "nhours"
    REST_N: int = 1
    HIST_OPTION: str = "ndays"
    HIST_N: int = 1
    RUN_STARTDATE: str = "2010-01-01"
    MODEL_START_TYPE: str = "initial"
    OLD_EXECUTABLE: str = ""
    RUNTIME: RuntimeScreamConfig = RuntimeScreamConfig()

    def __post__init__(self):
        # TODO: we may want to support other option
        # such as initial_conditions_type = "cloud"
        # where we need to download the initial conditions files
        assert (
            self.initial_conditions_type == "local"
        ), "at the moment, initial_conditions_type must be local, \
            meaning the input files were already on disk or \
            mounted through persistentVolume"

    def get_local_output_yaml(self, rundir: str) -> list:
        self.output_yaml = (
            [self.output_yaml]
            if isinstance(self.output_yaml, str)
            else self.output_yaml
        )
        local_output_yaml = []
        for filename in self.output_yaml:
            local_output_yaml.append(gather_output_yaml(filename, rundir))
        return local_output_yaml

    def compose_write_scream_run_directory_command(
        self, local_output_yaml: list, local_run_script: str
    ):
        command = local_run_script
        for key, value in asdict(self).items():
            if key != "RUNTIME":
                if isinstance(value, list):
                    if key == "output_yaml":
                        value = ",".join(local_output_yaml)
                    else:
                        value = ",".join(value)
                command += f" --{key} {value}"
        return command

    def submit_scream_run(self):
        case_scripts_dir = os.path.join(
            self.CASE_ROOT,
            self.CASE_NAME,
            f"{self.number_of_processers}x1",
            "case_scripts",
        )
        case_run_dir = os.path.join(
            self.CASE_ROOT, self.CASE_NAME, f"{self.number_of_processers}x1", "run",
        )
        with cwd(case_scripts_dir):
            with open("logs.txt", "w") as f:
                process = subprocess.Popen(
                    ["./case.submit",],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                # need this assertion so that mypy knows that stdout is not None
                assert process.stdout, "stdout should not be None"
                for line in process.stdout:
                    for out_file in [sys.stdout, f]:
                        print(line.strip(), file=out_file)

        if self.RUNTIME.upload_to_cloud_path is not None:
            output_dir = os.path.join(self.RUNTIME.upload_to_cloud_path, self.CASE_NAME)
            vcm.cloud.get_fs(output_dir).put(case_run_dir, output_dir)

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "ScreamConfig":
        if "RUNTIME" in kwargs:
            kwargs["RUNTIME"] = RuntimeScreamConfig.from_dict(kwargs["RUNTIME"])
        return dacite.from_dict(
            data_class=cls, data=kwargs, config=dacite.Config(strict=True)
        )
