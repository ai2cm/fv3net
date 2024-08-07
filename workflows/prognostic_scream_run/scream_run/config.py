import dataclasses
from typing import Any, Dict, Union, Optional, TextIO
import vcm.cloud.gsutil
import os
import dacite
import datetime
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


def parse_scream_log(process: subprocess.Popen, f: TextIO):
    assert process.stdout, "stdout should not be None"
    for line in process.stdout:
        for out_file in [sys.stdout, f]:
            print(line.strip(), file=out_file)


class GSUtilResumableUploadException(Exception):
    pass


def _decode_to_str_if_bytes(s, encoding="utf-8"):
    if isinstance(s, bytes):
        return s.decode(encoding)
    else:
        return s


@dataclasses.dataclass
class RuntimeScreamConfig:
    upload_to_cloud_path: Optional[str] = None
    STOP_OPTION: str = "nhours"
    STOP_N: int = 1
    REST_OPTION: str = "nhours"
    REST_N: int = 1
    HIST_OPTION: str = "ndays"
    HIST_N: int = 1

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
    number_of_processors: int = 16
    CASE_ROOT: str = ""
    CASE_NAME: str = "scream_test"
    COMPSET: str = "F2010-SCREAMv1"
    RESOLUTION: str = "ne30pg2_ne30pg2"
    ATM_NCPL: int = 48
    RUN_STARTDATE: Union[str, datetime.date] = "2010-01-01"
    MODEL_START_TYPE: str = "initial"
    OLD_EXECUTABLE: str = ""
    RUNTIME: RuntimeScreamConfig = dataclasses.field(
        default_factory=RuntimeScreamConfig
    )

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

    def set_runtime_option(self, case_scripts_dir: str):
        with cwd(case_scripts_dir):
            subprocess.run(
                f"./xmlchange STOP_OPTION={self.RUNTIME.STOP_OPTION}", shell=True
            )
            subprocess.run(f"./xmlchange STOP_N={self.RUNTIME.STOP_N}", shell=True)
            subprocess.run(
                f"./xmlchange REST_OPTION={self.RUNTIME.REST_OPTION}", shell=True
            )
            subprocess.run(f"./xmlchange REST_N={self.RUNTIME.REST_N}", shell=True)
            subprocess.run(
                f"./xmlchange HIST_OPTION={self.RUNTIME.HIST_OPTION}", shell=True
            )
            subprocess.run(f"./xmlchange HIST_N={self.RUNTIME.HIST_N}", shell=True)

    def submit_scream_run(self, rebuild: bool = False):
        case_scripts_dir = os.path.join(
            self.CASE_ROOT,
            self.CASE_NAME,
            f"{self.number_of_processors}x1",
            "case_scripts",
        )
        case_run_dir = os.path.join(
            self.CASE_ROOT, self.CASE_NAME, f"{self.number_of_processors}x1", "run",
        )
        with cwd(case_scripts_dir):
            with open("logs.txt", "w") as f:
                if rebuild:
                    process = subprocess.Popen(
                        ["./case.build",],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    parse_scream_log(process, f)
                self.set_runtime_option(case_scripts_dir)
                process = subprocess.Popen(
                    ["./case.submit",],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                parse_scream_log(process, f)

        if self.RUNTIME.upload_to_cloud_path is not None:
            output_dir = os.path.join(self.RUNTIME.upload_to_cloud_path, self.CASE_NAME)
            try:
                print(f"Uploading {case_run_dir} to {output_dir}")
                subprocess.check_output(
                    ["gsutil", "-m", "rsync", "-r", "-e", case_run_dir, output_dir,],
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as e:
                output = _decode_to_str_if_bytes(e.output)
                if "ResumableUploadException" in output:
                    raise GSUtilResumableUploadException()
                else:
                    raise e

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "ScreamConfig":
        if "RUNTIME" in kwargs:
            kwargs["RUNTIME"] = RuntimeScreamConfig.from_dict(kwargs["RUNTIME"])
        return dacite.from_dict(
            data_class=cls, data=kwargs, config=dacite.Config(strict=True)
        )
