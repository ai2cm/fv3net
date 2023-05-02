import dataclasses
from typing import Any, Dict, Union
import vcm.cloud.gsutil
import os
import dacite
import shutil
import sys
from dataclasses import asdict

# TODO: importlib.resources.files is not available prior to python 3.9
if sys.version_info.major == 3 and sys.version_info.minor < 9:
    import importlib_resources  # type: ignore
elif sys.version_info.major == 3 and sys.version_info.minor >= 9:
    import importlib.resources as importlib_resources  # type: ignore


def gather_output_yaml(output_yaml: str, rundir: str):
    fs = vcm.cloud.get_fs(output_yaml)
    assert fs.exists(output_yaml), f"{output_yaml} does not exist"
    local_filename = os.path.join(rundir, os.path.basename(output_yaml))
    local_filename = os.path.abspath(local_filename)
    fs.get(output_yaml, local_filename)
    return local_filename


@dataclasses.dataclass
class ScreamConfig:
    output_yaml: Union[str, list] = "gs://vcm-scream/config/output/default.yaml"
    initial_conditions_type: str = "local"
    create_newcase: bool = True
    case_setup: bool = True
    case_build: bool = True
    case_submit: bool = False
    upload_to_cloud: bool = False
    upload_to_cloud_path: str = "gs://vcm-ml-scratch/scream/scream_test"
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

    def __post__init__(self):
        # TODO: we may want to support other option
        # such as initial_conditions_type = "cloud"
        # where we need to download the initial conditions files
        assert (
            self.initial_conditions_type == "local"
        ), "at the moment, initial_conditions_type must be local, \
            meaning the input files were already on disk or \
            mounted through persistentVolume"

    def resolve_output_yaml(self, rundir: str):
        self.output_yaml = (
            [self.output_yaml]
            if isinstance(self.output_yaml, str)
            else self.output_yaml
        )
        local_output_yaml = []
        for filename in self.output_yaml:
            local_output_yaml.append(gather_output_yaml(filename, rundir))
        self.output_yaml = local_output_yaml

    def compose_run_scream_commands(self, rundir: str):
        run_script = importlib_resources.files("scream_run.template").joinpath(
            "run_eamxx.sh"
        )
        local_script = os.path.join(rundir, os.path.basename(run_script))
        shutil.copy(run_script, local_script)
        command = local_script
        for key, value in asdict(self).items():
            if isinstance(value, list):
                value = ",".join(value)
            command += f" --{key} {value}"
        return command

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "ScreamConfig":
        return dacite.from_dict(
            data_class=cls, data=kwargs, config=dacite.Config(strict=True)
        )
