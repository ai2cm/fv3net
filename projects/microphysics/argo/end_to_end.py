"""Configuration class for submitting end to end experiments with argo
"""
from copy import deepcopy
import yaml
import subprocess
import dataclasses
import base64
import os
from fv3net.artifacts.resolve_url import resolve_url
import pathlib
import sys


WORKFLOW_FILE = pathlib.Path(__file__).parent / "argo.yaml"


def load_yaml(path):
    return yaml.safe_load(pathlib.Path(path).read_bytes())


def submit_jobs(jobs, experiment_name):
    print("The following experiments are queued:")
    for job in jobs:
        print(job)

    ch = input("Submit? (Y/n)")
    if ch == "n":
        sys.exit(1)
    print("Submitting all these jobs")
    for job in jobs:
        with open("experiment-logs.txt", "a") as f:
            f.write(repr(job))
        job.submit(labels={"experiment": experiment_name})
    print(f"to monitor experiments: argo list -lexperiment={experiment_name}")


def _label_args(labels):
    label_vals = [f"{key}={val}" for key, val in labels.items()]
    return ["--labels", ",".join(label_vals)] if labels else []


@dataclasses.dataclass
class EndToEndJob:
    """A configuration for prognostic jobs

    Examples:

    A short configuration::

        from end_to_end import EndToEndJob, load_yaml

        job = EndToEndJob(
            name="test-v5",
            fv3fit_image_tag="d848e586db85108eb142863e600741621307502b",
            image_tag="d848e586db85108eb142863e600741621307502b",
            ml_config=load_yaml("../train/rnn.yaml"),
            prog_config=load_yaml("../configs/default_short.yaml"),
        )

    """

    name: str
    ml_config: dict = dataclasses.field(repr=False)
    prog_config: dict = dataclasses.field(repr=False)
    fv3fit_image_tag: str
    image_tag: str
    bucket: str = "vcm-ml-experiments"
    project: str = "microphysics-emulation"

    def to_argo_args(self, labels=()):
        model_out_url = resolve_url(self.bucket, self.project, self.name)
        assert model_out_url.startswith(
            "gs://"
        ), f"Must use a remote URL. Got {model_out_url}"

        ml_config = deepcopy(self.ml_config)
        ml_config["out_url"] = model_out_url

        prog_config = deepcopy(self.prog_config)
        model_path = os.path.join(model_out_url, "model.tf")
        assert model_path.startswith("gs://")
        prog_config["zhao_carr_emulation"] = {"model": {"path": model_path}}
        parameters = _argo_parameters(
            {
                "training-config": _encode(ml_config),
                "config": _encode(prog_config),
                "image_tag": self.image_tag,
                "fv3fit_image_tag": self.fv3fit_image_tag,
            }
        )

        return (
            [
                "argo",
                "submit",
                WORKFLOW_FILE.absolute().as_posix(),
                "--entrypoint",
                "end-to-end",
                "--name",
                self.name,
            ]
            + parameters
            + _label_args(labels)
        )

    def submit(self, labels):
        print(self)
        subprocess.check_call(self.to_argo_args(labels), stdout=subprocess.DEVNULL)


def _encode(dict):
    s = yaml.safe_dump(dict).encode()
    return base64.b64encode(s).decode()


def _argo_parameters(kwargs):
    args = []
    for key, val in kwargs.items():
        if isinstance(val, dict):
            encoded = _encode(val)
        else:
            encoded = str(val)
        args.extend(["-p", key + "=" + encoded])
    return args
