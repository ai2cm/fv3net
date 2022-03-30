"""Configuration class for submitting end to end experiments with argo
"""
from copy import deepcopy
import yaml
from typing import Mapping, Protocol, Optional, Sequence
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


class ArgoJob(Protocol):
    @property
    def parameters(self):
        """return dict of argo parameters"""
        pass

    @property
    def entrypoint(self) -> str:
        """name of argo template to use as an entrypoint"""
        pass


def submit(
    job: ArgoJob,
    labels: Mapping[str, str],
    other_parameters: Optional[Mapping[str, str]] = None,
):
    """Submit an ArgoJob to the cluster

    Args:
        job: the argo job to run
        labels: labels to apply to the k8s metadata of the argo job.
        other_parameters: Other common parameters to pass to the workflow an
            addition to ``job.parameters``
    """
    other_parameters = other_parameters or {}
    args = (
        [
            "argo",
            "submit",
            WORKFLOW_FILE.absolute().as_posix(),
            "--entrypoint",
            job.entrypoint,
            "--name",
            job.name,
        ]
        + _argo_parameters(dict(**job.parameters, **other_parameters))
        + _label_args(labels)
    )
    subprocess.check_call(args, stdout=subprocess.DEVNULL)


def _run_job_q(job):
    print(job)
    return input("Y/n") != "n"


def submit_jobs(jobs: Sequence[ArgoJob], experiment_name: str):
    """Submit a set of experiments

    Will display the set of proposed experiments and prompt the user
    if they should be run or not.

    Args:
        experiment_name: the argo workflow is tagged with this and the wandb
            jobs are tagged with ``experiment/{experiment_name}``

    """
    print("The following experiments are queued:")
    for job in jobs:
        print(job)

    ch = input("Submit? (Y/n,i)")
    if ch == "n":
        sys.exit(1)
    elif ch == "i":
        jobs = [job for job in jobs if _run_job_q(job)]

    with open("experiment-logs.txt", "a") as f:
        for job in jobs:
            submit(
                job,
                labels={"experiment": experiment_name},
                other_parameters={
                    "wandb-tags": f"experiment/{experiment_name}",
                    "wandb-group": experiment_name,
                },
            )
            print(job, file=f)
            print(job)
        print(
            f"to monitor experiments: argo list -lexperiment={experiment_name}", file=f
        )
    print(f"to monitor experiments: argo list -lexperiment={experiment_name}")


def _label_args(labels):
    label_vals = [f"{key}={val}" for key, val in labels.items()]
    return ["--labels", ",".join(label_vals)] if labels else []


@dataclasses.dataclass
class PrognosticJob:
    """A configuration for prognostic jobs

    Examples:

    A short configuration::

        from end_to_end import PrognosticJob, load_yaml

        job = PrognosticJob(
            name="test-v5",
            image_tag="d848e586db85108eb142863e600741621307502b",
            config=load_yaml("../configs/default_short.yaml"),
        )

    """

    name: str
    # dict for prognostic config, must be a
    # runtime.segmented_run.HighLevelConfig object.  Also supports a couple
    # other options. Consumed by projects/microphysics/scripts/prognostic_run.py
    # See the source for more information.
    config: dict = dataclasses.field(repr=False)
    image_tag: str

    @property
    def entrypoint(self):
        return "prognostic-run"

    @property
    def parameters(self):
        return {
            "config": _encode(self.config),
            "image_tag": self.image_tag,
        }


@dataclasses.dataclass
class TrainingJob:
    """

    Examples:

    A short configuration::

        from end_to_end import TrainingJob, load_yaml

        job = TrainingJob(
            name="test-v5",
            fv3fit_image_tag="d848e586db85108eb142863e600741621307502b",
            config=load_yaml("../train/rnn.yaml"),
        )

    """

    name: str
    config: dict = dataclasses.field(repr=False)
    fv3fit_image_tag: str
    bucket: str = "vcm-ml-experiments"
    project: str = "microphysics-emulation"

    @property
    def parameters(self):
        model_out_url = resolve_url(self.bucket, self.project, self.name)
        assert model_out_url.startswith(
            "gs://"
        ), f"Must use a remote URL. Got {model_out_url}"

        config = deepcopy(self.config)
        config["out_url"] = model_out_url
        return {
            "training-config": _encode(config),
            "fv3fit_image_tag": self.fv3fit_image_tag,
        }

    @property
    def entrypoint(self):
        return "training"


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

    @property
    def entrypoint(self):
        return "end-to-end"

    @property
    def parameters(self):
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
        return {
            "training-config": _encode(ml_config),
            "config": _encode(prog_config),
            "image_tag": self.image_tag,
            "fv3fit_image_tag": self.fv3fit_image_tag,
        }


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
