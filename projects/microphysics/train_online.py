from typing import List
import sys
import os

sys.path.insert(0, "scripts")  # noqa
import prognostic_run
from generic_algorithms import TrainOnline
from enum import Enum
import uuid
from fv3fit.train_microphysics import (
    main as train_microphysics,
    get_default_config,
    ArchitectureConfig,
)


Model = str  # a path to a trained model
Dataset = str  # a path to the generated dataset


id_ = uuid.uuid4().hex
base_url = f"gs://vcm-ml-scratch/noahb/test-output/{id_}"

os.environ["WANDB_RUN_GROUP"] = id_


class JobTypes(Enum):
    train = "models"
    datasets = "data"


def get_url(job_type: str, cycle: int):
    return os.path.join(base_url, job_type, str(cycle))


def train(cycle: int, data: Dataset):
    path = "artifacts/20160611.000000/netcdf_output"
    data = os.path.join(data, path)
    config = get_default_config()
    config.train_url = data
    config.test_url = data
    config.out_url = get_url(JobTypes.train.value, cycle)
    config.model.architecture = ArchitectureConfig("dense")
    train_microphysics(config, seed=0)


def generate(cycle: int, models: List[int], duration: int) -> str:
    """
    use the final model as the active model

    duration is hours
    """
    common = [
        "--tag",
        f"{id_}-cycle{cycle}",
        "--output-frequency",
        "900",
    ]
    if cycle == 0:
        args = common + ["--offline"]
    else:
        args = common + [
            "--model",
            models[-1],
            "--online",
        ]

    parser = prognostic_run.get_parser()
    config = parser.parse_args(args)
    return prognostic_run.main(config)


online_trainer = TrainOnline(train, generate)
# duration isn't used right now
online_trainer(10101202, cycles=10, initial_model="doesn't matter")
