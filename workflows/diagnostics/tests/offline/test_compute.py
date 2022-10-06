import dataclasses
import logging
import tempfile
import os
import loaders
import synth
from synth import (  # noqa: F401
    grid_dataset,
    grid_dataset_path,
    dataset_fixtures_dir,
)

import fv3fit
from fv3net.diagnostics.offline._helpers import DATASET_DIM_NAME
from fv3net.diagnostics.offline import compute
from fv3net.diagnostics.offline.views import create_report
import pathlib
import pytest
import numpy as np
import xarray as xr
import yaml

logger = logging.getLogger(__name__)


@pytest.fixture(params=["single_dataset", "multiple_datasets"])
def data_path(tmpdir, request):
    schema_path = pathlib.Path(__file__).parent / "data.zarr.json"

    with open(schema_path) as f:
        schema = synth.load(f)

    ranges = {"pressure_thickness_of_atmospheric_layer": synth.Range(0.99, 1.01)}
    ds = synth.generate(schema, ranges)
    if request.param == "multiple_datasets":
        ds = xr.concat([ds, ds], dim=DATASET_DIM_NAME)

    ds.to_zarr(str(tmpdir), consolidated=True)
    return str(tmpdir)


def test_offline_diags_integration(data_path, grid_dataset_path):  # noqa: F811
    """
    Test the bash endpoint for computing offline diagnostics
    """

    batches_kwargs = {
        "needs_grid": False,
        "res": "c8",
        "timesteps_per_batch": 1,
        "timesteps": ["20160801.001500", "20160801.003000"],
    }
    trained_model = fv3fit.testing.ConstantOutputPredictor(
        input_variables=["air_temperature", "specific_humidity"],
        output_variables=["dQ1", "dQ2"],
    )
    trained_model.set_outputs(dQ1=np.zeros([19]), dQ2=np.zeros([19]))
    data_config = loaders.BatchesFromMapperConfig(
        loaders.MapperConfig(function="open_zarr", kwargs={"data_path": data_path},),
        **batches_kwargs,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "trained_model")
        fv3fit.dump(trained_model, model_dir)
        data_config_filename = os.path.join(tmpdir, "data_config.yaml")
        with open(data_config_filename, "w") as f:
            yaml.safe_dump(dataclasses.asdict(data_config), f)
        compute_diags_args = compute._get_parser().parse_args(
            [
                model_dir,
                data_config_filename,
                os.path.join(tmpdir, "offline_diags"),
                "--evaluation-grid",
                grid_dataset_path,
                "--n-jobs",
                "1",
            ]
        )
        compute.main(compute_diags_args)
        if isinstance(data_config, loaders.BatchesFromMapperConfig):
            assert "transect_lon0.nc" in os.listdir(
                os.path.join(tmpdir, "offline_diags")
            )
        create_report_args = create_report._get_parser().parse_args(
            [
                os.path.join(tmpdir, "offline_diags"),
                os.path.join(tmpdir, "report"),
                "--no-wandb",
            ]
        )
        create_report.create_report(create_report_args)
        with open(os.path.join(tmpdir, "report/index.html")) as f:
            report = f.read()
        if isinstance(data_config, loaders.BatchesFromMapperConfig):
            assert "Transect snapshot at" in report
