import synth
import xarray as xr
import numpy as np
import pytest
import os
import tempfile
import logging
import diagnostics_utils as utils
import intake
from loaders import mappers, batches, SAMPLE_DIM_NAME
from fv3net.regression.sklearn import train
from fv3net.regression.sklearn._mapper import SklearnPredictionMapper
from offline_ml_diags._metrics import calc_metrics


logger = logging.getLogger(__name__)

DOMAINS = ["land", "sea", "global"]
OUTPUT_NC_NAME = "diagnostics.nc"


def generate_one_step_dataset(datadir_module, one_step_dir):

    with open(str(datadir_module.join("one_step.json"))) as f:
        one_step_schema = synth.load(f)
    one_step_dataset = synth.generate(one_step_schema)
    one_step_dataset_1 = one_step_dataset.assign_coords(
        {"initial_time": ["20160901.000000"]}
    )
    one_step_dataset_2 = one_step_dataset.assign_coords(
        {"initial_time": ["20160901.001500"]}
    )
    one_step_dataset_1.to_zarr(
        os.path.join(one_step_dir, "20160901.000000.zarr"), consolidated=True,
    )
    one_step_dataset_2.to_zarr(
        os.path.join(one_step_dir, "20160901.001500.zarr"), consolidated=True,
    )


@pytest.fixture(scope="module")
def one_step_dataset_path(datadir_module):

    with tempfile.TemporaryDirectory() as one_step_dir:
        generate_one_step_dataset(datadir_module, one_step_dir)
        yield one_step_dir


def generate_nudging_dataset(datadir_module, nudging_dir):

    nudging_after_dynamics_zarrpath = os.path.join(
        nudging_dir, "outdir-3h", "after_dynamics.zarr"
    )
    with open(str(datadir_module.join("after_dynamics.json"))) as f:
        nudging_after_dynamics_schema = synth.load(f)
    nudging_after_dynamics_dataset = synth.generate(
        nudging_after_dynamics_schema
    ).assign_coords(
        {
            "time": [
                np.datetime64("2016-09-01T00:00:00"),
                np.datetime64("2016-09-01T00:15:00"),
            ]
        }
    )
    nudging_after_dynamics_dataset.to_zarr(
        nudging_after_dynamics_zarrpath, consolidated=True
    )

    nudging_after_physics_zarrpath = os.path.join(
        nudging_dir, "outdir-3h", "after_physics.zarr"
    )
    with open(str(datadir_module.join("after_physics.json"))) as f:
        nudging_after_physics_schema = synth.load(f)
    nudging_after_physics_dataset = synth.generate(
        nudging_after_physics_schema
    ).assign_coords(
        {
            "time": [
                np.datetime64("2016-09-01T00:00:00"),
                np.datetime64("2016-09-01T00:15:00"),
            ]
        }
    )
    nudging_after_physics_dataset.to_zarr(
        nudging_after_physics_zarrpath, consolidated=True
    )

    nudging_tendencies_zarrpath = os.path.join(
        nudging_dir, "outdir-3h", "nudging_tendencies.zarr"
    )
    with open(str(datadir_module.join("nudging_tendencies.json"))) as f:
        nudging_tendencies_schema = synth.load(f)
    nudging_tendencies_dataset = synth.generate(
        nudging_tendencies_schema
    ).assign_coords(
        {
            "time": [
                np.datetime64("2016-09-01T00:00:00"),
                np.datetime64("2016-09-01T00:15:00"),
            ]
        }
    )
    nudging_tendencies_dataset.to_zarr(nudging_tendencies_zarrpath, consolidated=True)


@pytest.fixture(scope="module")
def nudging_dataset_path(datadir_module):

    with tempfile.TemporaryDirectory() as nudging_dir:
        generate_nudging_dataset(datadir_module, nudging_dir)
        yield nudging_dir


def generate_fine_res_dataset(datadir_module, fine_res_dir):
    """ Note that this does not follow the pattern of the other two datasets
    in that the synthetic data are not stored in the original format of the
    fine res data (tiled netcdfs), but instead as a zarr, because synth does
    not currently support generating netcdfs or splitting by tile
    """

    fine_res_zarrpath = os.path.join(fine_res_dir, "fine_res_budget.zarr")
    with open(str(datadir_module.join("fine_res_budget.json"))) as f:
        fine_res_budget_schema = synth.load(f)
    fine_res_budget_dataset = synth.generate(fine_res_budget_schema)
    fine_res_budget_dataset_1 = fine_res_budget_dataset.assign_coords(
        {"time": ["20160901.000000"]}
    )
    fine_res_budget_dataset_2 = fine_res_budget_dataset.assign_coords(
        {"time": ["20160901.001500"]}
    )
    fine_res_budget_dataset = xr.concat(
        [fine_res_budget_dataset_1, fine_res_budget_dataset_2], dim="time"
    )
    fine_res_budget_dataset.to_zarr(fine_res_zarrpath, consolidated=True)

    return fine_res_zarrpath


@pytest.fixture(scope="module")
def fine_res_dataset_path(datadir_module):

    with tempfile.TemporaryDirectory() as fine_res_dir:
        fine_res_zarrpath = generate_fine_res_dataset(datadir_module, fine_res_dir)
        yield fine_res_zarrpath


@pytest.fixture
def grid_dataset():

    cat = intake.open_catalog("catalog.yml")
    grid = cat["grid/c48"].to_dask()
    grid = grid.drop_vars(names=["y_interface", "y", "x_interface", "x"])
    surface_type = cat["landseamask/c48"].to_dask()
    surface_type = surface_type.drop_vars(names=["y", "x"])

    return grid.merge(surface_type)


@pytest.fixture
def one_step_training_diags_config():
    return {"mapping_function": "open_one_step", "mapping_function_kwargs": {}}


@pytest.fixture
def nudging_training_diags_config():
    return {
        "mapping_function": "open_merged_nudged_full_tendencies",
        "mapping_function_kwargs": {
            "nudging_timescale_hr": 3,
            "open_merged_nudged_kwargs": {
                "rename_vars": {
                    "air_temperature_tendency_due_to_nudging": "dQ1",
                    "specific_humidity_tendency_due_to_nudging": "dQ2",
                }
            },
            "open_checkpoints_kwargs": {
                "checkpoint_files": ("after_dynamics.zarr", "after_physics.zarr")
            },
        },
    }


@pytest.mark.regression
def test_compute_training_diags(
    datadir_module,
    one_step_dataset_path,
    nudging_dataset_path,
    fine_res_dataset_path,
    one_step_training_diags_config,
    nudging_training_diags_config,
    grid_dataset,
):

    data_config_mapping = {
        "one_step_tendencies": (one_step_dataset_path, one_step_training_diags_config),
        "nudging_tendencies": (nudging_dataset_path, nudging_training_diags_config),
        "fine_res_apparent_sources": (fine_res_dataset_path, None),
    }

    # compute the diagnostics for each source

    variable_names = [
        "dQ1",
        "dQ2",
        "pQ1",
        "pQ2",
        "pressure_thickness_of_atmospheric_layer",
    ]

    diagnostic_datasets = {}
    timesteps_per_batch = 1

    for (
        data_source_name,
        (data_source_path, data_source_config),
    ) in data_config_mapping.items():
        if data_source_name != "fine_res_apparent_sources":
            ds_batches_one_step = batches.diagnostic_batches_from_geodata(
                data_source_path,
                variable_names,
                timesteps_per_batch=timesteps_per_batch,
                mapping_function=data_source_config["mapping_function"],
                mapping_kwargs=data_source_config["mapping_function_kwargs"],
            )
            ds_diagnostic = utils.reduce_to_diagnostic(
                ds_batches_one_step, grid_dataset, domains=DOMAINS
            )
        else:
            rename_variables = {
                "delp": "pressure_thickness_of_atmospheric_layer",
                "pfull": "z",
                "grid_xt": "x",
                "grid_yt": "y",
            }
            ds_batches_fine_res = [
                xr.open_zarr(data_source_path).isel(time=0).rename(rename_variables),
                xr.open_zarr(data_source_path).isel(time=1).rename(rename_variables),
            ]
            ds_diagnostic = utils.reduce_to_diagnostic(
                ds_batches_fine_res, grid_dataset, domains=DOMAINS
            )
        diagnostic_datasets[data_source_name] = ds_diagnostic

    diagnostics_all = xr.concat(
        [
            dataset.expand_dims({"data_source": [data_source_name]})
            for data_source_name, dataset in diagnostic_datasets.items()
        ],
        dim="data_source",
    ).load()

    diags_output_schema_raw = synth.read_schema_from_dataset(diagnostics_all)
    # TODO standardize schema encoding in synth to avoid the casting that makes
    # the following line necessary (arrays vs lists)
    diags_output_schema = synth.loads(synth.dumps(diags_output_schema_raw))

    # test against reference

    with open(str(datadir_module.join("training_diags_reference.json"))) as f:
        reference_output_schema = synth.load(f)

    assert reference_output_schema == diags_output_schema


@pytest.fixture(
    params=["one_step_tendencies", "nudging_tendencies", "fine_res_apparent_sources"]
)
def data_source_name(request):
    return request.param


base_config_dict = {
    "model_type": "sklearn_random_forest",
    "hyperparameters": {"max_depth": 4, "n_estimators": 2},
    "input_variables": ["air_temperature", "specific_humidity"],
    "output_variables": ["dQ1", "dQ2"],
    "batch_function": "batches_from_geodata",
}


@pytest.fixture
def one_step_train_config():
    one_step_config_dict = base_config_dict.copy()
    one_step_config_dict.update(
        batch_kwargs={
            "timesteps_per_batch": 1,
            "init_time_dim_name": "initial_time",
            "mapping_function": "open_one_step",
        }
    )
    return train.ModelTrainingConfig(**one_step_config_dict)


@pytest.fixture
def nudging_train_config():
    nudging_config_dict = base_config_dict.copy()
    nudging_config_dict.update(
        batch_kwargs={
            "timesteps_per_batch": 1,
            "init_time_dim_name": "time",
            "mapping_function": "open_merged_nudged",
            "mapping_kwargs": {
                "nudging_timescale_hr": 3,
                "rename_vars": {
                    "air_temperature_tendency_due_to_nudging": "dQ1",
                    "specific_humidity_tendency_due_to_nudging": "dQ2",
                },
            },
        }
    )
    return train.ModelTrainingConfig(**nudging_config_dict)


@pytest.fixture
def fine_res_train_config():
    fine_res_config_dict = base_config_dict.copy()
    fine_res_config_dict.update(
        batch_kwargs={"timesteps_per_batch": 1, "init_time_dim_name": "time"}
    )
    return train.ModelTrainingConfig(**fine_res_config_dict)


@pytest.fixture
def data_source_path_and_train_config(
    data_source_name,
    one_step_dataset_path,
    nudging_dataset_path,
    fine_res_dataset_path,
    one_step_train_config,
    nudging_train_config,
    fine_res_train_config,
):
    data_path_mapping = {
        "one_step_tendencies": one_step_dataset_path,
        "nudging_tendencies": nudging_dataset_path,
        "fine_res_apparent_sources": fine_res_dataset_path,
    }
    data_config_mapping = {
        "one_step_tendencies": one_step_train_config,
        "nudging_tendencies": nudging_train_config,
        "fine_res_apparent_sources": fine_res_train_config,
    }
    return (
        data_source_name,
        data_path_mapping[data_source_name],
        data_config_mapping[data_source_name],
    )


@pytest.mark.regression
def test_sklearn_regression(data_source_path_and_train_config):

    (
        data_source_name,
        data_source_path,
        data_source_train_config,
    ) = data_source_path_and_train_config

    if data_source_name != "fine_res_apparent_sources":
        batched_data = train.load_data_sequence(
            data_source_path, data_source_train_config
        )
    else:
        rename_variables = {
            "delp": "pressure_thickness_of_atmospheric_layer",
            "pfull": "z",
            "grid_xt": "x",
            "grid_yt": "y",
        }
        mapper = {
            "20160901.000000": xr.open_zarr(data_source_path)
            .isel(time=0)
            .rename(rename_variables),
            "20160901.001500": xr.open_zarr(data_source_path)
            .isel(time=1)
            .rename(rename_variables),
        }

        batched_data = batches.batches_from_mapper(
            mapper,
            list(data_source_train_config.input_variables)
            + list(data_source_train_config.output_variables),
            **data_source_train_config.batch_kwargs,
        )

    assert len(batched_data) == 2
    wrapper = train.train_model(batched_data, data_source_train_config)
    assert wrapper.model.n_estimators == 2


def mock_predict_function(feature_data_arrays):
    return sum(feature_data_arrays)


class MockSklearnWrappedModel:
    def __init__(self, input_vars, output_vars):
        self.input_vars_ = input_vars
        self.output_vars_ = output_vars

    def predict(self, ds_stacked, sample_dim=SAMPLE_DIM_NAME):
        ds_pred = xr.Dataset()
        for output_var in self.output_vars_:
            feature_vars = [ds_stacked[var] for var in self.input_vars_]
            mock_prediction = mock_predict_function(feature_vars)
            ds_pred[output_var] = mock_prediction
        return ds_pred


input_vars = ("air_temperature", "specific_humidity")
output_vars = ("dQ1", "dQ2")


@pytest.fixture
def mock_model():
    return MockSklearnWrappedModel(input_vars, output_vars)


@pytest.fixture
def one_step_offline_diags_config():
    config = {
        "variables": [
            "air_temperature",
            "specific_humidity",
            "dQ1",
            "dQ2",
            "pressure_thickness_of_atmospheric_layer",
        ],
        "mapping_function": "open_one_step",
        "batch_kwargs": {
            "timesteps_per_batch": 1,
            "init_time_dim_name": "initial_time",
        },
    }

    return config


@pytest.fixture
def nudging_offline_diags_config():
    config = {
        "variables": [
            "air_temperature",
            "specific_humidity",
            "dQ1",
            "dQ2",
            "pressure_thickness_of_atmospheric_layer",
        ],
        "mapping_function": "open_merged_nudged",
        "mapping_kwargs": {"nudging_timescale_hr": 3, "initial_time_skip_hr": 0},
        "batch_kwargs": {
            "timesteps_per_batch": 2,
            "init_time_dim_name": "initial_time",
        },
    }

    return config


@pytest.fixture
def fine_res_offline_diags_config():
    config = {
        "variables": [
            "air_temperature",
            "specific_humidity",
            "dQ1",
            "dQ2",
            "pressure_thickness_of_atmospheric_layer",
        ],
        "model_mapper_kwargs": {"z_dim": "pfull"},
        "batch_kwargs": {
            "timesteps_per_batch": 2,
            "init_time_dim_name": "initial_time",
        },
    }

    return config


@pytest.fixture
def data_source_path_and_offline_config(
    data_source_name,
    one_step_dataset_path,
    nudging_dataset_path,
    fine_res_dataset_path,
    one_step_offline_diags_config,
    nudging_offline_diags_config,
    fine_res_offline_diags_config,
):
    data_path_mapping = {
        "one_step_tendencies": one_step_dataset_path,
        "nudging_tendencies": nudging_dataset_path,
        "fine_res_apparent_sources": fine_res_dataset_path,
    }
    data_config_mapping = {
        "one_step_tendencies": one_step_offline_diags_config,
        "nudging_tendencies": nudging_offline_diags_config,
        "fine_res_apparent_sources": fine_res_offline_diags_config,
    }
    return (
        data_source_name,
        data_path_mapping[data_source_name],
        data_config_mapping[data_source_name],
    )


@pytest.mark.regression
def test_compute_offline_diags(
    datadir_module, mock_model, data_source_path_and_offline_config, grid_dataset
):

    (
        data_source_name,
        data_source_path,
        data_source_offline_config,
    ) = data_source_path_and_offline_config

    if data_source_name != "fine_res_apparent_sources":
        base_mapping_function = getattr(
            mappers, data_source_offline_config["mapping_function"]
        )
        base_mapper = base_mapping_function(
            data_source_path, **data_source_offline_config.get("mapping_kwargs", {})
        )
        reference_schema_file = "offline_diags_reference.json"
    else:
        rename_variables = {
            "delp": "pressure_thickness_of_atmospheric_layer",
            "pfull": "z",
            "grid_xt": "x",
            "grid_yt": "y",
        }
        base_mapper = {
            "20160901.000000": xr.open_zarr(data_source_path)
            .isel(time=0)
            .rename(rename_variables),
            "20160901.001500": xr.open_zarr(data_source_path)
            .isel(time=1)
            .rename(rename_variables),
        }
        reference_schema_file = "offline_diags_reference_fine_res.json"

    pred_mapper = SklearnPredictionMapper(
        base_mapper,
        mock_model,
        **data_source_offline_config.get("model_mapper_kwargs", {}),
    )

    ds_batches = batches.diagnostic_batches_from_mapper(
        pred_mapper,
        data_source_offline_config["variables"],
        **data_source_offline_config["batch_kwargs"],
    )

    ds_diagnostic = utils.reduce_to_diagnostic(
        ds_batches, grid_dataset, domains=DOMAINS, primary_vars=["dQ1", "dQ2"]
    )

    # TODO standardize schema encoding in synth to avoid the casting that makes
    # the following lines necessary
    output_file = os.path.join(datadir_module, "offline_diags.nc")
    xr.merge([grid_dataset, ds_diagnostic]).to_netcdf(output_file)
    with open(output_file, "rb") as f:
        ds = xr.open_dataset(f).load()
    diags_output_schema_raw = synth.read_schema_from_dataset(ds)
    diags_output_schema = synth.loads(synth.dumps(diags_output_schema_raw))

    # test against reference
    with open(str(datadir_module.join(reference_schema_file))) as f:
        reference_output_schema = synth.load(f)

    assert reference_output_schema == diags_output_schema

    # compute metrics
    metrics = calc_metrics(ds_batches, area=grid_dataset["area"])
    assert metrics
