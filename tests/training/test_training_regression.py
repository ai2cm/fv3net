import xarray as xr
import pytest
import yaml
import os
import tempfile
import logging
import diagnostics_utils as utils
import synth
from loaders import mappers, batches, SAMPLE_DIM_NAME
from fv3net.regression.sklearn import train
from fv3net.regression.sklearn._mapper import SklearnPredictionMapper
from offline_ml_diags._metrics import calc_metrics


logger = logging.getLogger(__name__)

DOMAINS = ["land", "sea", "global"]
OUTPUT_NC_NAME = "diagnostics.nc"


@pytest.fixture
def training_diags_reference_schema(test_training_datadir):

    with open(str(test_training_datadir.join("training_diags_reference.json"))) as f:
        reference_output_schema = synth.load(f)
        yield reference_output_schema


@pytest.fixture
def training_data_diags_config():
    path = "./workflows/training_data_diags/training_data_sources_config.yml"
    with open(path, "r") as f:
        yield yaml.safe_load(f)


def get_data_source_training_diags_config(config, data_source_name):
    source_config = config["sources"][data_source_name]
    return {
        "mapping_function": source_config["mapping_function"],
        "mapping_kwargs": source_config.get("mapping_kwargs", {}),
    }


@pytest.mark.regression
def test_compute_training_diags(
    training_diags_reference_schema,
    one_step_dataset_path,
    nudging_dataset_path,
    fine_res_dataset_path,
    training_data_diags_config,
    grid_dataset,
):

    one_step_training_diags_config = get_data_source_training_diags_config(
        training_data_diags_config, "one_step_tendencies"
    )
    nudging_training_diags_config = get_data_source_training_diags_config(
        training_data_diags_config, "nudging_tendencies"
    )

    data_config_mapping = {
        "one_step_tendencies": (one_step_dataset_path, one_step_training_diags_config),
        "nudging_tendencies": (nudging_dataset_path, nudging_training_diags_config),
        "fine_res_apparent_sources": (fine_res_dataset_path, None),
    }

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
                mapping_kwargs=data_source_config["mapping_kwargs"],
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

    assert training_diags_reference_schema == diags_output_schema


def _one_step_train_config():
    path = "./tests/training/train_sklearn_model_onestep_source.yml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return train.ModelTrainingConfig(**config)


@pytest.fixture
def one_step_train_config():
    return _one_step_train_config()


def _nudging_train_config():
    path = "./tests/training/train_sklearn_model_nudged_source.yaml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return train.ModelTrainingConfig(**config)


@pytest.fixture
def nudging_train_config():
    return _nudging_train_config()


def _fine_res_train_config():
    path = "./tests/training/train_sklearn_model_fineres_source.yml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config["batch_kwargs"].pop("mapping_function", None)
    config["batch_kwargs"].pop("mapping_kwargs", None)
    return train.ModelTrainingConfig(**config)


@pytest.fixture
def fine_res_train_config():
    return _fine_res_train_config()


@pytest.fixture
def data_source_train_config(data_source_name):
    if data_source_name == "one_step_tendencies":
        data_source_train_config = _one_step_train_config()
    elif data_source_name == "nudging_tendencies":
        data_source_train_config = _nudging_train_config()
    elif data_source_name == "fine_res_apparent_sources":
        data_source_train_config = _fine_res_train_config()
    else:
        raise NotImplementedError()
    return data_source_train_config


@pytest.fixture
def training_batches(data_source_name, data_source_path, data_source_train_config):

    if data_source_name != "fine_res_apparent_sources":
        batched_data = train.load_data_sequence(
            data_source_path, data_source_train_config
        )
    else:
        # train.load_data_sequence is incompatible with synth's zarrs
        # (it looks for netCDFs); this is a patch until synth supports netCDF
        fine_res_ds = xr.open_zarr(data_source_path)
        mapper = {
            fine_res_ds.time.values[0]: fine_res_ds.isel(time=0),
            fine_res_ds.time.values[1]: fine_res_ds.isel(time=1),
        }

        batched_data = batches.batches_from_mapper(
            mapper,
            list(data_source_train_config.input_variables)
            + list(data_source_train_config.output_variables),
            **data_source_train_config.batch_kwargs,
        )

    return batched_data


@pytest.mark.regression
def test_sklearn_regression(training_batches, data_source_train_config):

    assert len(training_batches) == 2
    wrapper = train.train_model(training_batches, data_source_train_config)
    assert wrapper.model.n_estimators == 2


@pytest.fixture
def offline_diags_reference_schema(data_source_name, test_training_datadir):

    if data_source_name != "fine_res_apparent_sources":
        reference_schema_file = "offline_diags_reference.json"
    else:
        reference_schema_file = "offline_diags_reference_fine_res.json"

    # test against reference
    with open(str(test_training_datadir.join(reference_schema_file))) as f:
        reference_output_schema = synth.load(f)
        yield reference_output_schema


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


def _one_step_offline_diags_config():
    path = "./workflows/offline_ml_diags/tests/test_one_step_config.yml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def one_step_offline_diags_config():
    return _one_step_offline_diags_config()


def _nudging_offline_diags_config():
    path = "./workflows/offline_ml_diags/tests/test_nudging_config.yml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def nudging_offline_diags_config():
    return _nudging_offline_diags_config()


def _fine_res_offline_diags_config():
    path = "./workflows/offline_ml_diags/tests/test_fine_res_config.yml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def fine_res_offline_diags_config():
    return _fine_res_offline_diags_config()


@pytest.fixture
def data_source_offline_config(data_source_name):
    if data_source_name == "one_step_tendencies":
        data_source_offline_config = _one_step_offline_diags_config()
    elif data_source_name == "nudging_tendencies":
        data_source_offline_config = _nudging_offline_diags_config()
    elif data_source_name == "fine_res_apparent_sources":
        data_source_offline_config = _fine_res_offline_diags_config()
    else:
        raise NotImplementedError()
    return data_source_offline_config


@pytest.fixture
def prediction_mapper(
    mock_model, data_source_name, data_source_path, data_source_offline_config
):

    if data_source_name != "fine_res_apparent_sources":
        base_mapping_function = getattr(
            mappers, data_source_offline_config["mapping_function"]
        )
        base_mapper = base_mapping_function(
            data_source_path, **data_source_offline_config.get("mapping_kwargs", {})
        )
    else:
        # open_fine_res_apparent_sources is incompatible with synth's zarrs
        # (it looks for netCDFs); this is a patch until synth supports netCDF
        rename_variables = {
            "delp": "pressure_thickness_of_atmospheric_layer",
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

    prediction_mapper = SklearnPredictionMapper(
        base_mapper,
        mock_model,
        **data_source_offline_config.get("model_mapper_kwargs", {}),
    )

    return prediction_mapper


@pytest.fixture
def diagnostic_batches(prediction_mapper, data_source_offline_config):

    diagnostic_batches = batches.diagnostic_batches_from_mapper(
        prediction_mapper,
        data_source_offline_config["variables"],
        **data_source_offline_config["batch_kwargs"],
    )

    return diagnostic_batches


@pytest.mark.regression
def test_compute_offline_diags(
    offline_diags_reference_schema, diagnostic_batches, grid_dataset
):

    ds_diagnostic = utils.reduce_to_diagnostic(
        diagnostic_batches, grid_dataset, domains=DOMAINS, primary_vars=["dQ1", "dQ2"]
    )

    # TODO standardize schema encoding in synth to avoid the casting that makes
    # the following lines necessary
    with tempfile.TemporaryDirectory() as output_dir:
        output_file = os.path.join(output_dir, "offline_diags.nc")
        xr.merge([grid_dataset, ds_diagnostic]).to_netcdf(output_file)
        with open(output_file, "rb") as f:
            ds = xr.open_dataset(f).load()
    offline_diags_output_schema_raw = synth.read_schema_from_dataset(ds)
    offline_diags_output_schema = synth.loads(
        synth.dumps(offline_diags_output_schema_raw)
    )

    assert offline_diags_reference_schema == offline_diags_output_schema

    # compute metrics
    metrics = calc_metrics(diagnostic_batches, area=grid_dataset["area"])
    assert isinstance(metrics, dict)
    assert len(metrics) == 16
    for metric, metric_dict in metrics.items():
        assert isinstance(metric, str)
        assert isinstance(metric_dict, dict)
        for metric_key, metric_value in metric_dict.items():
            assert isinstance(metric_key, str)
            assert isinstance(metric_value, float)
