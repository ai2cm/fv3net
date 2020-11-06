import logging
import os
import tempfile

import numpy as np
import pytest
import xarray as xr
import yaml

import diagnostics_utils as utils
import synth
from fv3fit import _shared as shared
import fv3fit
from fv3fit.sklearn._train import train_model
from offline_ml_diags._mapper import PredictionMapper
from loaders import SAMPLE_DIM_NAME, batches, mappers
from offline_ml_diags.compute_diags import (
    _average_metrics_dict,
    _compute_diags_over_batches,
)

logger = logging.getLogger(__name__)

DIURNAL_VARS = [
    "column_integrated_dQ1",
    "column_integrated_dQ2",
    "column_integrated_pQ1",
    "column_integrated_pQ2",
    "column_integrated_Q1",
    "column_integrated_Q2",
]
OUTPUT_NC_NAME = "diagnostics.nc"


@pytest.fixture
def training_diags_reference_schema(datadir_module):
    with open(
        os.path.join(str(datadir_module), "training_diags_reference.json"), "r"
    ) as f:
        reference_output_schema = synth.load(f)
        yield reference_output_schema


@pytest.fixture
def training_data_diags_config(datadir_module):
    with open(
        os.path.join(str(datadir_module), "training_data_sources_config.yml"), "r"
    ) as f:
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
    nudging_dataset_path,
    fine_res_dataset_path,
    training_data_diags_config,
    C48_SHiELD_diags_dataset_path,
    grid_dataset,
):
    nudging_config = get_data_source_training_diags_config(
        training_data_diags_config, "nudging_tendencies"
    )
    nudging_config["mapping_kwargs"]["shield_diags_url"] = C48_SHiELD_diags_dataset_path
    nudging_config["data_path"] = nudging_dataset_path
    fine_res_config = get_data_source_training_diags_config(
        training_data_diags_config, "fine_res_apparent_sources"
    )
    fine_res_config["mapping_kwargs"][
        "shield_diags_url"
    ] = C48_SHiELD_diags_dataset_path
    fine_res_config["data_path"] = fine_res_dataset_path

    data_config_mapping = {
        "nudging_tendencies": (nudging_dataset_path, nudging_config),
        "fine_res_apparent_sources": (fine_res_dataset_path, fine_res_config),
    }

    variable_names = [
        "dQ1",
        "dQ2",
        "pQ1",
        "pQ2",
        "net_heating",
        "net_precipitation",
        "pressure_thickness_of_atmospheric_layer",
    ]

    diagnostic_datasets = {}
    for (
        data_source_name,
        (data_source_path, data_source_config),
    ) in data_config_mapping.items():
        ds_batches = batches.diagnostic_batches_from_geodata(
            data_source_path,
            variable_names,
            mapping_function=data_source_config["mapping_function"],
            mapping_kwargs=data_source_config["mapping_kwargs"],
            timesteps_per_batch=1,
            res="c8_random_values"
        )
        ds = xr.concat(ds_batches, dim="time")
        ds = ds.pipe(utils.insert_total_apparent_sources).pipe(
            utils.insert_column_integrated_vars
        )
        ds_diagnostic = utils.reduce_to_diagnostic(
            ds,
            grid_dataset,
            net_precipitation=ds["net_precipitation"].sel(
                derivation="coarsened_SHiELD"
            ),
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


def _nudging_train_config(datadir_module):
    with open(
        os.path.join(str(datadir_module), "train_sklearn_model_nudged_source.yml"), "r"
    ) as f:
        config = yaml.safe_load(f)
    return shared.ModelTrainingConfig("nudging_data_path", **config)


@pytest.fixture
def nudging_train_config(datadir_module):
    return _nudging_train_config(datadir_module)


def _fine_res_train_config(datadir_module):
    with open(
        os.path.join(str(datadir_module), "train_sklearn_model_fineres_source.yml"), "r"
    ) as f:
        config = yaml.safe_load(f)
    return shared.ModelTrainingConfig("fine_res_data_path", **config)


@pytest.fixture
def fine_res_train_config(datadir_module):
    return _fine_res_train_config(datadir_module)


@pytest.fixture
def data_source_train_config(data_source_name, datadir_module):
    if data_source_name == "nudging_tendencies":
        data_source_train_config = _nudging_train_config(datadir_module)
    elif data_source_name == "fine_res_apparent_sources":
        data_source_train_config = _fine_res_train_config(datadir_module)
    else:
        raise NotImplementedError()
    return data_source_train_config


@pytest.fixture
def training_batches(data_source_name, data_source_path, data_source_train_config):
    return shared.load_data_sequence(data_source_path, data_source_train_config)


@pytest.mark.regression
def test_sklearn_regression(training_batches, data_source_train_config):

    assert len(training_batches) == 2
    wrapper = train_model(training_batches, data_source_train_config)
    assert wrapper.model.n_estimators == 2


@pytest.fixture
def offline_diags_reference_schema(datadir_module):
    with open(
        os.path.join(str(datadir_module), "offline_diags_reference.json"), "r"
    ) as f:
        reference_output_schema = synth.load(f)
        yield reference_output_schema


class MockSklearnWrappedModel(fv3fit.Predictor):
    def __init__(self, input_vars, output_vars):
        self.input_variables = input_vars
        self.output_variables = output_vars
        self.sample_dim_name = "sample"

    def predict(self, ds_stacked, sample_dim=SAMPLE_DIM_NAME):
        ds_pred = xr.Dataset()
        for output_var in self.output_variables:
            feature_vars = [ds_stacked[var] for var in self.input_variables]
            mock_prediction = sum(feature_vars)
            ds_pred[output_var] = mock_prediction
        return ds_pred

    def load(self, *args, **kwargs):
        pass


input_vars = ("air_temperature", "specific_humidity")
output_vars = ("dQ1", "dQ2")


@pytest.fixture
def mock_model():
    return MockSklearnWrappedModel(input_vars, output_vars)


@pytest.fixture
def data_source_offline_config(
    data_source_name, datadir_module, C48_SHiELD_diags_dataset_path
):
    if data_source_name == "nudging_tendencies":
        with open(
            os.path.join(str(datadir_module), "train_sklearn_model_nudged_source.yml"),
            "r",
        ) as f:
            config = yaml.safe_load(f)
            config["batch_kwargs"]["mapping_kwargs"][
                "shield_diags_url"
            ] = C48_SHiELD_diags_dataset_path
        return config
    elif data_source_name == "fine_res_apparent_sources":
        with open(
            os.path.join(str(datadir_module), "train_sklearn_model_fineres_source.yml"),
            "r",
        ) as f:
            config = yaml.safe_load(f)
            config["batch_kwargs"]["mapping_kwargs"][
                "shield_diags_url"
            ] = C48_SHiELD_diags_dataset_path
            return config
    else:
        raise NotImplementedError()


@pytest.fixture
def prediction_mapper(
    mock_model, data_source_name, data_source_path, data_source_offline_config
):

    base_mapping_function = getattr(
        mappers, data_source_offline_config["batch_kwargs"]["mapping_function"]
    )
    base_mapper = base_mapping_function(
        data_source_path,
        **data_source_offline_config["batch_kwargs"].get("mapping_kwargs", {}),
    )

    prediction_mapper = PredictionMapper(base_mapper, mock_model)

    return prediction_mapper


timesteps = ["20160801.001500", "20160801.003000"]
variables = [
    "air_temperature",
    "specific_humidity",
    "dQ1",
    "dQ2",
    "pQ1",
    "pQ2",
    "pressure_thickness_of_atmospheric_layer",
    "net_heating",
    "net_precipitation",
]


@pytest.fixture
def diagnostic_batches(prediction_mapper, data_source_offline_config):

    data_source_offline_config["batch_kwargs"]["timesteps"] = timesteps
    data_source_offline_config["variables"] = variables
    del data_source_offline_config["batch_kwargs"]["mapping_function"]
    del data_source_offline_config["batch_kwargs"]["mapping_kwargs"]
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
    ds_diagnostics, ds_diurnal, ds_metrics = _compute_diags_over_batches(
        diagnostic_batches, grid_dataset
    )

    # convert metrics to dict
    metrics = _average_metrics_dict(ds_metrics)

    # TODO standardize schema encoding in synth to avoid the casting that makes
    # the following lines necessary
    with tempfile.TemporaryDirectory() as output_dir:
        output_file = os.path.join(output_dir, "offline_diags.nc")
        xr.merge([grid_dataset, ds_diagnostics], compat="override").to_netcdf(
            output_file
        )
        with open(output_file, "rb") as f:
            ds = xr.open_dataset(f).load()
    offline_diags_output_schema_raw = synth.read_schema_from_dataset(ds)
    offline_diags_output_schema = synth.loads(
        synth.dumps(offline_diags_output_schema_raw)
    )
    for var in set(offline_diags_output_schema.variables):
        assert (
            offline_diags_output_schema.variables[var]
            == offline_diags_reference_schema.variables[var]
        )
    for coord in set(offline_diags_output_schema.coords):
        assert (
            offline_diags_output_schema.coords[coord]
            == offline_diags_reference_schema.coords[coord]
        )

    for var in DIURNAL_VARS:
        assert "local_time_hr" in ds_diurnal[var].dims
        for dim in ds_diurnal[var].dims:
            assert dim in ["local_time_hr", "derivation", "surface_type"]

    assert isinstance(metrics, dict)
    assert len(metrics) == 40
    for metric, metric_entry in metrics.items():
        assert isinstance(metric, str)
        assert isinstance(metric_entry, dict)
        for metric_key, metric_value in metric_entry.items():
            assert isinstance(metric_key, str)
            assert isinstance(metric_value, (float, np.float32))
