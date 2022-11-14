import logging
import os

import json
import pytest
import synth
import xarray as xr

from pathlib import Path
from vcm.cubedsphere.coarsen_restarts import (
    coarsen_restarts_on_sigma,
    coarsen_restarts_on_pressure,
    coarsen_restarts_via_blended_method,
)

DIR_NAME = os.path.dirname(__file__)
SCHEMA_PATH = os.path.join(DIR_NAME, "_coarsen_restarts_regression_tests/schemas")
REFERENCE_PATH = os.path.join(DIR_NAME, "_coarsen_restarts_regression_tests/reference")

FACTOR = 2
RANGES = {
    "delp": synth.Range(3, 5),
    "area": synth.Range(0.5, 1),
    "dx": synth.Range(0.5, 1),
    "dy": synth.Range(0.5, 1),
}
RESTART_CATEGORIES = ["fv_core.res", "fv_tracer.res", "fv_srf_wnd.res", "sfc_data"]
REGRESSION_TESTS = {
    "mass-weighted-model-level-with-agrid-winds": (
        coarsen_restarts_on_sigma,
        {"coarsen_agrid_winds": True, "mass_weighted": True},
    ),
    "area-weighted-model-level-without-agrid-winds": (
        coarsen_restarts_on_sigma,
        {"coarsen_agrid_winds": False, "mass_weighted": False},
    ),
    "pressure-level-with-agrid-winds": (
        coarsen_restarts_on_pressure,
        {"coarsen_agrid_winds": True},
    ),
    "pressure-level-without-agrid-winds": (
        coarsen_restarts_on_pressure,
        {"coarsen_agrid_winds": False},
    ),
    "blended-mass-weighted-with-agrid-winds": (
        coarsen_restarts_via_blended_method,
        {"coarsen_agrid_winds": True},
    ),
    "blended-area-weighted-without-agrid-winds": (
        coarsen_restarts_via_blended_method,
        {"coarsen_agrid_winds": False, "mass_weighted": False},
    ),
}


def open_restart_schemas(schema_path):
    schemas = {}
    for category in RESTART_CATEGORIES:
        filename = os.path.join(schema_path, f"{category}-schema.json")
        with open(filename, "r") as file:
            schemas[category] = synth.load(file)
    return schemas


def open_grid_spec_schema(schema_path):
    filename = os.path.join(schema_path, "grid-schema.json")
    with open(filename, "r") as file:
        return synth.load(file)


def generate_synthetic_restart_data(schemas):
    data = {}
    for category, schema in schemas.items():
        data[category] = synth.generate(schema, RANGES)
    return data


def generate_synthetic_grid_spec_data(schema):
    return synth.generate(schema, RANGES)


def to_json(ds, filename, mode="w+"):
    # TODO: move elsewhere in vcm and add tests?
    with open(filename, mode) as file:
        json.dump(ds.load().to_dict(), file)


def from_dict(dictionary):
    # TODO: move elsewhere in vcm and add tests?
    data_vars = {}
    for v, data in dictionary["data_vars"].items():
        data_vars[v] = (data["dims"], data["data"], data["attrs"])
    coords = {}
    for v, data in dictionary["coords"].items():
        coords[v] = (data["dims"], data["data"], data["attrs"])
    attrs = dictionary["attrs"]
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def from_json(filename):
    # TODO: move elsewhere in vcm and add tests?
    with open(filename, "r") as file:
        dictionary = json.load(file)
    return from_dict(dictionary)


def reference_json(root, tag, category):
    return os.path.join(root, f"{tag}-{category}.json")


def open_reference_data(root, tag):
    data = {}
    for category in RESTART_CATEGORIES:
        filename = reference_json(root, tag, category)
        data[category] = from_json(filename)
    return data


def generate_synthetic_data(restart_schemas, grid_schema):
    restart_data = generate_synthetic_restart_data(restart_schemas)
    grid_data = generate_synthetic_grid_spec_data(grid_schema)
    return restart_data, grid_data


@pytest.mark.slow
@pytest.mark.parametrize("tag", REGRESSION_TESTS.keys())
def test_coarsen_restarts(tag):
    restart_schemas = open_restart_schemas(SCHEMA_PATH)
    grid_spec_schema = open_grid_spec_schema(SCHEMA_PATH)
    restart_data, grid_data = generate_synthetic_data(restart_schemas, grid_spec_schema)

    func, kwargs = REGRESSION_TESTS[tag]
    result = func(FACTOR, grid_data, restart_data, **kwargs)
    result = {category: ds.compute() for category, ds in result.items()}
    expected = open_reference_data(REFERENCE_PATH, tag)

    for category in result:
        xr.testing.assert_allclose(result[category], expected[category])


if __name__ == "__main__":
    # Reset the regression test data by running this file. TODO: guard this with
    # some sort of input prompt? E.g. "Confirm that you would like to overwrite
    # the existing coarsen_restarts regression test data [y/N]"
    logging.basicConfig(level=logging.INFO)

    Path(REFERENCE_PATH).mkdir(parents=True, exist_ok=True)
    restart_schemas = open_restart_schemas(SCHEMA_PATH)
    grid_spec_schema = open_grid_spec_schema(SCHEMA_PATH)
    restart_data, grid_data = generate_synthetic_data(restart_schemas, grid_spec_schema)

    for tag, (func, kwargs) in REGRESSION_TESTS.items():
        result = func(FACTOR, grid_data, restart_data, **kwargs)
        for category, ds in result.items():
            path = reference_json(REFERENCE_PATH, tag, category)
            logging.info(f"Writing new regression data to {path}")
            to_json(ds, path)
