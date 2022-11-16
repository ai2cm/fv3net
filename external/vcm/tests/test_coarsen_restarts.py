import argparse
import glob
import logging
import os

import pytest
import xarray as xr

from pathlib import Path
from vcm.cubedsphere.coarsen_restarts import (
    coarsen_restarts_on_sigma,
    coarsen_restarts_on_pressure,
    coarsen_restarts_via_blended_method,
)
from vcm.xarray_loaders import open_json, to_json

synth == pytest.importorskip("synth")  # noqa: F821

DIR_NAME = os.path.dirname(__file__)
SCHEMA_PATH = os.path.join(DIR_NAME, "_coarsen_restarts_regression_tests/schemas")
REFERENCE_PATH = os.path.join(DIR_NAME, "_coarsen_restarts_regression_tests/reference")

FACTOR = 2
RANGES = {
    "delp": synth.Range(3, 5),  # noqa: F821
    "area": synth.Range(0.5, 1),  # noqa: F821
    "dx": synth.Range(0.5, 1),  # noqa: F821
    "dy": synth.Range(0.5, 1),  # noqa: F821
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
            schemas[category] = synth.load(file)  # noqa: F821
    return schemas


def open_grid_spec_schema(schema_path):
    filename = os.path.join(schema_path, "grid-schema.json")
    with open(filename, "r") as file:
        return synth.load(file)  # noqa: F821


def generate_synthetic_restart_data(schemas):
    data = {}
    for category, schema in schemas.items():
        data[category] = synth.generate(schema, RANGES).compute()  # noqa: F821
    return data


def generate_synthetic_grid_spec_data(schema):
    return synth.generate(schema, RANGES).compute()  # noqa: F821


def reference_json(root, tag, category):
    return os.path.join(root, f"{tag}-{category}.json")


def open_reference_data(root, tag):
    data = {}
    for category in RESTART_CATEGORIES:
        filename = reference_json(root, tag, category)
        data[category] = open_json(filename)
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
    result = {category: ds for category, ds in result.items()}

    # To reset the reference data, run this module as a script:
    # python test_coarsen_restarts.py --overwrite
    # Note we cannot write these as checksum based regression tests, because
    # mappm does not produce bit-for-bit reproducible results on different
    # platforms.
    expected = open_reference_data(REFERENCE_PATH, tag)

    for category in result:
        xr.testing.assert_allclose(result[category], expected[category])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Reset test_coarsen_restarts regression test data"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite regression test data if it exists",
    )
    args = parser.parse_args()

    Path(REFERENCE_PATH).mkdir(parents=True, exist_ok=True)
    existing_reference_files = glob.glob(os.path.join(REFERENCE_PATH, "*.json"))
    if existing_reference_files and not args.overwrite:
        raise ValueError(
            f"Reference files already exist. If you would like to overwite the "
            f"existing reference files pass the '--overwrite' argument."
        )

    restart_schemas = open_restart_schemas(SCHEMA_PATH)
    grid_spec_schema = open_grid_spec_schema(SCHEMA_PATH)
    restart_data, grid_data = generate_synthetic_data(restart_schemas, grid_spec_schema)

    for tag, (func, kwargs) in REGRESSION_TESTS.items():
        result = func(FACTOR, grid_data, restart_data, **kwargs)
        for category, ds in result.items():
            path = reference_json(REFERENCE_PATH, tag, category)
            logging.info(f"Writing new regression data to {path}")
            to_json(ds, path)
