import synth
import os
import pytest
import xarray as xr

from fv3net.pipelines.coarsen_restarts.pipeline import main
from typing import Optional, Set


OUTPUT_CATEGORY_NAMES = {
    "fv_core.res": "fv_core_coarse.res",
    "fv_srf_wnd.res": "fv_srf_wnd_coarse.res",
    "fv_tracer.res": "fv_tracer_coarse.res",
    "sfc_data": "sfc_data_coarse",
}


def save_restarts(restarts, outdir, time):
    for category, tiles in restarts.items():
        for tile, dataset in tiles.items():
            out_category = OUTPUT_CATEGORY_NAMES[category]
            filename = os.path.join(outdir, f"{time}.{out_category}.tile{tile}.nc")
            dataset.to_netcdf(filename)


def _grid_spec(datadir, nx):
    scaling_factor = 384 / nx
    ranges = {
        "dx": synth.Range(20000 * scaling_factor, 28800 * scaling_factor),
        "dy": synth.Range(20000 * scaling_factor, 28800 * scaling_factor),
        "area": synth.Range(
            3.6205933e08 * scaling_factor ** 2, 8.3428736e08 * scaling_factor ** 2
        ),
    }
    path = str(datadir.join("grid_spec.json"))
    with open(path) as f:
        schema = synth.load(f)

    ds = synth.generate(schema, ranges)
    subset = ds.isel(
        grid_xt=slice(nx), grid_yt=slice(nx), grid_x=slice(nx + 1), grid_y=slice(nx + 1)
    )

    return subset


@pytest.fixture(
    params=[False, True], ids=["include_agrid_winds=False", "include_agrid_winds=True"]
)
def include_agrid_winds(request):
    return request.param


@pytest.fixture(
    params=[False, True], ids=["coarsen_agrid_winds=False", "coarsen_agrid_winds=True"]
)
def coarsen_agrid_winds(request):
    return request.param


@pytest.fixture()
def dropped_fv_core_variables(include_agrid_winds, coarsen_agrid_winds):
    if include_agrid_winds and not coarsen_agrid_winds:
        return {"ua", "va"}
    else:
        return None


@pytest.fixture()
def restart_dir(tmpdir, datadir, include_agrid_winds):

    nx = 48

    time = "20160101.000000"
    output = tmpdir.mkdir(time)
    tmpdir.mkdir("grid_spec")
    tmpdir.mkdir("output").mkdir(time)
    restarts = synth.generate_restart_data(
        nx=nx, include_agrid_winds=include_agrid_winds
    )
    save_restarts(restarts, output, time)

    grid_spec = _grid_spec(datadir, nx)
    for i in range(6):
        path = str(tmpdir.join(f"grid_spec/grid_spec.tile{i+1}.nc"))
        grid_spec.to_netcdf(path)

    return tmpdir


@pytest.mark.regression
def test_regression_coarsen_restarts(
    restart_dir, include_agrid_winds, coarsen_agrid_winds, dropped_fv_core_variables
):
    grid_spec_path = str(restart_dir.join("grid_spec"))
    src_path = str(restart_dir)
    in_res = "48"
    out_res = "6"
    time = "20160101.000000"
    dest = str(restart_dir.join("output"))

    args = [src_path, grid_spec_path, in_res, out_res, dest]
    if coarsen_agrid_winds:
        args.append("--coarsen-agrid-winds")

    if coarsen_agrid_winds and not include_agrid_winds:
        with pytest.raises(ValueError, match="'ua' and 'va'"):
            main(args)
    else:
        main(args)
        for destination_category, source_category in OUTPUT_CATEGORY_NAMES.items():
            if "fv_core" in destination_category:
                assert_expected_variables_were_coarsened(
                    source_category,
                    destination_category,
                    time,
                    src_path,
                    dest,
                    dropped_fv_core_variables,
                )
            else:
                assert_expected_variables_were_coarsened(
                    source_category, destination_category, time, src_path, dest
                )


def open_category(category, time, directory):
    files = os.path.join(directory, time, f"{time}.{category}.tile[1-6].nc")
    return xr.open_mfdataset(files, concat_dim=["tile"], combine="nested")


def assert_expected_variables_were_coarsened(
    source_category: str,
    destination_category: str,
    time: str,
    source_dir: str,
    destination_dir: str,
    dropped_variables: Optional[Set] = None,
):
    """Check that expected variables in the original restart files can be found in
    the coarsened restart files."""
    source = open_category(source_category, time, source_dir)
    result = open_category(destination_category, time, destination_dir)

    if dropped_variables is None:
        dropped_variables = set()

    expected_variables = set(source.data_vars.keys()) - dropped_variables
    result_variables = set(result.data_vars.keys())
    assert expected_variables == result_variables
