from runtime.segmented_run.prepare_config import HighLevelConfig
from runtime.segmented_run.api import create, append
from runtime.steppers.radiation import RadiationStepper
from runtime.types import State
import cftime
import yaml
import xarray as xr
import numpy as np
import pytest
import dataclasses
import os


@dataclasses.dataclass
class RadiationDiagnostic:
    name: str
    validate: bool = False
    is_flux: bool = True

    @property
    def python_name(self):
        return f"{self.name}_python"


RADIATION_DIAGNOSTICS = [
    RadiationDiagnostic("clear_sky_downward_longwave_flux_at_surface"),
    RadiationDiagnostic("clear_sky_downward_shortwave_flux_at_surface"),
    RadiationDiagnostic("clear_sky_upward_longwave_flux_at_surface"),
    RadiationDiagnostic("clear_sky_upward_shortwave_flux_at_surface"),
    RadiationDiagnostic("clear_sky_upward_longwave_flux_at_top_of_atmosphere"),
    RadiationDiagnostic("clear_sky_upward_shortwave_flux_at_top_of_atmosphere"),
    RadiationDiagnostic("total_sky_downward_longwave_flux_at_surface"),
    RadiationDiagnostic("total_sky_downward_shortwave_flux_at_surface"),
    RadiationDiagnostic("total_sky_upward_longwave_flux_at_surface"),
    RadiationDiagnostic("total_sky_upward_shortwave_flux_at_surface"),
    RadiationDiagnostic(
        "total_sky_downward_shortwave_flux_at_top_of_atmosphere", validate=True,
    ),
    RadiationDiagnostic("total_sky_upward_longwave_flux_at_top_of_atmosphere"),
    RadiationDiagnostic("total_sky_upward_shortwave_flux_at_top_of_atmosphere"),
    RadiationDiagnostic("total_sky_longwave_heating_rate", is_flux=False),
    RadiationDiagnostic("clear_sky_longwave_heating_rate", is_flux=False),
    RadiationDiagnostic("total_sky_shortwave_heating_rate", is_flux=False),
    RadiationDiagnostic("clear_sky_shortwave_heating_rate", is_flux=False),
]

base_config = r"""
base_version: v0.7
initial_conditions: gs://vcm-fv3config/data/initial_conditions/c12_restart_initial_conditions/v1.0
namelist:
  coupler_nml:
    minutes: 30
    current_date:
    - 2016
    - 8
    - 1
    - 0
    - 0
    - 0
  gfdl_cloud_microphysics_nml:
    fast_sat_adj: false
  gfs_physics_nml:
    fhlwr: 1800.0
    fhswr: 1800.0
    hybedmf: true
    satmedmf: false
  fv_core_nml:
    npx: 13
    npy: 13
    npz: 63
fortran_diagnostics:
  - name: sfc_dt_atmos.zarr
    chunks:
      time: 2
    times:
      frequency: 900
      kind: interval
    variables:
    - {module_name: dynamics, field_name: grid_lont, output_name: lon}
    - {module_name: dynamics, field_name: grid_latt, output_name: lat}
    - {module_name: dynamics, field_name: grid_lon, output_name: lonb}
    - {module_name: dynamics, field_name: grid_lat, output_name: latb}
    - {module_name: dynamics, field_name: area, output_name: area}
    - {module_name: gfs_phys, field_name: dusfci, output_name: uflx}
    - {module_name: gfs_phys, field_name: dvsfci, output_name: vflx}
    - {module_name: gfs_phys, field_name: cnvprcpb_ave, output_name: CPRATsfc}
    - {module_name: gfs_phys, field_name: totprcpb_ave, output_name: PRATEsfc}
    - {module_name: gfs_phys, field_name: DSWRF, output_name: DSWRFsfc}
    - {module_name: gfs_phys, field_name: USWRF, output_name: USWRFsfc}
    - {module_name: gfs_phys, field_name: DSWRFtoa, output_name: DSWRFtoa}
    - {module_name: gfs_phys, field_name: USWRFtoa, output_name: USWRFtoa}
    - {module_name: gfs_phys, field_name: ULWRFtoa, output_name: ULWRFtoa}
    - {module_name: gfs_phys, field_name: ULWRF, output_name: ULWRFsfc}
    - {module_name: gfs_phys, field_name: DLWRF, output_name: DLWRFsfc}
    - {module_name: gfs_phys, field_name: lhtfl_ave, output_name: LHTFLsfc}
    - {module_name: gfs_phys, field_name: shtfl_ave, output_name: SHTFLsfc}
"""  # noqa: 501


def get_fv3config():
    config = HighLevelConfig.from_dict(yaml.safe_load(base_config))
    fv3config_dict = config.to_fv3config()
    # can't call normal dump on representation of the data table without this
    fv3config_dict["diag_table"] = fv3config_dict["diag_table"].asdict()
    return fv3config_dict


def radiation_scheme_config():
    config = get_fv3config()
    config["radiation_scheme"] = {"kind": "python"}
    diagnostics = [
        {
            "name": "state_after_timestep.zarr",
            "chunks": {"time": 2},
            "times": {"frequency": 900, "kind": "interval"},
            "variables": [
                diagnostic.name
                for diagnostic in RADIATION_DIAGNOSTICS
                if diagnostic.is_flux
            ],
        }
    ]
    diagnostics.append(
        {
            "name": "radiation_diagnostics.zarr",
            "chunks": {"time": 2},
            "times": {"frequency": 900, "kind": "interval"},
            "variables": [
                diagnostic.python_name for diagnostic in RADIATION_DIAGNOSTICS
            ],
        }
    )
    config["diagnostics"] = diagnostics
    return config


def run_model(config, rundir):
    create(rundir, config)
    append(rundir)


@pytest.fixture(scope="module")
def completed_rundir(tmpdir_factory):
    config = radiation_scheme_config()
    rundir = tmpdir_factory.mktemp("rundir").join("subdir")
    run_model(config, str(rundir))
    return rundir


def get_zarr(rundir, zarrname):
    zarrpath = os.path.join(rundir, zarrname)
    return xr.open_zarr(zarrpath)


def test_radiation_diagnostics_output(completed_rundir):
    ds = get_zarr(completed_rundir, "radiation_diagnostics.zarr")
    for diagnostic in RADIATION_DIAGNOSTICS:
        assert diagnostic.python_name in ds.data_vars


def test_radiation_diagnostics_validate(completed_rundir):
    rtol = 1.0e-7
    python_radiation = get_zarr(completed_rundir, "radiation_diagnostics.zarr")
    fortran_radiation = get_zarr(completed_rundir, "state_after_timestep.zarr")
    to_validate = [
        diagnostic for diagnostic in RADIATION_DIAGNOSTICS if diagnostic.validate
    ]
    for diagnostic in to_validate:
        python_radiation_diagnostic = python_radiation[diagnostic.python_name]
        fortran_radiation_diagnostic = fortran_radiation[diagnostic.name]
        try:
            xr.testing.assert_allclose(
                python_radiation_diagnostic, fortran_radiation_diagnostic, rtol=rtol
            )
        except AssertionError as err:
            raise AssertionError(
                f"Port failed to validate at relative tolerance {rtol}"
                f" for flux {diagnostic.name}."
            ) from err


class MockRadiation:

    input_variables = ["x", "y"]

    def __init__(self):
        pass

    def __call__(self, time, state: State):
        return {"mock_rad_flux": state["x"] + state["y"]}


class MockInputGenerator:
    def __init__(self, output_variables):
        self.output_variables = output_variables

    def __call__(self, time, state: State):
        state_updates: State = {var: state[var] + 1.0 for var in self.output_variables}
        return {}, {}, state_updates


def get_data_array():
    data = np.random.random([10, 1])
    dims = ["lat", "lon"]
    return xr.DataArray(data, dims=dims)


@pytest.mark.parametrize(
    ["generated_names", "offset"],
    [
        pytest.param(["x"], 1.0, id="add_to_one"),
        pytest.param(["x", "y"], 2.0, id="add_to_both"),
        pytest.param(["z"], 0.0, id="add_to_none"),
    ],
)
def test_input_generator_changes_fluxes(generated_names, offset):

    radiation = MockRadiation()
    input_generator = MockInputGenerator(generated_names)

    stepper = RadiationStepper(radiation, input_generator)
    state: State = {"x": get_data_array(), "y": get_data_array(), "z": get_data_array()}
    time = cftime.DatetimeJulian(2016, 8, 1, 0, 0, 0)
    _, diags, _ = stepper(time, state)

    expected = state["x"] + state["y"] + offset
    xr.testing.assert_allclose(diags["mock_rad_flux"], expected)
