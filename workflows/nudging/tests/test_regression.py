import pytest
import yaml
import subprocess
from pathlib import Path

# need to check if fv3gfs exists in a subprocess, importing fv3gfs into this module
# causes tests to fail. Not sure why.
# See https://github.com/VulcanClimateModeling/fv3gfs-python/issues/79
# - noah
FV3GFS_INSTALLED = subprocess.call(["python", "-c", "import fv3gfs"]) == 0
with_fv3gfs = pytest.mark.skipif(not FV3GFS_INSTALLED, reason="fv3gfs not installed")

PREP_CONFIG_PY = Path(__file__).parent.joinpath("prepare_config.py").as_posix()
RUNFILE_PY = Path(__file__).parent.joinpath("runfile.py").as_posix()

native_data_path = "/inputdata/fv3config-cache/gs/vcm-fv3config/vcm-fv3config/"
default_config = f"""
base_version: v0.4
forcing: {native_data_path}/data/base_forcing/v1.1
initial_conditions: {native_data_path}/data/initial_conditions/c12_restart_initial_conditions/v1.0
nudging:
  restarts_path: {native_data_path}/data/initial_conditions/c12_restart_initial_conditions/v1.0
  timescale_hours:
    air_temperature: 3
    specific_humidity: 3
    x_wind: 3
    y_wind: 3
  output_times:
    - "20160801.003000"
namelist:
  coupler_nml:
    calendar: julian
    current_date:
    - 2016
    - 8
    - 1
    - 0
    - 15
    - 0
    days: 0
    hours: 6
    minutes: 0
    months: 0
    seconds: 0
  fv_core_nml:
    do_sat_adj: false
  gfdl_cloud_microphysics_nml:
    fast_sat_adj: false
"""


@pytest.fixture
def nudge_config(tmpdir):

    config_file = tmpdir.join("nudging_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(default_config, f)

    result = subprocess.run(["python", PREP_CONFIG_PY, config_file])

    return yaml.safe_load(result.stdout)


@pytest.mark.regression
def prepare_configuration(nudge_config):
    pass
