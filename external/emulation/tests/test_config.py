from emulation._emulate.microphysics import TimeMask
from emulation.config import (
    EmulationConfig,
    ModelConfig,
    StorageConfig,
    _load_nml,
    _get_timestep,
    _get_storage_hook,
    get_hooks,
)
import datetime


def test_EmulationConfig_from_dict():
    seconds = 60
    month = 2
    config = EmulationConfig.from_dict(
        {
            "model": {
                "path": "some-path",
                "online_schedule": {
                    "period": seconds,
                    "initial_time": datetime.datetime(2000, month, 1),
                },
            }
        }
    )
    assert config.model.online_schedule.period == datetime.timedelta(seconds=seconds)
    assert config.model.online_schedule.initial_time.month == month


def test_ModelConfig_no_interval():
    config = ModelConfig(path="")
    assert len(list(config._build_masks())) == 0


def test_ModelConfig_with_interval():
    def schedule(time):
        return 1.0

    config = ModelConfig(path="", online_schedule=schedule)
    time_schedule = [
        mask for mask in config._build_masks() if isinstance(mask, TimeMask)
    ][0]
    assert time_schedule.schedule == schedule


def test__get_timestep(dummy_rundir):
    namelist = _load_nml()
    timestep = _get_timestep(namelist)

    assert timestep == 900


def test__load_nml(dummy_rundir):

    namelist = _load_nml()
    assert namelist["coupler_nml"]["hours"] == 1


def test__get_storage_hook(dummy_rundir):
    config = StorageConfig()
    hook = _get_storage_hook(config)
    assert hook


def test_get_hooks(dummy_rundir):
    gscond, model, storage = get_hooks()
    assert storage
    assert model
    assert gscond
