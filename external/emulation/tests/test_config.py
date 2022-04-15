from emulation._emulate.microphysics import TimeMask
from emulation.config import EmulationConfig, ModelConfig
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
    assert len(list(config._build_masks())) == 1


def test_ModelConfig_with_interval():
    def schedule(time):
        return 1.0

    config = ModelConfig(path="", online_schedule=schedule)
    time_schedule = [
        mask for mask in config._build_masks() if isinstance(mask, TimeMask)
    ][0]
    assert time_schedule.schedule == schedule
