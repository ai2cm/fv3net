"""Zhao Carr specific emulation options

the functions in this submodule know the variable names of the ZC microphysics

"""
from ._typing import FortranState
from typing import Optional
import numpy as np


class Input:
    cloud_water = "cloud_water_mixing_ratio_input"
    humidity = "specific_humidity_input"
    temperature = "air_temperature_input"


class GscondOutput:
    cloud_water = "cloud_water_mixing_ratio_after_gscond"
    humidity = "specific_humidity_after_gscond"
    temperature = "air_temperature_after_gscond"


class PrecpdOutput:
    cloud_water = "cloud_water_mixing_ratio_after_precpd"
    humidity = "specific_humidity_after_precpd"
    temperature = "air_temperature_after_precpd"


def squash_water_water_conserving(cloud, humidity, bound: float):
    cloud_out = np.where(cloud < bound, 0, cloud)
    qv_out = humidity + (cloud - cloud_out)
    return cloud_out, qv_out


def _apply_squash(struct, emulator, cloud_squash: float):
    if struct.cloud_water in emulator:
        if cloud_squash:
            cloud, humidity = squash_water_water_conserving(
                emulator[struct.cloud_water], emulator[struct.humidity], cloud_squash,
            )

            emulator[struct.cloud_water] = cloud
            emulator[struct.humidity] = humidity


def modify_zhao_carr(
    state: FortranState,
    emulator: FortranState,
    cloud_squash: Optional[float],
    gscond_cloud_conservative: bool,
):
    """

    Args:
        cloud_squash: if not None, convert any cloud amounts less than this to humidity
        gscond_cloud_conservative: if True, then compute the cloud after gscond
            from humidity conservation
    """
    # fill in cloud output with conservation if not present
    if gscond_cloud_conservative:
        humidity_change = emulator[GscondOutput.humidity] - state[Input.humidity]
        state[GscondOutput.cloud_water] = state[Input.cloud_water] - humidity_change

    if cloud_squash is not None:
        _apply_squash(GscondOutput, emulator, cloud_squash)
        _apply_squash(PrecpdOutput, emulator, cloud_squash)
    return emulator
