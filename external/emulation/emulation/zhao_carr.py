"""Zhao Carr specific emulation options

the functions in this submodule know the variable names of the ZC microphysics

"""
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


def squash_water_water_conserving(cloud, qv, bound: float):
    cloud_out = np.where(cloud < bound, 0, cloud)
    qv_out = qv + (cloud - cloud_out)
    return qv_out


def modify_zhao_carr(
    state, emulator, cloud_squash: float, gscond_cloud_conservative: bool
):
    """

    Args:
        squash: convert any cloud amounts less than this to humidity
        gscond_cloud_conservative: if True, then compute the cloud after gscond
            from humidity conservation
    """
    # fill in cloud output with conservation if not present
    if gscond_cloud_conservative:
        humidity_change = emulator[GscondOutput.humidity] - state[Input.humidity]
        state[GscondOutput.cloud_water] = state[Input.cloud_water] - humidity_change

    if GscondOutput.cloud_water in emulator:
        if cloud_squash:
            emulator[GscondOutput.cloud_water] = squash_water_water_conserving(
                emulator[GscondOutput.cloud_water],
                emulator[GscondOutput.humidity],
                cloud_squash,
            )

    if PrecpdOutput.humidity in emulator:
        if cloud_squash:
            emulator[PrecpdOutput.cloud_water] = squash_water_water_conserving(
                emulator[PrecpdOutput.cloud_water],
                emulator[PrecpdOutput.humidity],
                cloud_squash,
            )
    return emulator
