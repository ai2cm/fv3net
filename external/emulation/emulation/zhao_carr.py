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


def squash_water_water_conserving(cloud, humidity, bound: float):
    cloud_out = np.where(cloud < bound, 0, cloud)
    qv_out = humidity + (cloud - cloud_out)
    return cloud_out, qv_out


def _apply_squash(struct, output_state, cloud_squash: float):
    out = {**output_state}
    if struct.cloud_water in output_state:
        cloud, humidity = squash_water_water_conserving(
            output_state[struct.cloud_water],
            output_state[struct.humidity],
            cloud_squash,
        )
        out[struct.cloud_water] = cloud
        out[struct.humidity] = humidity
    return out


def infer_gscond_cloud_from_conservation(state, emulator):
    out = {**emulator}
    humidity_change = emulator[GscondOutput.humidity] - state[Input.humidity]
    out[GscondOutput.cloud_water] = state[Input.cloud_water] - humidity_change
    return out


def squash_gscond(state, emulator, cloud_squash):
    return _apply_squash(GscondOutput, emulator, cloud_squash)


def squash_precpd(state, emulator, cloud_squash):
    return _apply_squash(PrecpdOutput, emulator, cloud_squash)
