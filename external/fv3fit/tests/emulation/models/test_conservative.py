import tensorflow as tf
from fv3fit.emulation.models import (
    ConservativeWaterModel,
    Names,
    MicrophysicsConfig,
    ArchitectureConfig,
)


def test_ConservativeWaterModel_conserves():

    input_names = Names("qc0", "qv0", "delp")
    output_names = Names("qc1", "qv1", "delp_out")

    class MockModel(tf.keras.Model):
        def call(self, in_):

            ins = [
                input_names.cloud_water,
                input_names.specific_humidity,
                input_names.pressure_thickness,
            ]

            outs = [
                output_names.cloud_water,
                output_names.specific_humidity,
                output_names.pressure_thickness,
            ]
            lookup = dict(zip(ins, outs))
            return [in_[key] for key in sorted(in_)]

    conservative = ConservativeWaterModel(
        model, input_names=input_names, output_names=output_names
    )
    out = conservative(ins_)
