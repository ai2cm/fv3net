import pathlib
import fv3fit
from fv3fit.keras._models.shared.pure_keras import PureKerasDictPredictor
import tensorflow as tf
import pytest
import xarray


def test_PureKerasDictPredictor_dump_load(tmp_path: pathlib.Path):
    in_ = tf.keras.Input(shape=[1], name="a")
    out = tf.keras.layers.Lambda(lambda x: x, name="out")(in_)
    dict_model = tf.keras.Model(inputs=in_, outputs=out)
    predictor = PureKerasDictPredictor(dict_model)
    fv3fit.dump(predictor, tmp_path.as_posix())
    loaded = fv3fit.load(tmp_path.as_posix())
    assert predictor.input_variables == loaded.input_variables
    assert predictor.output_variables == loaded.output_variables


def test_PureKerasDictPredictor_predict():
    # random number
    r = 0.3432134898
    in_ = tf.keras.Input(shape=[1], name="a")
    out = tf.keras.layers.Lambda(lambda x: x * r, name="out")(in_)
    dict_model = tf.keras.Model(inputs=in_, outputs=out)
    predictor = PureKerasDictPredictor(dict_model)
    in_xr = xarray.Dataset({"a": (["blah", "z"], [[1]])})
    out = predictor.predict(in_xr)
    assert float(out["out"][0]) == pytest.approx(r)
