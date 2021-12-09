import unittest
import numpy as np
import emulation.serialize
import tensorflow as tf
import tempfile


class SerializationTest(unittest.TestCase):
    def setUp(self):
        self.data = {"a": tf.ones((1, 10)), "time": tf.constant("20210101T00:00:00")}
        self.serialized = emulation.serialize.serialize_tensor_dict(self.data)
        self.parser = emulation.serialize.get_parser(self.data)

    def test_serialize_tensor_dict(self):
        self.assertIsInstance(self.serialized, bytes)

    def test_parse_example(self):
        records = tf.constant([self.serialized])
        b = self.parser.parse_example(records)
        np.testing.assert_array_equal(b["a"][0], self.data["a"])

    def test_parse_single_example(self):
        record = tf.constant(self.serialized)
        b = self.parser.parse_single_example(record)
        np.testing.assert_array_equal(b["a"], self.data["a"])
        self.assertEqual(b["time"].numpy(), self.data["time"].numpy())

    def test_save_parser(self):
        with tempfile.TemporaryDirectory() as dir_:
            tf.saved_model.save(self.parser, dir_)
            loaded = tf.saved_model.load(dir_)
        self.assertTrue(hasattr(loaded, "parse_single_example"))
        self.assertTrue(hasattr(loaded, "parse_example"))
