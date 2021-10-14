import tensorflow as tf
from typing import Optional

from .normalization import get_norm_class, get_denorm_class


class FieldInput(tf.keras.layers.Layer):
    """Normalize input tensor and subselect along feature if specified"""

    def __init__(
        self,
        *args,
        sample_in: Optional[tf.Tensor] = None,
        normalize: Optional[str] = None,
        selection: Optional[slice] = None,
        **kwargs,
    ):
        """
        Args:
            sample_in: sample for fitting normalization layer
            normalize: normalize layer key to use for inputs
            selection: slice selection taken along the feature dimension
                of the input
            norm_layer:
        """
        super().__init__(*args, **kwargs)

        self._selection = selection

        if normalize is not None:
            self.normalize = get_norm_class(normalize)(name=f"normalized_{self.name}")
            self.normalize.fit(sample_in)
        else:
            self.normalize = tf.keras.layers.Lambda(
                lambda x: x, name=f"passthru_{self.name}"
            )

        self.selection = selection

    def call(self, tensor: tf.Tensor):

        tensor = self.normalize(tensor)
        if self.selection is not None:
            tensor = tensor[..., self.selection]

        return tensor

    def get_config(self):

        config = super().get_config()

        if self._selection is not None:
            serializable_selection = [
                self._selection.start,
                self._selection.stop,
                self._selection.step,
            ]
        else:
            serializable_selection = None

        config.update(
            {"norm_layer": self.normalize, "selection": serializable_selection}
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        norm_layer = config.pop("norm_layer")
        selection = slice(*config.pop("selection"))

        obj = cls(selection=selection, **config)
        obj.normalize = norm_layer

        return obj


class FieldOutput(tf.keras.layers.Layer):
    """Connect linear dense output layer and denormalize"""

    def __init__(
        self,
        nfeatures: int,
        *args,
        sample_out: Optional[tf.Tensor] = None,
        denormalize: Optional[str] = None,
        enforce_positive=False,
        **kwargs,
    ):
        """
        Args:
            nfeatures: size of the output feature dimension
            sample_out: Output sample for variable to fit denormalization layer.
            denormalize: denormalize layer key to use on
                the dense layer output
            enforce_positive: add a ReLU on the final layer output
                call to enforce only positive values
        """
        super().__init__(*args, **kwargs)

        self._enforce_positive = enforce_positive
        self._nfeatures = nfeatures

        self.unscaled = tf.keras.layers.Dense(
            nfeatures, activation="linear", name=f"unscaled_{self.name}"
        )

        if denormalize is not None:
            self.denorm = get_denorm_class(denormalize)(
                name=f"denormalized_{self.name}"
            )
            self.denorm.fit(sample_out)
        else:
            self.denorm = tf.keras.layers.Lambda(
                lambda x: x, name=f"passthru_{self.name}"
            )

        self.relu = tf.keras.layers.ReLU()
        self.use_relu = enforce_positive

    def call(self, tensor):

        tensor = self.unscaled(tensor)
        tensor = self.denorm(tensor)

        if self.use_relu:
            tensor = self.relu(tensor)

        return tensor

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "nfeatures": self._nfeatures,
                "denorm_layer": self.denorm,
                "enforce_positive": self._enforce_positive,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        denorm_layer = config.pop("denorm_layer")

        obj = cls(**config)
        obj.denorm = denorm_layer

        return obj


class IncrementStateLayer(tf.keras.layers.Layer):
    """
    Layer for incrementing states with a tendency tensor

    Attributes:
        dt_sec: timestep delta in seconds
    """

    def __init__(self, dt_sec: int, *args, dtype=tf.float32, **kwargs):

        self._dt_sec_arg = dt_sec
        self.dt_sec = tf.constant(dt_sec, dtype=dtype)
        super().__init__(*args, **kwargs)

    def call(self, initial: tf.Tensor, tendency: tf.Tensor) -> tf.Tensor:
        """
        Increment state with tendency * timestep

        args:
            tensors: Input state field and corresponding tendency tensor to
                increment by.
        """
        return initial + tendency * self.dt_sec

    def get_config(self):
        config = super().get_config()
        config.update({"dt_sec": self._dt_sec_arg})
        return config


class IncrementedFieldOutput(tf.keras.layers.Layer):
    """
    Add input tensor to an output tensor using timestep increment.
    This residual-style architecture is analogous to learning tendency-based updates.
    """

    def __init__(
        self,
        nfeatures: int,
        dt_sec: int,
        *args,
        sample_out: Optional[tf.Tensor] = None,
        denormalize: Optional[str] = None,
        enforce_positive: bool = False,
        **kwargs,
    ):
        """
        Args:
            sample_out: Output sample for variable to set shape
                and fit denormalization layer.
            dt_sec: Timestep length in seconds to use for incrementing
                input state.
            denormalize: denormalize layer key to use on
                the dense layer output
            enforce_positive: add a ReLU on the final layer output
                call to enforce only positive values
        """
        super().__init__(*args, **kwargs)

        self._enforce_positive = enforce_positive
        self._dt_sec = dt_sec
        self._nfeatures = nfeatures

        self.tendency = FieldOutput(
            nfeatures,
            denormalize=denormalize,
            sample_out=sample_out,
            enforce_positive=False,
            name=f"tendency_of_{self.name}",
        )
        self.increment = IncrementStateLayer(dt_sec, name=f"increment_{self.name}")
        self.use_relu = enforce_positive
        self.relu = tf.keras.layers.ReLU()

    def call(self, field_input, network_output):

        tendency = self.tendency(network_output)
        tensor = self.increment(field_input, tendency)

        if self.use_relu:
            tensor = self.relu(tensor)

        return tensor

    def get_tendency_output(self, network_output):
        return self.tendency(network_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "enforce_positive": self._enforce_positive,
                "dt_sec": self._dt_sec,
                "nfeatures": self._nfeatures,
            }
        )

        config["tendency_layer"] = self.tendency.get_config()
        config["increment_layer"] = self.increment.get_config()

        return config

    @classmethod
    def from_config(cls, config):
        tendency = config.pop("tendency_layer")
        increment = config.pop("increment_layer")

        obj = cls(**config)
        obj.tendency = tendency
        obj.increment = increment
