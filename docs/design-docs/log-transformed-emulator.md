# Goal

Train a microphysics emulator in log-transformed humidity/cloud water space.

# Background

Microphysics emulators have especially poor predictions of cloud water in polar
regions
([proof]()https://wandb.ai/ai2cm/microphysics-emulation/reports/Shared-panel-21-12-21-10-12-70--VmlldzoxMzY1OTYx)
and the upper troposphere.
The prediction of specific humidity and temperature are significantly better except for the most poleward latitudes.
Log-transforming the cloud water in particular should be helpful.

# Requirements

- data has input fields: specific humidity, cloud water
- loss is evaluated in log space...model predicts and uses log transformed input
- metrics information is logged equivalently to allow clean comparison in w and b
- serialized artifact must predict in original (non-transformed space)
- can use with ConservativeWaterConfig

# Design

## Configuration

Add `transform` option to `variables`. this will be a dataclass
`TransformedVariable` which can be configured like this:

```
...
transform:
  variables:
  - to: log_specific_humidity
    from: specific_humidity
    transform: log
```

## Implementation

We first need a code abstraction defining a "Transform".  Losses and models all
take in and received dicts of tensors so this should too.  These suggests the
following transform abstraction:

```python
TensorDict = Mapping[str, tf.Tensor]

class TensorTransform:
    def forward(self, x: TensorDict):
        """transform inputs into a new prognostic basis (e.g. log transform)"""

    def backward(self, transformed: TensorDict) -> TensorDict:
        """undo the transformation"""

class TransformedLayer(tf.keras.layers.Layer):
    model: tf.keras.layers.Layer
    transform: TensorTransform
    def call(self, x):
        return transform.backward(self.model(transform.forward(x)))
```

I will need to write the following implementations

```python
class PerVariableTransform(Transform):
    # potentially empty
    _fields: List[TransformedVariable]
```

Model-building: hook into `TrainingConfig.build`. Implementation will look like:
```python
transform = get_transform(transform_config.variables)
data_with_transform = transform.forward(data)
model = build_model(config, data_with_transforms)
TransformedLayer(model, transform)
```

Data: hook into  `fv3fit.emulation.data.config.TransformConfig.call`