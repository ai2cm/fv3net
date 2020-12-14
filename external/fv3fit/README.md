FV3Fit
======

FV3Fit is a library for machine learning workflows used by the Vulcan Climate Modeling group.

* Free software: BSD license


# fv3fit contribution guide - Base classes

fv3fit models should be subclassed from either ``fv3fit.Predictor`` or ``fv3fit.Estimator``. The former defines the interface required by the prognostic run and offline reports to load and predict with a model. The latter defines the methods needed to train and save a model. See the implementations of these classes for more details.

Moreover, to be saved and loaded via the generic `fv3fit.dump` and
`fv3fit.load` functions, a new `Estimator` will need to be decorated with
`fv3fit._shared.io.register`.


### Configuration
- model_type: "sklearn_random_forest" or "DenseModel" are currently supported
- hyperparameters: dict of model hyperparameters; for keras models, the following are supported
    as nested dicts within "hyperparameters":
    - fit_kwargs: dict of arguments to the tf.keras.model.fit() method
    - optimizer: dict of arguments to create an optimizer for use in training keras models,
        e.g., "name" and "learning_rate"
- input_variables: list of variables used as features
- output_variables: list of variables to predict
- additional_variables: optional list of variables that are needed (e.g. pressure thickness is needed for mass scaling)
    but are not features or outputs
- batch_function: function from `fv3fit.batches` that is used to load batched data
- batch_kwargs: kwargs for batch function.
- scaler_type: optional, used to specify scaler for sklearn training. Defaults to standard scaler.
- scaler_kwargs: optional, used to specify kwargs for creating scaler in sklearn training. 

An example configuration for training a model is provided below. 
```
  model_type: sklearn_random_forest
  hyperparameters:
    max_depth: 13
    n_estimators: 1
  input_variables:
    - air_temperature
    - specific_humidity
  output_variables:
    - dQ1
    - dQ2
  additional_variables:  # optional
    - pressure_thickness_of_atmospheric_layer
  batch_function: batches_from_geodata
  batch_kwargs:
    timesteps_per_batch: 1
    mapping_function: open_merged_nudged_full_tendencies
    mapping_kwargs:
      consolidated: True
      open_merged_nudged_kwargs:
        rename_vars:
          air_temperature_tendency_due_to_nudging: dQ1
          specific_humidity_tendency_due_to_nudging: dQ2
  scaler_type: # optional- defaults to standard
    - mass
  scaler_kwargs: # optional
    variable_scale_factors: 1000000
```