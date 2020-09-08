FV3Fit
======

FV3Fit is a library for machine learning workflows used by the Vulcan Climate Modeling group.

* Free software: BSD license


# fv3fit contribution guide - Predictor base class

All model training routines defined in fv3fit should produce a subclass of the Predictor
type defined in `.external/fv3fit/_shared/predictor.py`, which requires definition
of the `input_variables`, `output_variables`, and `sample_dim_name` attributes as well
as the `load` and `predict` methods. This provides a unified public API for users
of this package to make predictions in a diagnostic or prognostic setting. Internally,
the `fit` and `dump` methods, among others, will presumably be defined as part of
training routines, but that is not enforced by the Predictor class. 


### Configuration
- model_type: "sklearn_random_forest" or "DenseModel" are currently supported
- hyperparameters: dict of model hyperparameters
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