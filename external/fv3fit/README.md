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
