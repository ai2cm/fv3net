.. configuration_:

Derived prediction models
=========

The DerivedModel class wraps a trained ML model to include prediction variables that may be derived from the ML prediction. 
If one or more `derived_output_variables` are specified in the `TrainingConfig`, the model will be automatically be wrapped 
as a `DerivedModel` before dumping. Predictions made by a `DerivedModel` will include the `derived_output_variables` in their
predictions.

Models that include derived prediction variables that can also be defined manually, by writing a configuration yaml named 
``derived_model.yaml`` and a ``name`` file to a directory. 


.. code-block:: yaml

    model: gs://vcm-ml-scratch/annak/2021-07-07-nn-no-net-sw/trained_model
    derived_output_variables:
        - net_shortwave_sfc_flux_derived

A ``name`` file can be written using:

.. code-block:: bash

    echo -n derived_model > name

It is important that the created ``name`` file does not have a newline, as can be done with ``echo -n``.

The resulting directory with these two files ``name`` and ``derived_model.yaml`` can be used as a model directory in prognostic run.
