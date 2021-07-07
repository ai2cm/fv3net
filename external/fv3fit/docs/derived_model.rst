.. configuration_:

Derived prediction models
=========

The DerivedModel class wraps a trained ML model to include prediction variables that may be derived from the ML prediction. 
Models that include "derived" prediction variables that can be defined manually, by writing a configuration yaml named 
``derived_model.yaml`` and a ``name`` file to a directory. 


.. code-block:: yaml

    model: gs://vcm-ml-scratch/annak/2021-07-07-nn-no-net-sw/trained_model
    additional_input_variables:
        - surface_diffused_shortwave_albedo
    derived_output_variables:
        - net_downward_shortwave_sfc_flux_derived

A ``name`` file can be written using:

.. code-block:: bash

    echo -n derived_model > name

It is important that the created ``name`` file does not have a newline, as can be done with ``echo -n``.

The resulting directory with these two files ``name`` and ``derived_model.yaml`` can be used as a model directory in prognostic run.
Offline diagnostics can be computed for a DerivedModel if the base ML model's model training configuration yaml in the derived_model directory;
however, only the base ML model's outputs will be included in the report, so you should just use the base model directly in the offline diags.
