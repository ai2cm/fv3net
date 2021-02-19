.. py:module:: runtime.config
.. _configuration:

Configuration
--------------------

The python model is configured using a nested hierarchy of dataclasses_. This
static structure allows the type-checker to find errors related to improper
access of configurations, and serves as a centralization point for
documentation. :class:`UserConfig` is the top-level configuration object. It
refers to several component configurations.

The model reads these dataclasses from a yaml file format like this:

.. literalinclude:: config-example.yaml
    :language: yaml

These entries are translations to dataclasses. Any yaml entries not-described
in the :class:`UserConfig` are ignored when the model loads its
configurations with :func:`get_config`. See :class:`UserConfig` and the
configuration objects it links to for detailed documentation on the available
configuration options.

The run can can be configured to run with the following

#. Machine learning (prognostic)
#. Nudge-to-fine
#. Baseline (no ML or nudging)
#. Nudge-to-obs

Machine learning is done by including a machine learning model path in the
:py:attr:`runtime.steppers.machine_learning.MachineLearningConfig.model_url`.
Nudge-to-fine is done by including a `nudging` config section.


Diagnostics
^^^^^^^^^^^

Default diagnostics are computed and saved to .zarrs depending on whether ML,
nudge-to-fine, nudge-to-obs, or baseline runs are chosen. To save additional
tendencies and storages across physics and nudging/ML time steps, add
:py:attr:`UserConfig.step_tendency_variables` and
:py:attr:`UserConfig.step_storage_variables` entries to specify these
variables. (If not specified these default to ``air_temperature``,
``specific_humidity``, ``eastward_wind``, and ``northward_wind`` for ``tendency``,
and ``specific_humidity`` and ``total_water`` for ``storage``.) Then add an
additional output .zarr which includes among its variables the desired
tendencies and/or path storages of these variables due to physics
(``_due_to_fv3_physics``) and/or ML/nudging (``_due_to_python``). See the example
below of an additional diagnostic file configuration.

.. code-block:: yaml

    namelist:
    coupler_nml:
        hours: 0
        minutes: 60
        seconds: 0
    step_tendency_variables: 
    - air_temperature
    - specific_humidity
    - eastward_wind
    - northward_wind
    - cloud_water_mixing_ratio
    step_storage_variables: 
    - specific_humidity
    - cloud_water_mixing_ratio
    diagnostics:
    - name: step_diags.zarr
        chunks:
        time: 4
        variables:
        - tendency_of_cloud_water_mixing_ratio_due_to_fv3_physics
        - storage_of_specific_humidity_path_due_to_fv3_physics
        - storage_of_cloud_water_mixing_ratio_path_due_to_fv3_physics
        - storage_of_specific_humidity_path_due_to_python

.. _user config:

API
~~~

Top-level
^^^^^^^^^

.. automodule:: runtime.config
    :members: get_config, get_namelist

.. autoclass:: UserConfig

Python "Physics"
^^^^^^^^^^^^^^^^


.. py:module:: runtime.steppers.machine_learning
.. autoclass:: MachineLearningConfig

.. py:module:: runtime.nudging
.. autoclass:: NudgingConfig


Diagnostics
^^^^^^^^^^^

.. py:module:: runtime.diagnostics.manager

.. autoclass:: DiagnosticFileConfig

.. autoclass:: TimeConfig

.. _dataclasses: https://docs.python.org/3/library/dataclasses.html