.. py:module:: runtime.config
.. _configuration:

Python Configuration
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


.. _prepare-config: 

Configuration Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~

The prognostic run provides a command line script ``prepare_config.py`` to
minimize the boilerplate needed by a full yaml configuration described in
:ref:`configuration`. For example to create an initial value experiment from
this minimal configuration:

.. literalinclude:: prognostic_config.yml
    :language: yaml


you can run::

    python3 prepare_config.py \
        minimal.yaml \
        gs://vcm-ml-code-testing-data/c48-restarts-for-e2e \
        20160801.001500 \
        > fv3config.yaml

The second two arguments describe the time and GCS location of the initial
data. The configuration is printed to the standard output, so the command
above pipes to the file fv3config.yaml.


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