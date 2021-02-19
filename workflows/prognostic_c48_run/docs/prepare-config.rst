.. _prepare config:

``prepare_config.py``
---------------------
.. warning::

    This API will likely change to make prepare_config and implementation
    detail of ``runfv3 run-native`` and ``runfv3 append``.

The prognostic run provides a command line script ``prepare_config.py`` to
minimize the boilerplate needed by a full yaml configuration described in
:ref:`configuration`. Call ``python prepare_config.py -h`` to see arguments.
This script will update the specified base fv3config with any values
specified in ``user_config`` argument (see :ref:`UserConfig`).

The ``--model_url`` command line argument adds a model to
:py:class:`runtime.steppers.machine_learning.MachineLearningConfig`. It can
be used multiple times to specify multiple models.

A nudge-to-obs run can be executed by including the argument
``--nudge-to-observations URL`` to ``prepare_config.py``; note that
nudge-to-obs is not mutually exclusive with any of the first three options as
it is conducted within the Fortran physics routine.

For example to create an initial value experiment from
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

Diagnostics handling in ``prepare_config``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

    should we delete this section? Do we need or want to document this
    feature beyond the command line argument?

To modify the output frequency of the run's Python diagnostics (which
defaults to every 15 minutes), provide to ``prepare_config.py`` either the
command line argument ``--output-timestamps`` as a file path or
``--output_frequency`` as minutes, but not both. Fortran diagnostics are
configured through the diag_table