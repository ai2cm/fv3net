.. configuration_:

Ensembles
=========

Ensemble models can be defined manually, by writing a configuration yaml named ``ensemble_model.yaml`` and a ``name`` file to a directory.

.. code-block:: yaml

    models:  # one or more model output directories
    - gs://vcm-ml-experiments/2021-01-26-c3072-nn/l2/tq-seed-0/trained_model
    - gs://vcm-ml-experiments/2021-01-26-c3072-nn/l2/tq-seed-1/trained_model
    - gs://vcm-ml-experiments/2021-01-26-c3072-nn/l2/tq-seed-2/trained_model
    - gs://vcm-ml-experiments/2021-01-26-c3072-nn/l2/tq-seed-3/trained_model
    - gs://vcm-ml-experiments/2021-01-26-c3072-nn/l2/tq-seed-4/trained_model
    reduction: median  # can be "median" or "mean"

A ``name`` file can be written using:

.. code-block:: bash

    echo -n ensemble > name

It is important that the created ``name`` file does not have a newline, as can be done with ``echo -n``.

The resulting directory with these two files ``name`` and ``ensemble_model.yaml`` can be used as a model directory in prognostic run or offline diagnostic configurations.
