.. _microphysics:

Microphysics Emulation
----------------------

.. module:: fv3fit.train_microphysics

Running the training script can be done via the command-line

.. code-block:: bash

    python -m fv3fit.train_microphysics --config-path <default | /path/to/config>

where "default" loads a default configuration using :py:func:`get_default_config`

Any parameter that exists in the configuration :code:`--config-path` can be updated from the command-line.  Nested configuration updates are achieved using '.'-joined keys.

E.g.,

.. code-block:: bash

    python -m fv3fit.train_microphysics --config-path default \
        --epochs 5 \
        --out_url gs://bucket/new_output \
        --transform.output_variables A B C \
        --model.architecture.name rnn


Programmatically, the :py:class:`TrainConfig` can be instanced from YAML (dictionaries), flattened WandB style stored configs (flat '.'-joined key dictionaries), as well as custom argument sequences.


Important Configurations
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TrainConfig
.. autoclass:: fv3fit.emulation.models.MicrophysicsConfig
.. autoclass:: TransformConfig
.. autoclass:: CustomLoss
