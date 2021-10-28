.. _microphysics

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
.. autoclass:: MicrophysicsConfig
.. autoclass:: TransformConfig
.. autoclass:: CustomLoss
.. autoclass:: StandardLoss


Example :py:class:`TrainConfig` YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    batch_size: 128
    epochs: 4
    loss:
    _fitted: false
    loss_variables:
    - air_temperature_output
    - specific_humidity_output
    - cloud_water_mixing_ratio_output
    - total_precipitation
    metric_variables:
    - tendency_of_air_temperature_due_to_microphysics
    - tendency_of_specific_humidity_due_to_microphysics
    - tendency_of_cloud_water_mixing_ratio_due_to_microphysics
    normalization: mean_std
    optimizer:
        kwargs:
        learning_rate: 0.0001
        name: Adam
    weights:
        air_temperature_output: 50000.0
        cloud_water_mixing_ratio_output: 1.0
        specific_humidity_output: 50000.0
        total_precipitation: 0.04
    model:
    architecture:
        kwargs: {}
        name: linear
    direct_out_variables:
    - cloud_water_mixing_ratio_output
    - total_precipitation
    enforce_positive: true
    input_variables:
    - air_temperature_input
    - specific_humidity_input
    - cloud_water_mixing_ratio_input
    - pressure_thickness_of_atmospheric_layer
    normalize_key: mean_std
    residual_out_variables:
        air_temperature_output: air_temperature_input
        specific_humidity_output: specific_humidity_input
    selection_map:
        air_temperature_input:
        start: null
        step: null
        stop: -10
        cloud_water_mixing_ratio_input:
        start: null
        step: null
        stop: -10
        pressure_thickness_of_atmospheric_layer:
        start: null
        step: null
        stop: -10
        specific_humidity_input:
        start: null
        step: null
        stop: -10
    tendency_outputs:
        air_temperature_output: tendency_of_air_temperature_due_to_microphysics
        specific_humidity_output: tendency_of_specific_humidity_due_to_microphysics
    timestep_increment_sec: 900
    nfiles: 80
    nfiles_valid: 80
    out_url: gs://vcm-ml-scratch/andrep/test-train-emulation
    shuffle_buffer_size: 100000
    test_url: gs://vcm-ml-experiments/microphysics-emu-data/2021-07-29/validation_netcdfs
    train_url: gs://vcm-ml-experiments/microphysics-emu-data/2021-07-29/training_netcdfs
    transform:
    antarctic_only: false
    derived_microphys_timestep: 900
    input_variables:
    - air_temperature_input
    - specific_humidity_input
    - cloud_water_mixing_ratio_input
    - pressure_thickness_of_atmospheric_layer
    output_variables:
    - cloud_water_mixing_ratio_output
    - total_precipitation
    - air_temperature_output
    - specific_humidity_output
    - tendency_of_air_temperature_due_to_microphysics
    - tendency_of_specific_humidity_due_to_microphysics
    use_tensors: true
    vertical_subselections: null
    valid_freq: 1
    verbose: 2
    wandb:
    entity: ai2cm
    job_type: training
    wandb_project: scratch-project
