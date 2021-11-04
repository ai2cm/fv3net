.. _config usage:

CLI Configuration Preparation
-----------------------------

The prognostic run can can be configured to run with the following

#. :ref:`Baseline (no ML or nudging) <baseline>`
#. :ref:`Nudge-to-fine <nudge to fine>`
#. :ref:`Nudge-to-obs <nudge to obs>`
#. :ref:`Machine learning (prognostic) <ml config>`

The prognostic run provides a command line script ```prepare-config``  to
minimize the boilerplate required to configure a run. This script allows
specifying changes over the "default" configurations stored `here <https://github.com/VulcanClimateModeling/fv3net/tree/master/external/fv3kube/fv3kube/base_yamls>`_.


.. _baseline:

Baseline Run
~~~~~~~~~~~~

To configure a simple baseline run, save the following to a file ``minimal.yaml``

.. literalinclude:: prognostic_config.yml
    :language: yaml

This file contains a subset of options described by fv3config_. This file can
contain both fv3config_ settings like ``namelist``, as well as the python
runtime configurations described in :ref:`configuration-api`. To generate a
"full configuration" usable by fv3config_ and the python runtime
configurations, run the following::

    prepare-config \
        minimal.yaml \
        gs://vcm-ml-code-testing-data/c48-restarts-for-e2e \
        20160801.001500 \
        > fv3config.yaml


The output file ``fv3config.yaml`` (which is to long to include in these
docs) is now compatible with ``fv3config.write_run_directory`` or
:ref:`execution`.

.. _nudge to fine:

Nudge to fine
~~~~~~~~~~~~~

A nudged-to-fine run can be configured by setting the
:py:attr:`runtime.config.UserConfig.nudging` configuration option. This can
be done by adding the following section to the ``minimal.yaml`` file::

    nudging:
        restarts_path: gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts
        timescale_hours:
            air_temperature: 3
            specific_humidity: 3
            x_wind: 3
            y_wind: 3
            pressure_thickness_of_atmospheric_layer: 3

Notice how these configurations correspond with
:py:attr:`runtime.config.UserConfig.nudging`. Refer to those docs as a
reference.

.. _nudge to obs:

Nudge to obs
~~~~~~~~~~~~

Obervational nudging is implemented in the Fortran model. It is activated by
setting the namelist parameter ``fv_core_nml.nudge`` to ``True``. The nudging
is configured through the ``fv_nwp_nudge_nml`` namelist. For convenience, a
base YAML (``v0.6``) is provided which provides useful defaults nudge to obs
runs. The ``gfs_analysis_data`` section defines the location and naming of the
reference analysis data. Here is an example ``minimal.yaml``:

.. literalinclude:: nudge_to_obs_config.yml
    :language: yaml

.. note::

    Nudge-to-obs is not mutually exclusive with any of the first three
    options as it is conducted within the Fortran physics routine.

.. _ml config:

Machine learning
~~~~~~~~~~~~~~~~

A machine learning run can be configured in two ways. The first is by
specifying a path to a fv3fit_ model in
:py:attr:`runtime.config.UserConfig.scikit_learn.model`. This can be done
by adding the following to the ``minimal.yaml`` example::

    scikit_learn:
        model: ["path/to/model"]


For convenient scripting, the ``--model_url`` command line argument adds a
model to :py:class:`runtime.steppers.machine_learning.MachineLearningConfig`.
It can be used multiple times to specify multiple models. For example::

    prepare-config \
        minimal.yaml \
        gs://vcm-ml-code-testing-data/c48-restarts-for-e2e \
        20160801.001500 \
        --model_url path/to/model
        --model_url path/to_another/model
        > fv3config.yaml
 
Diagnostics
~~~~~~~~~~~

Python diagnostics
^^^^^^^^^^^^^^^^^^

To save custom diagnostics from the python wrapper, provide a ``diagnostics`` section.
To save additional tendencies and storages across physics and nudging/ML time steps,
include variables named like ``tendency_of_{variable}_due_to_{step_name}`` or 
``storage_of_{variable}_path_due_to_{step_name}`` where ``variable`` is the name
of a state variable and ``step_name`` can be either ``fv3_physics``, ``dynamics``
or ``python`` (i.e. ML or nudging).

Note that the diagnostic output named ``state_after_timestep.zarr`` is a special case;
it can only be used to save variables that have getters in the wrapper.

This example configures a run with stepwise tendency outputs for several
variables. These tendencies are averaged online over 3 hour intervals before
being saved.

.. code-block:: yaml

    diagnostics:
    - name: step_diags.zarr
      chunks:
        time: 4
      times:
        kind: interval-average
        frequency: 10800  # 3 hours = 10800 seconds
      variables:
        - tendency_of_cloud_water_mixing_ratio_due_to_fv3_physics
        - storage_of_specific_humidity_path_due_to_fv3_physics
        - storage_of_cloud_water_mixing_ratio_path_due_to_fv3_physics
        - storage_of_specific_humidity_path_due_to_python


Fortran diagnostics
^^^^^^^^^^^^^^^^^^^

Diagnostics to be output by the Fortran model are specified in the
:py:attr:`UserConfig.fortran_diagnostics` section. This section will be converted
to the Fortran ``diag_table`` representation of diagnostics (see fv3config_ docs).


Chunking
^^^^^^^^

The desired chunking can be specified for each diagnostic file to be output. 

.. warning::

    Segmented runs have specific requirements for chunks. See 
    :ref:`segmented-run-cli` for details.


.. _fv3config: https://fv3config.readthedocs.io/en/latest/
.. _fv3fit: https://vulcanclimatemodeling.com/docs/fv3fit/


Python Configuration Preparation
--------------------------------

.. currentmodule:: runtime.segmented_run.prepare_config

We also offer a python API equivalent to the prepare config script. The input to
``prepare-config`` is represented by :py:class:`HighLevelConfig`. For example, here
is how you initialize a run from a directory of timesteps.

.. doctest::

    >>> from runtime.segmented_run.prepare_config import HighLevelConfig, InitialCondition
    >>> config = HighLevelConfig(
    ...     base_version="v0.5",
    ...     initial_conditions=InitialCondition(
    ...         base_url="gs://base", timestep="20160101.000000")
    ... )
    >>> fv3config_dict_ = config.to_fv3config()
    >>> fv3config_dict_["initial_conditions"][0]
    {'source_location': 'gs://base/20160101.000000', 'source_name': '20160101.000000.fv_core.res.tile1.nc', 'target_location': 'INPUT', 'target_name': 'fv_core.res.tile1.nc', 'copy_method': 'copy'}

At this point, you can use ``fv3config_dict_`` with any fv3config utilities. For instance::

    import fv3config
    fv3config.write_run_directory(fv3config_dict_, "rundir")


