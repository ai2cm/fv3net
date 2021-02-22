.. _execution:

Execution
---------

.. note:: 

    This section assumes you are within an environment with the prognostic
    run installed. See :ref:`development` for instructions on how to set this up.


.. _segmented-run-cli:

Segmented Runs
~~~~~~~~~~~~~~

The prognostic run can be run via the local command line interface: ``runfv3``.

For robustness, the FV3 model is typically run in several short segments,
with restart files saved after each segment. This ensures that the model
still produces outputs if a given segment crashes. For example, assume you
would like to run a full fv3config file (see :ref:`config usage` for instructions), 
with the runfile ``sklearn_runfile.py`` and save the outputs to a google
cloud storage bucket ``gs://bucket/prognostic_run``.

First, save the file ``fv3config.yaml``. Then, the output location for the segmented run needs to be prepared by running::

    runfv3 create gs://bucket/prognostic_run fv3config.yaml  /path/to/sklearn_runfile.py

This sets up a simple directory structure like this (which you can see by running ``gsutil ls gs://bucket/prognostic_run``::

    fv3config.yml
    runfile.py

Now that the folder is setup, you can run a segment locally and save the outputs to the remote location using the ``append`` subcommand::

    runfv3 append gs://bucket/prognostic_run

After being append to at least once the segmented run GCS location contains the following objects::

    # Same as above
    fv3config.yml
    runfile.py

    # Files specific to a given segment
    artifacts/20160801.001500/

    # Post-processed diagnostic outputs
    atmos_8xdaily.zarr/
    atmos_dt_atmos.zarr/
    diags.zarr/
    sfc_dt_atmos.zarr/

The post-processed diagnostic outputs from each segment will automatically be
appended to the previous segment's. All other outputs
(restart files, logging, etc.) will be saved to
``output-url/artifacts/{timestamp}`` where ``timestamp`` corresponds to the start
time of each segment. The duration of each segment is defined by the root level ``fv3config.yml``.

Every subsequent time this command is executed, a new segment starting at
the end of the previous one will be appended. For example, the following for loop will run 5 segments::

    for i in {1..5}
    do
        runfv3 append gs://bucket/prognostic_run
    done


.. note::

    The entire "state" of the segmented run is stored in the
    ``gs://bucket/prognostic_run`` and does not depend on data local to your
    machine. This means that a segmented run can be continued from a machine
    other than the one it was created with. This is useful for debugging e.g.
    failing segmented runs in the integration tests or prognostic run `argo
    workflow <https://github.com/VulcanClimateModeling/fv3net/blob/master/workflows/argo/README.md>`_.
    To debug this run, simply open an prognostic_run development environment
    and run::

        runfv3 append gs://path/to/failing/run

Low-level usage
~~~~~~~~~~~~~~~

Sometimes it is nice to avoid the complexities of a segmented run
(:ref:`segmented-run-cli`) for local development. For this reason, the ``runfv3``
tool provides a command ``run-native`` which you can use like this::

    runfv3 run-native fv3config.yaml path/to/local/rundir path/to/sklearn_runfile.py

This writes the run directory described by the ``fv3config.yaml`` to the
specified local path and executes the model there. The command is used for
example by the tests. 

.. warning::

    ``runfv3 run-native`` produces outputs that aren't post-processed for
    downstream analysis. This subcommand is only intended for debugging purposes.
    Use the ``append`` and ``create`` subcommands to generate analysis-ready
    datasets.
    

Post Processing
~~~~~~~~~~~~~~~

After each segment the outputs are post processed (netCDF's are converted to zarr, and zarr's are rechuncked) using fv3post_.

.. _fv3post: https://github.com/VulcanClimateModeling/fv3net/tree/master/workflows/post_process_run
