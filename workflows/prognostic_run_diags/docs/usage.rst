.. _usage:

Usage
=====

An argo workflow template is provided that can be used to create a report
from a set of runs on cloud resources. See ``fv3net/workflows/argo/README.md``.

Individual steps can also be run locally using the ``prognostic_run_diags``
utility that is installed by this package. Assuming a prognostic run
has saved appropriate outputs to ``gs://bucket/prognostic-run``,
diagnostics and metrics can be computed with:

.. code-block:: bash

   prognostic_run_diags save gs://bucket/prognostic-run diags.nc
   prognostic_run_diags metrics diags.nc > metrics.json

Movies of the prognostic run can be saved to a given directory with:

.. code-block:: bash

   prognostic_run_diags movie gs://bucket/prognostic-run /tmp/movie_stills
   stitch_movie_stills.sh /tmp/movie_stills output_directory

Once the diagnostics and metrics (and optionally movies) for two or more runs
have been computed and saved in the following structure:

::

    report_data
    ├── baseline_run
    │   ├── column_heating_moistening.mp4
    │   ├── diags.nc
    │   └── metrics.json
    └── prognostic_run
        ├── column_heating_moistening.mp4
        ├── diags.nc
        └── metrics.json

The report can be generated with

.. code-block:: bash

   prognostic_run_diags report report_data output_report_location

See :ref:`cli` for details about the options available for the above commands.
