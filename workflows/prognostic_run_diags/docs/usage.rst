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

   prognostic_run_diags movie gs://bucket/prognostic-run output_directory

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

To make a report from runs that are not in a combined folder like
``report_data`` above, you can use the alternative command::

   prognostic_run_diags report-from-urls \
      -o <output_path> \
      path/to/baseline_run_diags \
      path/to/prognostic_run_diags

Or you may encode the paths to URLs in a JSON and use the
`prognostic_run_diags report-from-json` CLI.

See :ref:`cli` for details about the options available for the above commands.
