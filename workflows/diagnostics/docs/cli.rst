.. _cli:

Command line interface
======================

prognostic_run_diags
^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: fv3net.diagnostics.prognostic_run.cli
   :func: get_parser
   :prog: prognostic_run_diags

artifacts
^^^^^^^^^

.. argparse::
   :module: fv3net.artifacts.query
   :func: get_parser
   :prog: artifacts

memoized_compute_diagnostics.sh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This script checks for availability of pre-computed diagnostics and metrics of a given
prognostic run. If not available, they will be computed and then be copied to specified
output location::

   memoized_compute_diagnostics.sh runURL output flags

where ``flags`` is provided to ``prognostic_run_diags save``. `runURL` must be a GCS url.

The cache for the diagnostics is assumed to be at ``${runURL}_diagnostics`.
For example::

   memoized_compute_diagnostics.sh gs://vcm-ml-experiments/default/2021-05-04/tag/fv3gfs_run output flags

will save the diagnostics and metrics for the given run (if they don't already exist) at::

   gs://vcm-ml-experiments/default/2021-05-04/tag/fv3gfs_run_diagnostics

in addition to saving the diagnostics and metrics to the given ``output`` location.
