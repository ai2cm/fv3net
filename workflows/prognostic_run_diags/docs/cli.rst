.. _cli:

Command line interface
======================

prognostic_run_diags
^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: fv3net.diagnostics.prognostic_run.cli
   :func: get_parser
   :prog: prognostic_run_diags

memoized_compute_diagnostics.sh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This script checks for availability of diagnostics and metrics of a given
prognostic run in the diagnostics cache. If not available, they will be computed
and added to the cache. The diagnostics will then be copied to specified
output location::

   memoized_compute_diagnostics.sh runURL output flags

where ``flags`` is provided to ``prognostic_run_diags save``.

The cache is located at ``gs://vcm-ml-archive/prognostic_run_diags``. The key
used to describe a run in the cache is the ``runURL`` with ``gs://`` stripped from the
start of the URL and forward slashes replaced by dashes. For example::

   memoized_compute_diagnostics.sh gs://vcm-ml-experiments/sample-run output flags

will save the diagnostics and metrics for the given run at::

   gs://vcm-ml-archive/prognostic_run_diags/vcm-ml-experiments-sample-run

in addition to saving the diagnostics and metrics to the given ``output`` location.
