The Prognostic Run
==================

A machine-learning capable python wrapper of the FV3 dynamical core. It also
supports nudging workflows either to observations or fine resolution
datasets. This the primary tool used to pre-process datasets and evaluate
machine learning models online.

See the [Sphinx
Documentation](https://vulcanclimatemodeling.com/docs/prognostic_c48_run/)
for more details.


Docker quickstart
-----------------

Use entrypoints at the top level of fv3net to develop this, particularly
``make image_test_prognostic_run`` for non-interactive testing, or
``make enter_prognostic_run`` to enter the docker image for development.

See the [developer's guide](https://www.vulcanclimatemodeling.com/docs/prognostic_c48_run/development.html).