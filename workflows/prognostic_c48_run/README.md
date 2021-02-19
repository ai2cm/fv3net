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

Use docker-compose to develop this.

    # slow/correct
    docker-compose build fv3

    # fast/possibly incorrect
    docker pull us.gcr.io/vcm-ml/prognostic_run:latest

    # run tests
    docker-compose run fv3 pytest

    # get a shell
    docker-compose run fv3 bash

  
See the [developer's guide](file:///Users/noah/workspace/VulcanClimateModeling/fv3net/workflows/prognostic_c48_run/docs/_build/html/development.html).