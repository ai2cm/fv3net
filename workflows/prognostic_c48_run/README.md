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

Emulator
--------

Edit parameters in run.sh, and then run it.

To run

    docker-compose run fv3 bash run.sh


Monitor the progress via tensorboard at http://localhost:6006
  
See the [developer's guide](https://www.vulcanclimatemodeling.com/docs/prognostic_c48_run/development.html).
