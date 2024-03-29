# Microphysics Emulation

## Quickstart

To submit a job (training or online evaluation) from a yaml:

    end_to_end.py path/to/config.yaml

To reproduce a given run in wandb, run

    reproduce.py ai2cm/microphysics-emulation/id > config.yaml

Then edit the `config.yaml` as you see fit and submit as above.

## Prognostic Evaluation

Be sure to set up your environment file for authentication from the instructions in our [quickstart documentation](https://vulcanclimatemodeling.com/docs/fv3net/quickstarts.html#quickstarts).

Once authentication is configured, you can enter the docker image for development with ``make enter_prognostic_run``.
Within this image, the microphysics projects folder is at ``/fv3net/projects/microphysics``.

From within the microphysics project folder, you can run the prognostic run using::

    python3 scripts/prognostic_run.py

By default this will use the configuration at ``configs/default.yml``.
This can be modified using the ``--config-path`` argument.

Pass `--help` to this script for more information.

## ARGO

If not running locally, the `argo/` subdirectory provides a cloud workflow
and make targets for running offline or online prognostic experiments.  The
workflow includes both online and piggy-backed result evaluations.  See
the README for more details.

## Training data creation

The `create_training/` subdirectory provides make targets for performing the
monthly-initialized training data generation runs as well as gathering
of netcdfs into training/testing GCS buckets after all runs have finished.

### Training a model
The `train/` subdirectory provides an argo workflow that trains with
`fv3fit.train_microphysics` and `scripts/score_training.py`. Scoring
uses the final saved model from training or the last saved epoch at
`config.out_url`.  A script, `run.sh` provides a convenience method
to submit a suite of training experiments using Argo.

To run scoring on a pre-trained model, `score_training.py` accepts
`--model_url <URL>` as an argument to directly reference a model.

To train a model using a GPU node via the ARGO workflow, add the
flag `-p gpu-train=true`.  This will spin up a node w/ a
GPU and available CUDA libraries.


# NeurIPS Workshop Manuscript

The configurations for the best machine learning models are in `configs/models/`. If an argo cluster is available they can be submitted by running:

    end_to_end.py path/to/yaml

To evaluate the machine learning models in a prognostic run you can use:

    end_to_end.py configs/online-evaluation/combined-gscond-precpd.yaml

You'll need to modify the .zhao_carr_emulation...path configurations.

Data and trained models are available here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7109065.svg)](https://doi.org/10.5281/zenodo.7109065)