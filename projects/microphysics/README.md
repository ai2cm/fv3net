# Microphysics Emulation

## Train Models

To train models

    bash -c 'cd projects/microphysics/train && ./run.sh'

from the fv3net root directory.

## Prognostic Evaluation

Setup environment file to authenticate against external services
```
cat << EOF> .env
GOOGLE_CLOUD_PROJECT=vcm-ml
WANDB_API_KEY=<wandb api key>
EOF

# login to google cloud
gcloud auth application-default login
```

Pull docker image

    TAG=latest # replace with the desired git sha
    docker pull us.gcr.io/vcm-ml/prognostic_run:$TAG
    # tag with latest to use with docker-compose
    docker tag us.gcr.io/vcm-ml/prognostic_run:$TAG us.gcr.io/vcm-ml/prognostic_run:latest

Enter the docker image

    docker-compose run --rm fv3

Run the prognostic run

    python3 scripts/prognostic_run.py --duration 1h

Pass `--help` to this script for more information
## All in one prognostic run script


See [this script](scripts/run_all_prognostic.sh) for an example of how to run a
prognostic run and the diagnostics at the same time.


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
