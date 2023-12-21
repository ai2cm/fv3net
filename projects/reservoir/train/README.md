# Local reservoir training

For the SST reservoirs local training can be performed on a 16-core VM quite easily.

This folder contains the scripts I used to train the reservoirs locally.

Train all 6 models in parallel using `./train-local.sh`

This training script includes all the naming parameters at the top which will define
the experiment name and trial used for storing models under the
`vcm-ml-experiments/sst-reservoir-training` remote project directory. The wandb
naming also reflects the names defined in this file.

The training will run in the background, but it will populate local `*.log`
files as the training progresses.