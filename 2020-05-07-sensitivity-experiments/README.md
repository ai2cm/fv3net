## Sensitivity experiments

The experiment directories here explore the effect of feature set, random seed, and sample size on the random forest's offline and prognostic skill. The baseline for comparison is the clouds-off configuration, with parameters as specified in `vcm-workflow-control/2020-05-07-sensitivity-experiments/base/fv3net_v0.2.1_base/train_sklearn_model.yml`. For more information refer to this dropbox paper doc: https://paper.dropbox.com/doc/2020-05-05-Experiments-feature-set-hyperparameter-exploration-in-random-forest-training--A0NcSGenCsBEU7D0wIFR~jZXAg-aH89tkEdAqQVSQMsodosF

This configuration contains two subdirectories. The step to create training data should be run first. After it has completed, the experiments can be submitted in parallel. 

The makefile in this directory contains the commands to run the jobs as well as check the configurations beforehand.
- `make check_configs` will check that the kustomize configurations are generated without error.
- `make create_training_data` will submit the job that runs the one-step and trainign data workflows in sequence.
- `make run_experiments` will submit all the experiments within the `1_experiments` directory.
- `kubectl apply -k <experiment directory>` will submit a single job.

