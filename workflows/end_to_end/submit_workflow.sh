#!/bin/bash

set -e

config_file=$1

# get arguments for all steps
all_step_args=$(python workflows/end_to_end/get_experiment_args.py $config_file)
echo $all_step_args

# # extra fv3net packages
# cd external/vcm
# python setup.py sdist

# cd external/mappm
# python setup.py sdist

# cd ../../../..

# # Coarsening step
# coarsen_args=$(echo $all_step_args | jq -r .coarsen_restarts)
# workflows/coarsen_restarts/orchestrator_job.sh $coarsen_args

# # One-step jobs
# one_step_args=$(echo $all_step_args | jq -r .one_step_run)
# python workflows/one_step_jobs/orchestrate_submit_jobs.py $one_step_args

# # ML training data creation step
# training_data_args=$(echo $all_step_args | jq -r .create_training_data)
# workflows/create_training_data/orchestrator_job_directrunner_test.sh $training_data_args

# # train ML model step
# train_model_args=$(echo $all_step_args | jq -r .train_sklearn_model)
# workflows/sklearn_regression/orchestrator_train_sklearn.sh $train_model_args

# test ML model step
test_model_args=$(echo $all_step_args | jq -r .test_sklearn_model)
workflows/sklearn_regression/orchestrator_test_sklearn.sh $test_model_args
