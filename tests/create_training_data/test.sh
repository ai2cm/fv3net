coarsen_diags=gs://vcm-ml-data/testing-noah/2020-04-18/25b5ec1a1b8a9524d2a0211985aa95219747b3c6/coarsen_diagnostics/

python -m fv3net.pipelines.create_training_data \
    gs://vcm-ml-data/test-end-to-end-integration/integration-debug/one_step_run_ \
    $coarsen_diags \
    /etc/config/create_training_data_variable_names.yml \
    /home/jovyan/fv3net/training_times.json \
    gs://vcm-ml-data/test-end-to-end-integration/integration-debug/create_training_data_ \
    --timesteps-per-output-file 1
