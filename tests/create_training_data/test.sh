python -m fv3net.pipelines.create_training_data \
    gs://vcm-ml-data/test-end-to-end-integration/integration-debug/one_step_run_ \
    gs://vcm-ml-data/test-end-to-end-integration/integration-debug/coarsen_diagnostics_ \
    /etc/config/create_training_data_variable_names.yml \
    /home/jovyan/fv3net/training_times.json \
    gs://vcm-ml-data/test-end-to-end-integration/integration-debug/create_training_data_ \
    --timesteps-per-output-file 1
