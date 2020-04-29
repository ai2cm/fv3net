coarsen_diags=gs://vcm-ml-data/testing-noah/2020-04-18/25b5ec1a1b8a9524d2a0211985aa95219747b3c6/coarsen_diagnostics/
OUTPUT=gs://vcm-ml-data/testing-noah/$(git rev-parse HEAD)/training_data

python -m fv3net.pipelines.create_training_data \
    gs://vcm-ml-data/test-end-to-end-integration/integration-debug/one_step_run_ \
    $coarsen_diags \
    times.json \
    $OUTPUT
