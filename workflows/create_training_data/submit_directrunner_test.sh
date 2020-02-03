python -m fv3net.pipelines.create_training_data \
 --gcs-input-data-path 2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/one_step_output/C48 \
 --gcs-output-data-dir test-annak/test_dataflow_rundir_pipeline \
 --mask-to-surface-type sea \
  --gcs-bucket gs://vcm-ml-data \
  --runner DirectRunner