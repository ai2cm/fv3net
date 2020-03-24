python -m fv3net.pipelines.create_training_data \
gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/one_step_output/C48 \
gs://vcm-ml-data/2019-12-05-40-day-X-SHiELD-simulation-C384-diagnostics/C48_gfsphysics_15min_coarse.zarr \
gs://vcm-ml-data/test-annak/2020-02-05_train_data_pipeline/ \
--job_name test-job-create-training-data-brianh \
--project vcm-ml \
--region us-central1 \
--runner DataflowRunner \
--temp_location gs://vcm-ml-data/tmp_dataflow \
--num_workers 4 \
--max_num_workers 30 \
--disk_size_gb 30 \
--worker_machine_type n1-standard-1 \
--setup_file ./setup.py \
--extra_package external/vcm/dist/vcm-0.1.0.tar.gz
