python -m fv3net.pipelines.create_training_data \
gs://vcm-ml-data/test-annak/2020-03-27_test_one_step/big.zarr \
gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04 \
gs://vcm-ml-data/test-annak/2020-03-30_test_bigzarr_train_pipeline \
--job_name test-job-create-training-data-annak \
--project vcm-ml \
--region us-central1 \
--runner DataflowRunner \
--temp_location gs://vcm-ml-data/tmp_dataflow \
--num_workers 4 \
--max_num_workers 30 \
--disk_size_gb 30 \
--worker_machine_type n1-standard-1 \
--setup_file ./setup.py \
--extra_package external/vcm/dist/vcm-0.1.1.tar.gz
