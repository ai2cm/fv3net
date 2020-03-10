python -m fv3net.pipelines.create_training_data \
gs://vcm-ml-data/2020-02-28-X-SHiELD-2019-12-02-deep-and-mp-off \
gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04 \
gs://vcm-ml-data/experiments-2020-03/deep-conv-mp-off/train-test-data \
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
--extra_package external/vcm/dist/vcm-0.1.0.tar.gz
