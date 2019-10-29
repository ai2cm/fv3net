python dataflow/pipeline.py \
    --job_name coarsen-sfc-data-test \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 1 \
    --max_num_workers 4 \
    --disk_size_gb 80  \
    --worker_machine_type n1-standard-4
    #--service_account_email noah-vm-sa@vcm-ml.iam.gserviceaccount.com
