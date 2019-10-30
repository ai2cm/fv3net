python -m coarsen_surface_data  \
    --job_name coarsen-sfc-data-production \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 40 \
    --max_num_workers 80 \
    --disk_size_gb 50 \
    --worker_machine_type n1-standard-2
    #--service_account_email noah-vm-sa@vcm-ml.iam.gserviceaccount.com
