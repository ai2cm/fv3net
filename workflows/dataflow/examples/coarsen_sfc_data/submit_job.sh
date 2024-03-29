python -m coarsen_surface_data  \
    --job_name coarsen-sfc-data-testing-$(uuid) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-scratch/tmp_dataflow \
    --num_workers 64 \
    --max_num_workers 128 \
    --disk_size_gb 50 \
    --worker_machine_type n1-standard-2
