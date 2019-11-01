# Adjust the following to meet job needs
# ------------------
# num_workers: number of workers to start with
# max_num_workers: autoscaling limit
# 
# Running currently uses machine-type the 1-core 3.75 GB memory workers


python -m extractflow \
    --job_name extract-all-tars-$(whoami) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataflowRunner \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 40 \
    --max_num_workers 40 \
    --disk_size_gb 80 \
    --type_check_strictness 'ALL_REQUIRED' \
    --worker_machine_type n1-standard-1
