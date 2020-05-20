rand=$(openssl rand -hex 6)

    ./dataflow.sh submit  \
        -m budget \
        gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr/ \
        gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr \
        gs://vcm-ml-scratch/noah/2020-05-19/ \
        --job_name fine-res-budget-$rand \
        --project vcm-ml \
        --region us-central1 \
        --runner DataFlow \
        --temp_location gs://vcm-ml-data/tmp_dataflow \
        --num_workers 64 \
        --autoscaling_algorithm=NONE \
        --worker_machine_type n1-highmem-2

