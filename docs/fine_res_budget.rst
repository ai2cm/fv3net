Execute these from the root

Local example::

    python -m fv3net.pipelines.fine_res_budget \
    gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr/ \
    gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr \
    gs://vcm-ml-scratch/noah/2020-05-12/



Remote example::

    rand=$(openssl rand -hex 6)

    python -m fv3net.pipelines.fine_res_budget \
        gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr/ \
        gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr \
        gs://vcm-ml-scratch/noah/2020-05-12/ \
        --setup $(pwd)/setup.py \
        --job_name fine-res-budget-$rand \
        --project vcm-ml \
        --region us-central1 \
        --runner DataFlow \
        --temp_location gs://vcm-ml-data/tmp_dataflow \
        --num_workers 128 \
        --autoscaling_algorithm=NONE \
        --worker_machine_type n1-standard-2 \
        --disk_size_gb 30

