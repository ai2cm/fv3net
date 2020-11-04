C384_ATMOS="gs://vcm-ml-raw/2020-05-27-40-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr"
C384_RESTARTS="gs://vcm-ml-experiments/2020-06-02-fine-res/2020-05-27-40-day-X-SHiELD-simulation-C384-restart-files.zarr"
ATMOS_AVG="gs://vcm-ml-raw/2020-05-27-40-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr"
OUTPUT='gs://vcm-ml-experiment/2020-05-27-40day-fine-res-coarsening/'

rand=$(openssl rand -hex 6)

./dataflow.sh submit  \
    -m budget \
    $C384_ATMOS \
    $C384_RESTARTS \
    $ATMOS_AVG \
    $OUTPUT \
    --job_name fine-res-budget-$rand \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --temp_location gs://vcm-ml-scratch/tmp_dataflow \
    --num_workers 64 \
    --autoscaling_algorithm=NONE \
    --worker_machine_type n1-highmem-2
