C384_ATMOS="gs://vcm-ml-raw/2020-05-27-40-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr"
C384_RESTARTS="gs://vcm-ml-experiments/2020-06-02-fine-res/2020-05-27-40-day-X-SHiELD-simulation-C384-restart-files.zarr"
OUTPUT="gs://vcm-ml-experiments/default/2021-12-16/2020-05-27-40-day-X-SHiELD-simulation-v2/fine-res-budget.zarr"

cd ../dataflow

./dataflow.sh submit  \
    -m budget \
    $C384_ATMOS \
    $C384_RESTARTS \
    gs://vcm-ml-raw/2020-05-27-40-day-X-SHiELD-simulation-C384-diagnostics/gfsphysics_15min_coarse.zarr/ \
    gs://vcm-ml-raw/2020-11-10-C3072-to-C384-exposed-area.zarr \
    $OUTPUT \
    --job_name find-res-budget-$(openssl rand -hex 6) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --temp_location gs://vcm-ml-scratch/tmp_dataflow \
    --autoscaling_algorithm=NONE \
    --num_workers 64 \
    --worker_machine_type n1-highmem-2
