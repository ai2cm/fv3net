## C384 Restart Directory to Zarr

**Workflow location:** `workflows/restarts_to_zarr`

This workflow converts a raw directory of restart netCDF files as uploaded from GFDL to a single zarr image. To test this locally with the beam DirectRunner, run:

```
NUM_WORKERS=1

python -m fv3net.pipelines.restarts_to_zarr  \
    --init  \
    --runner Direct \
    --num_workers $NUM_WORKERS \
    --n-steps 1  \
    --url gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files \
    --output gs://vcm-ml-data/testing-noah/big_local.zarr
```

Otherwise, to submit a production job use this one:

```
NUM_WORKERS=256


python -m fv3net.pipelines.restarts_to_zarr  \
    --setup $(pwd)/setup.py \
    --job_name test-$(uuid) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers $NUM_WORKERS \
    --autoscaling_algorithm=NONE \
    --worker_machine_type n1-standard-1 \
    --disk_size_gb 30 \
    --url gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files \
    --output gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr
```

As usual for dataflow jobs, these should be run from the project root.

For convenience, these two scripts are stored at `_run.sh` and `_run_local.sh`.