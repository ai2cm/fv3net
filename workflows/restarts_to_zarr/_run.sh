NUM_WORKERS=32

cd /home/noahb/fv3net

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
    --disk_size_gb 20 \
    --n-steps 50  \
    --url gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files \
    --output gs://vcm-ml-data/testing-noah/big1.zarr