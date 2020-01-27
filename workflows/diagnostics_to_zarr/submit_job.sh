# submit the job
PIPELINE=fv3net.pipelines.diagnostics_to_zarr
RUNDIR=gs://vcm-ml-data/2019-12-12-baseline-FV3GFS-runs/nudged/C48/test-nudged.935498d5-3528-4e88-b5f4-018e3f54da50/output
NUM_WORKERS=6

python -m $PIPELINE  \
    --rundir $RUNDIR \
    --job_name diagnostics-to-zarr \
    --project vcm-ml \
    --region us-central1 \
    --runner DataflowRunner \
    --setup_file workflows/diagnostics_to_zarr/setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers $NUM_WORKERS \
    --max_num_workers $NUM_WORKERS \
    --disk_size_gb 100 \
    --worker_machine_type n1-highmem-2
