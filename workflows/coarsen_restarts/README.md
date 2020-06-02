Restart timestep coarsening
===========================

This workflow provides a reusuable Dataflow module to coarsen a set of restart
timesteps in a directory on cloud storage.  The workflow performs the coarsening
using pressure-level coarsening defined in `vcm.coarsen`.

To submit via dataflow run the following commands from the fv3net root:

```
#!/bin/sh

GCS_SRC="gs://vcm-ml-data/2019-12-02-40-day-X-SHiELD-simulation-C384-restart-files"
GCS_GRIDSPEC="gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy"
SRC_RESOLUTION=384
TARGET_RESOLUTION=48
GCS_DST="gs://vcm-ml-scratch/noah/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts"

time=$(openssl rand -hex 6)

./dataflow.sh submit -m fv3net.pipelines.coarsen_restarts\
    $GCS_SRC \
    $GCS_GRIDSPEC \
    $SRC_RESOLUTION \
    $TARGET_RESOLUTION \
    $GCS_DST \
    --runner DataflowRunner \
    --job_name coarsen-restarts-$time \
    --autoscaling_algorithm=NONE \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 128 \
    --max_num_workers 128 \
    --disk_size_gb 50 \
    --worker_machine_type n1-highmem-2 \
```
