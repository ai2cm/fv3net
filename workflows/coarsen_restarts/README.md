Restart timestep coarsening
===========================

This workflow provides a reusuable Dataflow module to coarsen a set of restart
timesteps in a directory on cloud storage.  The workflow performs the coarsening
using pressure-level coarsening defined in `vcm.coarsen`.

```python
fv3net.pipelines.coarsen_restarts

usage: __main__.py [-h] --gcs-src-dir GCS_SRC_DIR [--gcs-dst-dir GCS_DST_DIR]
                   --gcs-grid-spec-path GCS_GRID_SPEC_PATH --source-resolution
                   SOURCE_RESOLUTION --target_resolution TARGET_RESOLUTION

optional arguments:
  -h, --help            show this help message and exit
  --gcs-src-dir GCS_SRC_DIR
                        Full GCS path to input data for downloading timesteps.
  --gcs-dst-dir GCS_DST_DIR
                        Full GCS path to output coarsened timestep data.
                        Defaults to input pathwith target resolution appended
                        as a directory
  --gcs-grid-spec-path GCS_GRID_SPEC_PATH
                        Full path with file wildcard 'grid_spec.tile*.nc' to
                        select grid spec files with same resolution as the
                        source data
  --source-resolution SOURCE_RESOLUTION
                        Source data cubed-sphere grid resolution.
  --target_resolution TARGET_RESOLUTION
                        Target coarsening resolution to output
```

See `workflows/coarsen_restarts/submit_job.sh` to see an example of calling this
module for submission of a dataflow job.  Any coarsening job scripts we'd like to document
should go into the `workflows/coarsen_restarts` directory.
**Note**: The pressure level coarsening takes quite a bit of memory
for the C384 coarsening.  Make sure the machine type has enough memory.  If you
see the job failing without any useful errors, it's probably memory related.
