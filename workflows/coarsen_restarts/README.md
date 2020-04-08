Restart timestep coarsening
===========================

This workflow provides a reusuable Dataflow module to coarsen a set of restart
timesteps in a directory on cloud storage.  The workflow performs the coarsening
using pressure-level coarsening defined in `vcm.coarsen`.

```python
fv3net.pipelines.coarsen_restarts

usage: __main__.py [-h] [--add-target-subdir]
                   gcs_src_dir gcs_grid_spec_path source_resolution target_resolution gcs_dst_dir

positional arguments:
  gcs_src_dir          Full GCS path to input data for downloading timesteps.
  gcs_grid_spec_path   Full path with file wildcard 'grid_spec.tile*.nc' to select grid spec files with same resolution as the source data.
  source_resolution    Source data cubed-sphere grid resolution.
  target_resolution    Target coarsening resolution to output.
  gcs_dst_dir          Full GCS path to output coarsened timestep data. Defaults to input pathwith target resolution
                       appended as a directory

optional arguments:
  -h, --help           show this help message and exit
  --add-target-subdir  Add subdirectory with C{target-resolution} to the specified destination directory.

```

See `workflows/coarsen_restarts/submit_job.sh` to see an example of calling this
module for submission of a dataflow job.  Any coarsening job scripts we'd like to document
should go into the `workflows/coarsen_restarts` directory.
**Note**: The pressure level coarsening takes quite a bit of memory
for the C384 coarsening.  Make sure the machine type has enough memory.  If you
see the job failing without any useful errors, it's probably memory related.
