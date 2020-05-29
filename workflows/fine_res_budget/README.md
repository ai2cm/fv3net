## Fine Resolution derived budget calculation

This dataflow pipline estimates Q1 and Q2 by coarse-graining the C3072 sub-grid-scale physical tendencies and adding the
vertical eddy-fluxes which are resolved in C3072 but not at C48.

### Local usage

To test locally use this:

    python -m budget \
        gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr/ \
        gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr \
        gs://vcm-ml-scratch/noah/2020-05-12/

Parallel usage:

	 python -m budget \
          gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr/ \
          gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr \
          gs://vcm-ml-scratch/noah/2020-05-18 \
          --runner Direct --direct_num_workers 8 --direct_running_mode multi_processing

This produces the following outputs:
```
$ gsutil ls gs://vcm-ml-scratch/noah/2020-05-12/                                                                                                                                                                                                                       (base)
gs://vcm-ml-scratch/noah/2020-05-12/20160801.000730.tile1.nc
gs://vcm-ml-scratch/noah/2020-05-12/20160801.000730.tile2.nc
gs://vcm-ml-scratch/noah/2020-05-12/20160801.000730.tile3.nc
gs://vcm-ml-scratch/noah/2020-05-12//
```


### Remote dataflow usage

Run this from the fv3net root directory:

```
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
```