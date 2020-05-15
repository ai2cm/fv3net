## Fine Resolution derived budget calculation

This dataflow pipline estimates Q1 and Q2 by coarse-graining the C3072 sub-grid-scale physical tendencies and adding the
vertical eddy-fluxes which are resolved in C3072 but not at C48.

### Local usage

Execute this from the current directory:

    python -m src \
        gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr/ \
        gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr \
        gs://vcm-ml-scratch/noah/2020-05-12/


This produces the following outputs:
```
$ gsutil ls gs://vcm-ml-scratch/noah/2020-05-12/                                                                                                                                                                                                                       (base)
gs://vcm-ml-scratch/noah/2020-05-12/20160801.000730.tile1.nc
gs://vcm-ml-scratch/noah/2020-05-12/20160801.000730.tile2.nc
gs://vcm-ml-scratch/noah/2020-05-12/20160801.000730.tile3.nc
gs://vcm-ml-scratch/noah/2020-05-12//
```


### Proposed Google Dataflow  usage (not working)

Remote example:

    rand=$(openssl rand -hex 6)

    python -m src \
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

