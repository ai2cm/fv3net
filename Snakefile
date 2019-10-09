from src.data import save_zarr
from snakemake.remote.GS import RemoteProvider as GSRemoteProvider
from os.path import join
GS = GSRemoteProvider(keep_local=True)

bucket = "gs://vcm-ml-data/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data"
TAR="data/raw/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data/{timestep}.tar"
EXTRACTED="data/extracted/{timestep}/"
c3072_grid_spec_pattern = "gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-diagnostics/grid_spec.tile{tile}.nc.{subtile:04d}"
coarsened_sfc_data_wildcard="data/coarsened/c3072/{timestep}.sfc_data.nc"
restart_dir_wildcard="data/restart/c96/{timestep}/"
fv_srf_wnd_prefix = "data/extracted/{timestep}/{timestep}.fv_srf_wnd_coarse.res"
fv_tracer_prefix = "data/extracted/{timestep}/{timestep}.fv_tracer_coarse.res"
fv_core_prefix = "data/extracted/{timestep}/{timestep}.fv_core_coarse.res"
c96_orographic_data_gcs = "gs://vcm-ml-data/2019-10-01-C96-oro-data.tar.gz"
c384_grid_spec_url_gcs = "gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-diagnostics/grid_spec_coarse.tile?.nc.*"
c384_grid_spec = "data/grid_spec/c384.nc"
c96_orographic_data = "data/oro_data/c96/"

trained_models = [
    "models/random_forest/default.pkl"
]

timesteps = [
"20160805.170000"
]

tiles = [1, 2, 3, 4, 5, 6]
subtiles = list(range(16))

c3072_grid_spec = expand(c3072_grid_spec_pattern, tile=tiles, subtile=subtiles)

rule all:
    input: expand(restart_dir_wildcard, timestep=timesteps)

rule prepare_c96_restart_directory:
    input: sfc_data=coarsened_sfc_data_wildcard,
           extracted=EXTRACTED, oro_data=c96_orographic_data,
           grid_spec=c384_grid_spec
    params: srf_wnd=fv_srf_wnd_prefix, core=fv_core_prefix,
            tracer=fv_tracer_prefix
    output: directory(restart_dir_wildcard)
    run:
        import xarray as xr
        from src.data.cubedsphere import open_cubed_sphere
        from src.fv3 import make_experiment
        import os
        import logging
        logging.basicConfig(level=logging.INFO)

        tiles_to_save = [
            ('fv_tracer.res', open_cubed_sphere(params.tracer)),
            ('fv_core.res', open_cubed_sphere(params.core)),
            ('fv_srf_wnd.res', open_cubed_sphere(params.srf_wnd)),
            ('sfc_data.res', xr.open_dataset(input.sfc_data)),
            ('grid_spec', xr.open_dataset(input.grid_spec))
        ]

        make_experiment(
            output[0], tiles_to_save,
            namelist_path='assets/restart_c48.nml',
            template_dir = 'experiments/2019-10-02-restart_C48_from_C3072_rundir/restart_C48_from_C3072_nosfc/',
            oro_path=input.oro_data
        )


rule coarsen_sfc_data:
    input: grid="data/raw/grid_specs/c3072",
           time=EXTRACTED
    output: coarsened_sfc_data_wildcard
    shell: """
    python src/data/raw_step_directory_to_restart.py \
      --num-tiles 6 \
      --num-subtiles 16 \
      --method median \
      --factor 32 \
      {wildcards.timestep} {output}
    """

rule download_c3072_grid_spec:
    output: directory("data/raw/grid_specs/c3072")
    shell: """
    rm -rf {output}
    mkdir -p {output}
    gsutil -m cp {c3072_grid_spec} {output}/
    """

rule download_c384_grid_spec:
    output: directory("data/raw/grid_specs/c384")
    shell: """
    rm -rf {output}
    mkdir -p {output}
    gsutil -m cp {c384_grid_spec_url_gcs} {output}/
    rename 's/_coarse//' {output}/*.nc.*
    """

rule combine_grid_spec:
    input: "data/raw/grid_specs/c384"
    output: c384_grid_spec
    run:
        from src.data.cubedsphere import open_cubed_sphere
        grid_spec = open_cubed_sphere(input[0] + '/grid_spec')
        grid_spec.to_netcdf(output[0])

rule download_timestep:
    output: TAR
    shell: """
    gsutil -o 'GSUtil:parallel_thread_count=1' -o 'GSUtil:sliced_object_download_max_component=32' \
          cp  {bucket}/{wildcards.timestep}.tar {output}
    """

rule download_c96_orographic_data:
    input: GS.remote(c96_orographic_data_gcs)
    output: directory(c96_orographic_data)
    shell: """
    tar --strip-components 1 -xzf {input}
    mkdir -p {output}
    mv oro_data* {output}
    """

rule extract_timestep:
    output: directory(EXTRACTED)
    input: TAR
    shell: "path=$(pwd)/{input};  mkdir -p {output} && cd {output} && tar xf $path"

rule convert_to_zarr:
    output: directory(save_zarr.output_2d), directory(save_zarr.output_3d)
    shell: "python -m src.data.save_zarr"

rule train_model:
    input: config="configurations/{model_type}/{options}.yaml"
    output: "models/{model_type}/{options}.pkl"
    shell: "python -m src.models.{wildcards.model_type}.train --options {input.config} {output}"
