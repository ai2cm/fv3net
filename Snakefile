from src.data import save_zarr
from snakemake.remote.GS import RemoteProvider as GSRemoteProvider
from os.path import join
GS = GSRemoteProvider(keep_local=True)

# Wildcard values

trained_models = [
    "models/random_forest/default.pkl"
]

timesteps = [
    "20160805.170000"
]

grids = [
    "C384"
]

tiles = [1, 2, 3, 4, 5, 6]
subtiles = list(range(16))

# Remote files
bucket                  = "gs://vcm-ml-data/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data"
c3072_grid_spec_pattern = "gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-diagnostics/grid_spec.tile{tile}.nc.{subtile:04d}"
c96_orographic_data_gcs = "gs://vcm-ml-data/2019-10-01-C96-oro-data.tar.gz"
c384_grid_spec_url_gcs  = "gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-diagnostics/grid_spec_coarse.tile?.nc.*"
grid_and_orography_data = "gs://vcm-ml-data/2019-10-05-coarse-grid-and-orography-data.tar"

# Wildcards for tarball and extracted data
TAR                         = "data/raw/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data/{timestep}.tar"
EXTRACTED                   = "data/extracted/{timestep}/"

# Paths in the extracted tarballs
fv_srf_wnd_prefix           = "data/extracted/{timestep}/{timestep}.fv_srf_wnd_coarse.res"
fv_tracer_prefix            = "data/extracted/{timestep}/{timestep}.fv_tracer_coarse.res"
fv_core_prefix              = "data/extracted/{timestep}/{timestep}.fv_core_coarse.res"

# Grid Specifications
c3072_grid_spec_tiled       = "data/raw/grid_specs/C3072"

# vertical grid
vertical_grid = GS.remote("gs://vcm-ml-data/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data/fv_core.res.nc")

# template directory
template_dir='data/raw/2019-10-02-restart_C48_from_C3072_rundir/restart_C48_from_C3072_nosfc/'

# Orographic Data
oro_and_grid_data = "data/raw/coarse-grid-and-orography-data"
oro_manifest = "assets/coarse-grid-and-orography-data-manifest.txt"
with open(oro_manifest) as f:
    oro_files = [line.strip() for line in f]

grid_spec         = expand("data/raw/coarse-grid-and-orography-data/{{grid}}/{{grid}}_grid.tile{tile:d}.nc", tile=tiles)
oro_data          = expand("data/raw/coarse-grid-and-orography-data/{{grid}}/oro_data.tile{tile:d}.nc", tile=tiles)

# Intermediate steps
coarsened_sfc_data_wildcard = "data/coarsened/{grid}/{timestep}.sfc_data.nc"
restart_dir_wildcard        = "data/restart/{grid}/{timestep}/"
restart_dir_done            = "data/restart/{grid}/{timestep}.done"


c3072_grid_spec = expand(c3072_grid_spec_pattern, tile=tiles, subtile=subtiles)

rule all:
    input: expand(restart_dir_done, timestep=timesteps, grid=grids)

rule prepare_restart_directory:
    input: sfc_data=coarsened_sfc_data_wildcard,
           extracted=EXTRACTED,
           oro_data=oro_data,
           grid_spec=grid_spec,
           vertical_grid=vertical_grid,
	   template_dir = template_dir
    params: srf_wnd=fv_srf_wnd_prefix,
            core=fv_core_prefix,
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
            ('sfc_data', xr.open_dataset(input.sfc_data)),
            ('grid_spec', xr.open_mfdataset(sorted(input.grid_spec), concat_dim='tiles'))
        ]

        make_experiment(
            output[0], tiles_to_save,
            # TODO move these hardcoded strings to the top
            namelist_path='assets/c384_input.nml',
            diag_table='assets/restart_1_step_diag_table',
            template_dir = template_dir,
            oro_paths=input.oro_data,
	    vertical_grid=vertical_grid,
            files_to_copy=[
                ('assets/c384_submit_job.sh', 'submit_job.sh')
            ]
        )

rule run_restart:
    input: restart_dir_wildcard
    output: touch(restart_dir_done)
    run:
        from src.fv3 import run_experiment
        run_experiment(input[0])


def coarsen_factor_from_grid(wildcards):
    target_n = int(wildcards.grid[1:])
    base_n = 3072
    if base_n % target_n != 0:
        raise ValueError("Target grid size must be a factor of 3072")
    return base_n // target_n
    

rule coarsen_sfc_data:
    input: grid=c3072_grid_spec_tiled,
           time=EXTRACTED
    output: coarsened_sfc_data_wildcard
    params: factor=coarsen_factor_from_grid
    shell: """
    python src/data/raw_step_directory_to_restart.py \
      --num-tiles 6 \
      --num-subtiles 16 \
      --method median \
      --factor {params.factor} \
      {wildcards.timestep} {output}
    """

rule download_template_rundir:
    output: directory(template_dir)
    shell:"""
    file=2019-10-02-restart_C48_from_C3072_rundir.tar
    gsutil cp gs://vcm-ml-data/$file .
    mkdir -p data/raw
    tar -xf $file -C data/raw
    rm -f $file
    """
        


rule download_c3072_grid_spec:
    output: directory(c3072_grid_spec_tiled)
    shell: """
    rm -rf {output}
    mkdir -p {output}
    gsutil -m cp {c3072_grid_spec} {output}/
    """

rule download_timestep:
    output: TAR
    shell: """
    gsutil -o 'GSUtil:parallel_thread_count=1' -o 'GSUtil:sliced_object_download_max_component=32' \
          cp  {bucket}/{wildcards.timestep}.tar {output}
    """

rule download_oro_and_grid:
    output: oro_files
    shell:"""
    file=2019-10-05-coarse-grid-and-orography-data.tar 
    gsutil cp gs://vcm-ml-data/$file .
    mkdir -p {oro_and_grid_data}
    tar --strip-components=1 -xf $file -C {oro_and_grid_data}
    rm -f $file
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
