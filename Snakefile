from src.data import save_zarr
from snakemake.remote.GS import RemoteProvider as GSRemoteProvider
from os.path import join
from src.data import cubedsphere


GS = GSRemoteProvider(keep_local=True)

# Wildcard values
trained_models = [
    "models/random_forest/default.pkl"
]

timesteps = [
    "20160805.170000"
]

grids = [
    "C48", "C384"
]

tiles = [1, 2, 3, 4, 5, 6]
subtiles = list(range(16))


def raw_restart_filenames(wildcards):
    timestep = wildcards['timestep']
    category = wildcards['category']
    return cubedsphere.all_filenames(join(
        'data/extracted',
        f'{timestep}',
        f'{timestep}.{category}')
    )


def coarsened_restart_filenames(wildcards):
    timestep = wildcards['timestep']
    grid = wildcards['grid']
    category = wildcards['category']
    return [f'data/coarsened/{grid}/{timestep}/{category}.tile{tile}.nc' for
            tile in tiles]


RESTART_CATEGORIES = [
    'fv_core.res',
    'fv_srf_wnd.res',
    'fv_tracer.res',
    'sfc_data'
]

OUTPUT_CATEGORY_NAMES = {
    'fv_core.res': 'fv_core_coarse.res',
    'fv_srf_wnd.res': 'fv_srf_wnd_coarse.res',
    'fv_tracer.res': 'fv_tracer_coarse.res',
    'sfc_data': 'sfc_data'
}


# Remote files
bucket                      = "gs://vcm-ml-data/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data"
c3072_grid_spec_pattern     = "gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-diagnostics/grid_spec.tile{tile}.nc.{subtile:04d}"
grid_and_orography_data     = "gs://vcm-ml-data/2019-10-05-coarse-grid-and-orography-data.tar"
vertical_grid               = GS.remote("gs://vcm-ml-data/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data/fv_core.res.nc")
input_data                  = "gs://vcm-ml-public/2019-09-27-FV3GFS-docker-input-c48-LH-nml/fv3gfs-data-docker_2019-09-27.tar.gz"
image_name                  = "us.gcr.io/vcm-ml/fv3gfs-compiled:latest"

# Local Assets (under version control)
oro_manifest                = "assets/coarse-grid-and-orography-data-manifest.txt"

# Wildcards for tarball and extracted data
TAR                         = "data/raw/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data/{timestep}.tar"
EXTRACTED                   = "data/extracted/{timestep}/"

# Paths in the extracted tarballs
fv_srf_wnd_prefix           = "data/extracted/{timestep}/{timestep}.fv_srf_wnd_coarse.res"
fv_tracer_prefix            = "data/extracted/{timestep}/{timestep}.fv_tracer_coarse.res"
fv_core_prefix              = "data/extracted/{timestep}/{timestep}.fv_core_coarse.res"

# Grid Specifications
c3072_grid_spec_tiled       = "data/raw/grid_specs/C3072"
native_grid_spec            = [GS.remote(f'gs://vcm-ml-data/2019-10-16-C3072-grid-spec/grid_spec.tile{tile:d}.nc')
                               for tile in tiles]

# template directory
template_dir                = 'data/raw/2019-10-02-restart_C48_from_C3072_rundir/restart_C48_from_C3072_nosfc/'

# input directory
input_data_dir              = "data/inputdata/"

# Orographic Data
oro_and_grid_data           = "data/raw/coarse-grid-and-orography-data"
with open(oro_manifest) as f:
    oro_files               = [line.strip() for line in f]

grid_spec                   = expand("data/raw/coarse-grid-and-orography-data/{{grid}}/{{grid}}_grid.tile{tile:d}.nc", tile=tiles)
oro_data                    = expand("data/raw/coarse-grid-and-orography-data/{{grid}}/oro_data.tile{tile:d}.nc", tile=tiles)

# Intermediate steps
coarsened_sfc_data_wildcard = "data/coarsened/{grid}/{timestep}.sfc_data.nc"
fv3_image_pulled_done       = "fv3gfs-compiled.done"
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
           input_data_dir=input_data_dir,
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
            ('grid_spec', xr.open_mfdataset(sorted(input.grid_spec), concat_dim='tile'))
        ]

        make_experiment(
            output[0], tiles_to_save,
            template_dir = template_dir,
            oro_paths=input.oro_data,
	    vertical_grid=vertical_grid,
            files_to_copy=[
                # TODO move these hardcoded strings to the top
                ('assets/c384_submit_job.sh', 'submit_job.sh'),
                ('assets/c384_input.nml', 'input.nml'),
                ('assets/restart_1_step_diag_table', 'diag_table')
            ]
        )
        
rule pull_fv3_image:
    output: touch(fv3_image_pulled_done)
    shell: "docker pull {image_name}"
           
rule run_restart:
    input: restart_dir_wildcard,
           fv3_image_pulled_done
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

rule download_input_data:
    output: directory(input_data_dir)
    shell:"""
    filename=fv3gfs-data-docker_2019-09-27.tar.gz
    gsutil cp {input_data} .
    mkdir -p {input_data_dir}
    tar xzf $filename -C {input_data_dir}
    rm -f $filename
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
    output:
        raw_restart_filenames({'timestep': '{timestep}', 'category': 'fv_core_coarse.res'}),
        raw_restart_filenames({'timestep': '{timestep}', 'category': 'fv_srf_wnd_coarse.res'}),
        raw_restart_filenames({'timestep': '{timestep}', 'category': 'fv_tracer_coarse.res'}),
        raw_restart_filenames({'timestep': '{timestep}', 'category': 'sfc_data'}),
        'data/extracted/{timestep}/{timestep}.coupler.res'
    params:
        target_directory='data/extracted/{timestep}'
    input:
        TAR
    shell:
        "tar -xvf {input} -C {params.target_directory}"

rule convert_to_zarr:
    output: directory(save_zarr.output_2d), directory(save_zarr.output_3d)
    shell: "python -m src.data.save_zarr"

rule train_model:
    input: config="configurations/{model_type}/{options}.yaml"
    output: "models/{model_type}/{options}.pkl"
    shell: "python -m src.models.{wildcards.model_type}.train --options {input.config} {output}"


rule coarsen_grid_spec:
    input:
        native_grid_spec
    output:
        coarsened_grid_spec='data/coarsened/coarsened_grid_spec.nc'
    run:
        import pandas as pd
        import xarray as xr
        
        from src.data.cubedsphere import open_cubed_sphere
        from src.fv3.coarsen import block_edge_sum, block_sum        

        tile = pd.Index(tiles, name='tile')
        native_grid_spec = xr.open_mfdataset(input, concat_dim=tile)

        coarse_dx = block_edge_sum(native_grid_spec.dx, 384, 'grid_xt', 'grid_y', 'x')
        coarse_dy = block_edge_sum(native_grid_spec.dy, 384, 'grid_x', 'grid_yt', 'y')
        coarse_area = block_sum(native_grid_spec.area, 384, 'grid_xt', 'grid_yt')
        result = xr.merge([
            coarse_dx.rename(native_grid_spec.dx.name),
            coarse_dy.rename(native_grid_spec.dy.name),
            coarse_area.rename(native_grid_spec.area.name)
        ])

        result.to_netcdf(output.coarsened_grid_spec)


def sync_dimension_order(a, b):
    for var in a:
        a[var] = a[var].transpose(*b[var].dims)
    return a
        

rule coarsen_restart_category:
    input:
        restart_files=rules.extract_timestep.output,
        coarse_grid_spec=rules.coarsen_grid_spec.output.coarsened_grid_spec,
        native_grid_spec=native_grid_spec
    output:
        coarsened_restart_filenames(
            {'timestep': '{timestep}',
             'grid': '{grid}',
             'category': '{category}'}
        )
    run:
        import pandas as pd
        import xarray as xr

        from src.data.cubedsphere import open_cubed_sphere
        from src.fv3.coarsen import (
            coarse_grain_fv_core,
            coarse_grain_fv_srf_wnd,
            coarse_grain_fv_tracer,
            coarse_grain_sfc_data
        )

        timestep = wildcards['timestep']
        native_category_name = wildcards['category']
        target_resolution = int(wildcards['grid'][1:])

        category = OUTPUT_CATEGORY_NAMES[native_category_name]
        
        grid_spec = xr.open_dataset(input.coarse_grid_spec, chunks={'tile': 1})
        tile = pd.Index(tiles, name='tile')
        native_grid_spec = xr.open_mfdataset(input.native_grid_spec, concat_dim=tile)
        source = open_cubed_sphere(f'data/extracted/{timestep}/{timestep}.{category}')

        if category == 'fv_core_coarse.res':
            coarsened = coarse_grain_fv_core(
                source,
                source.delp,
                grid_spec.area.rename({'grid_xt': 'xaxis_1', 'grid_yt': 'yaxis_2'}),
                grid_spec.dx.rename({'grid_xt': 'xaxis_1', 'grid_y': 'yaxis_1'}),
                grid_spec.dy.rename({'grid_x': 'xaxis_2', 'grid_yt': 'yaxis_2'}), 
                target_resolution
            )
        elif category == 'fv_srf_wnd_coarse.res':
            coarsened = coarse_grain_fv_srf_wnd(
                source, 
                grid_spec.area.rename({'grid_xt': 'xaxis_1', 'grid_yt': 'yaxis_1'}),
                target_resolution
            )
        elif category == 'fv_tracer_coarse.res':
            fv_core = open_cubed_sphere(f'data/extracted/{timestep}/{timestep}.fv_core_coarse.res')
            coarsened = coarse_grain_fv_tracer(
                source,
                fv_core.delp.rename({'yaxis_2': 'yaxis_1'}),
                grid_spec.area.rename({'grid_xt': 'xaxis_1', 'grid_yt': 'yaxis_1'}),
                target_resolution
            )
        elif category == 'sfc_data':
            coarsened = coarse_grain_sfc_data(
                source, 
                native_grid_spec.area.rename({'grid_xt': 'xaxis_1', 'grid_yt': 'yaxis_1'}),
                target_resolution
            )
        else:
            raise ValueError(
                f"Cannot coarse grain files for unknown 'category',"
                "{category}."
            )

        coarsened = sync_dimension_order(coarsened, source)
        for tile, file in zip(tiles, output):
            coarsened.sel(tile=tile).drop('tile').to_netcdf(file)


rule coarsen_all_restart_data:
    input:
        expand(
            ['data/coarsened/{{grid}}/{{timestep}}/{{category}}.tile{tile}.nc'.format(tile=tile) for
             tile in tiles],
            timestep=timesteps,
            grid=grids,
            category=RESTART_CATEGORIES
        )
