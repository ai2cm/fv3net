from src.data import save_zarr
from snakemake.remote.GS import RemoteProvider as GSRemoteProvider
from os.path import join
from src.data import cubedsphere

GS = GSRemoteProvider()

# Wildcard values
trained_models = [
    "models/random_forest/default.pkl"
]

timesteps = [
    "20160805.170000"
]

ORIGINAL_COARSE_RESOLUTION = 384

grids = [
    "C48"
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
            
def coarsened_sfc_filename(wildcards):
    timestep = wildcards['timestep']
    grid = wildcards['grid']
    #category = wildcards['category']
    return f"gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/coarsened/{grid}/{timestep}.sfc_data.nc"


ORIGINAL_RESOLUTIONS = {
    'fv_core.res': 384,
    'fv_tracer.res': 384,
    'fv_srf_wnd.res': 384,
    'sfc_data': 3072,
    'grid_spec': 3072
}

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

# Remote outputs
restart_uploaded            = "gs://vcm-ml-data/2019-10-22-restart-workflow/restart/{grid}/{timestep}/"
restart_uploaded_status     = "workflow-status/restart_{grid}_{timestep}.done"

# Local Assets (under version control)
oro_manifest                = "assets/coarse-grid-and-orography-data-manifest.txt"

# Wildcards for tarball and extracted data
TAR                         = "data/raw/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data/{timestep}.tar"
EXTRACTED                   = "data/extracted/{timestep}/"

extracted = [
    raw_restart_filenames({'timestep': '{timestep}', 'category': 'fv_core_coarse.res'}),
    raw_restart_filenames({'timestep': '{timestep}', 'category': 'fv_srf_wnd_coarse.res'}),
    raw_restart_filenames({'timestep': '{timestep}', 'category': 'fv_tracer_coarse.res'}),
    raw_restart_filenames({'timestep': '{timestep}', 'category': 'sfc_data'}),
    'data/extracted/{timestep}/{timestep}.coupler.res'
]
extraction_directory = 'data/extracted/{timestep}'
extraction_directory_root = 'data/extracted'


# Paths in the extracted tarballs
fv_srf_wnd_prefix           = "data/extracted/{timestep}/{timestep}.fv_srf_wnd_coarse.res"
fv_tracer_prefix            = "data/extracted/{timestep}/{timestep}.fv_tracer_coarse.res"
fv_core_prefix              = "data/extracted/{timestep}/{timestep}.fv_core_coarse.res"

# Grid Specifications
c3072_grid_spec_tiled       = "data/raw/grid_specs/C3072"
native_grid_spec            = [GS.remote(f'gs://vcm-ml-data/2019-10-16-C3072-grid-spec/grid_spec.tile{tile:d}.nc')
                               for tile in tiles]
coarsened_grid_spec = 'data/coarsened/coarsened_grid_spec.nc'

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

coarsened_sfc_data_wildcard = 
coarsened_restart_filenames_wildcard = coarsened_restart_filenames(
    {'timestep': '{timestep}', 'grid': '{grid}', 'category': '{category}'}
)
srf_wnd = coarsened_restart_filenames(
    {'timestep': '{timestep}',
     'grid': '{grid}',
     'category': 'fv_srf_wnd.res'}
)
core = coarsened_restart_filenames(
    {'timestep': '{timestep}',
     'grid': '{grid}',
     'category': 'fv_core.res'}
)
tracer = coarsened_restart_filenames(
    {'timestep': '{timestep}',
     'grid': '{grid}',
     'category': 'fv_tracer.res'}
)
#sfc_data = coarsened_restart_filenames(
#    {'timestep': '{timestep}',
#     'grid': '{grid}',
#     'category': 'sfc_data'}
#)
sfc_data = coarsened_sfc_filename(
    {'timestep' : '{timestep}',
    'grid' : {'grid'}
    }
)
coupler = 'data/extracted/{timestep}/{timestep}.coupler.res'
all_coarsened_restart_files = expand(
    ['data/coarsened/{{grid}}/{{timestep}}/{{category}}.tile{tile}.nc'.format(tile=tile) for
     tile in tiles],
    timestep=timesteps,
    grid=grids,
    category=RESTART_CATEGORIES
)


restart_dir_wildcard        = "data/restart/{grid}/{timestep}/"
restart_dir_done            = "data/restart/{grid}/{timestep}.done"

c3072_grid_spec = expand(c3072_grid_spec_pattern, tile=tiles, subtile=subtiles)


rule all:
    input: GS.remote(expand(restart_uploaded_status, timestep=timesteps, grid=grids))


rule prepare_restart_directory:
    input:
        oro_data=oro_data,
        grid_spec=grid_spec,
        vertical_grid=vertical_grid,
        input_data_dir=input_data_dir,
	template_dir=template_dir,
        srf_wnd=srf_wnd,
        core=core,
        tracer=tracer,
        sfc_data=sfc_data,
        coupler=coupler
    output:
        restart_dir=directory(restart_dir_wildcard)
    run:
        import os
        import logging
        
        import xarray as xr
        
        from datetime import datetime
        
        from src.data.cubedsphere import open_cubed_sphere
        from src.fv3 import make_experiment
        
        logging.basicConfig(level=logging.INFO)

        grid = wildcards['grid']
        
        tiles_to_save = [
            ('fv_tracer.res', xr.open_mfdataset(input.tracer, concat_dim='tile')),
            ('fv_core.res', xr.open_mfdataset(input.core, concat_dim='tile')),
            ('fv_srf_wnd.res', xr.open_mfdataset(input.srf_wnd, concat_dim='tile')),
            # TODO should surface data and other input data have the same 6-tile format?
            ('sfc_data', xr.open_dataset(input.sfc_data),
            ('grid_spec', xr.open_mfdataset(sorted(input.grid_spec), concat_dim='tile'))
        ]

        make_experiment(
            output.restart_dir,
            tiles_to_save,
            template_dir=template_dir,
            oro_paths=input.oro_data,
	    vertical_grid=vertical_grid,
            files_to_copy=[
                # TODO move these hardcoded strings to the top
                (f'assets/{grid}_submit_job.sh', 'submit_job.sh'),
                (f'assets/{grid}_input.nml', 'input.nml'),
                (input.coupler, 'INPUT/coupler.res')
            ]
        )

        # Create a diag table with the appropriate time stamp.
        with open(join(output.restart_dir, 'rundir', 'diag_table'), 'w') as file:
            date = datetime.strptime(wildcards['timestep'], '%Y%m%d.%H%M%S')
            date_string = date.strftime('%Y %m %d %H %M %S')
            file.write(f'20160801.00Z.C48.32bit.non-mono\n{date_string}')


            
rule run_restart:
    input:
        experiment=restart_dir_wildcard,
    output:
        touch(restart_dir_done)
    run:
        from src.fv3 import run_experiment
        run_experiment(input.experiment)

rule upload_restart:
    input: restart_dir_done
    output: touch(restart_uploaded_status)
    params: restart=restart_dir_wildcard, gs_path=restart_uploaded
    shell: "gsutil -m rsync -r {params.restart}  {params.gs_path}"


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

rule download_oro_and_grid:
    input: GS.remote(grid_and_orography_data)
    output: oro_files
    shell:"""
    file={input}
    mkdir -p {oro_and_grid_data}
    tar --strip-components=1 -xf $file -C {oro_and_grid_data}
    rm -f $file
    """

rule extract_timestep:
    output:
        extracted
    params:
        extraction_directory=extraction_directory,
        TAR=TAR
    shell:"""
    tarfile={params.TAR}

    gsutil -o 'GSUtil:parallel_thread_count=1' -o 'GSUtil:sliced_object_download_max_component=32' \
          cp  {bucket}/{wildcards.timestep}.tar $tarfile
    
    tar -xvf $tarfile -C {params.extraction_directory}
    rm -f $tarfile
    """

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
        coarsened_grid_spec=coarsened_grid_spec
    run:
        from src.fv3.coarsen import coarsen_grid_spec

        coarsening_factor = ORIGINAL_RESOLUTIONS['grid_spec'] // ORIGINAL_COARSE_RESOLUTION
        coarsen_grid_spec(
            input,
            coarsening_factor,
            output.coarsened_grid_spec
        )        


rule coarsen_restart_category:
    input:
        restart_files=rules.extract_timestep.output,
        coarse_grid_spec=rules.coarsen_grid_spec.output.coarsened_grid_spec,
        native_grid_spec=native_grid_spec
    output:
        coarsened_restart_filenames_wildcard
    run:
        from src.fv3.coarsen import coarsen_restart_file_category

        timestep = wildcards['timestep']
        native_category_name = wildcards['category']
        target_resolution = int(wildcards['grid'][1:])
        coarsening_factor = ORIGINAL_RESOLUTIONS[native_category_name] // target_resolution
        
        coarsen_restart_file_category(
            timestep,
            native_category_name,
            coarsening_factor,
            input.coarse_grid_spec,
            input.native_grid_spec,
            extraction_directory_root,
            output
        )


rule coarsen_all_restart_data:
    input:
        all_coarsened_restart_files
