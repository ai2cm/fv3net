from src.data import save_zarr
from snakemake.remote.GS import RemoteProvider as GSRemoteProvider
from os.path import join
GS = GSRemoteProvider(keep_local=True)

bucket = "gs://vcm-ml-data/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data"
TAR="data/raw/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data/{timestep}.tar"
EXTRACTED="data/extracted/{timestep}/"
c3072_grid_spec_pattern = "gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-diagnostics/grid_spec.tile{tile}.nc.{subtile:04d}"
coarsened_sfc_data_wildcard="data/coarsened/c3072/{timestep}.sfc_data.nc"

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
    input: expand(coarsened_sfc_data_wildcard, timestep=timesteps)

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

rule download_grid_spec:
    output: directory("data/raw/grid_specs/c3072")
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
