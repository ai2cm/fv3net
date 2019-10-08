from src.data import save_zarr
from snakemake.remote.GS import RemoteProvider as GSRemoteProvider
from os.path import join
GS = GSRemoteProvider()

bucket = "gs://vcm-ml-data/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data"
TAR="data/raw/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data/{timestep}.tar"
EXTRACTED="data/extracted/{timestep}/"

trained_models = [
    "models/random_forest/default.pkl"
]

timesteps = [
"20160805.170000"
]

rule all:
    input: expand(EXTRACTED, timestep=timesteps)


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
