from src.data import save_zarr
from snakemake.remote.GS import RemoteProvider as GSRemoteProvider
from os.path import join
GS = GSRemoteProvider()

bucket = "gs://vcm-ml-data/2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data/"

trained_models = [
    "models/random_forest/default.pkl"
]

rule all:
    input: "tmp/20160805.170000"


rule process_restart:
    input: GS.remote(join(bucket, "{timestep}.tar"), keep_local=True)
    output: directory("tmp/{timestep}")
    shell: "cd {output} && tar xf {input}"


rule convert_to_zarr:
    output: directory(save_zarr.output_2d), directory(save_zarr.output_3d)
    shell: "python -m src.data.save_zarr"


rule train_model:
    input: config="configurations/{model_type}/{options}.yaml"
    output: "models/{model_type}/{options}.pkl"
    shell: "python -m src.models.{wildcards.model_type}.train --options {input.config} {output}"
