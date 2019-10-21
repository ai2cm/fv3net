import apache_beam  # type: ignore
from apache_beam.options.pipeline_options import PipelineOptions  # type: ignore
from apache_beam.pvalue import PCollection  # type: ignore
from google.cloud.storage import Client  # type: ignore

from coarseflow.file_lister import FileLister, GCSLister
from coarseflow.transformer import Transformer


def run(file_lister: FileLister, prefix: str, file_extension: str) -> None:
    # Run pipeline stuff here
    """
    Current workflow taken from snakemake

    Get data extracted from tar
    Place in restart directory
    Coarsen surface data place in restart directory
    get the C3702 gridspec place in restart directory?
    get grid/oro files and place in restart directory
    prepare the restart directory

    final step is a restart directory with all of the correct files in import
    could place a finished file 
    
    """
    pipeline = apache_beam.Pipeline(options=PipelineOptions(pipeline_type_check=True))

    matches: PCollection[str] = pipeline | apache_beam.Create(
        file_lister.list(
            prefix=prefix,
            file_extension=file_extension
        )
    ).with_output_types(str)

    # _ = matches | apache_beam.ParDo(Transformer('vcm-ml-'))

    result = pipeline.run()
    result.wait_until_finish()


if __name__ == '__main__':
    run(GCSLister(Client(), 'vcm-ml-data'))