import apache_beam  
import logging 
from apache_beam.options.pipeline_options import PipelineOptions  
from apache_beam.pvalue import PCollection  
from google.cloud.storage import Client  

from coarseflow.file_lister import FileLister, GCSLister 
from coarseflow.transforms import ExtractAndUploadTimestepWithC3072SurfaceData

logging.basicConfig(level=logging.INFO)


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

    transform_dofn = ExtractAndUploadTimestepWithC3072SurfaceData() 
    _ = matches | apache_beam.ParDo(transform_dofn)

    result = pipeline.run()
    result.wait_until_finish()