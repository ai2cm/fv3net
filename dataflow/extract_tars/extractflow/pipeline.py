import apache_beam  
import logging 
from apache_beam.options.pipeline_options import PipelineOptions  
from apache_beam.pvalue import PCollection  
from google.cloud.storage import Client  

from dataflow_utils.gcs_utils import FileLister, GCSLister 
import extractflow.transforms as cftransforms

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def run(file_lister: FileLister, prefix: str, file_extension: str,
        output_prefix: str) -> None:
    """
    Pipeline currently specified for tar extraction processing.

    Downloads timestep tarfile and extracts high-res surface data
    into a directory and coarsened atmosphere files into another.
    """
    pipeline = apache_beam.Pipeline(options=PipelineOptions(pipeline_type_check=True))

    to_extract: PCollection[str] = apache_beam.Create(
        file_lister.list(
            prefix=prefix,
            file_extension=file_extension
        )
    ).with_output_types(str)

    filter_finished: PCollection[str] = apache_beam.Filter(
        cftransforms.not_finished_with_tar_extract,
        output_prefix
    ).with_output_types(str)

    extract_fn = apache_beam.ParDo(
        cftransforms.ExtractAndUploadTimestepWithC3072SurfaceData(
            output_prefix
        )
    )

    _ = pipeline | to_extract | filter_finished | extract_fn

    result = pipeline.run()
    result.wait_until_finish()