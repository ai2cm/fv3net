import logging

from google.cloud.storage import Client

from extractflow.pipeline import run
from dataflow_utils.gcs import GCSLister

# tar_file_source = '2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data'
tar_file_source = 'test_dataflow'
extracted_destination = '2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted'

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    run(
        GCSLister(Client(), 'vcm-ml-data'),
        prefix=tar_file_source,
        file_extension='tar',
        output_prefix=extracted_destination
    )

    