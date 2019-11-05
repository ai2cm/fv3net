import logging
import argparse
from google.cloud.storage import Client

from extractflow.pipeline import run
from dataflow_utils import gcs


if __name__ == "__main__":

    """
    extractflow __main__ expects two provided commandline arguments to run:

    tarfile_source_prefix: GCS bucket prefix to grab list of tarfiles from
    extracted_destination_prefix: GCS bucket prefix to place
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("tarfile_source_prefix")
    parser.add_argument("extracted_destination_prefix")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    required_args, pipeline_args = parser.parse_known_args()
    logger.info(f'Tarfile input GCS Prefix: {required_args.tarfile_source_prefix}')
    logger.info('Extracted files destination GCS Prefix: '
                f'{required_args.extracted_destination_prefix}')

    run(
        gcs.list_bucket_files(
            Client(),
            'vcm-ml-data',
            prefix=required_args.tarfile_source_prefix,
            file_extension='tar'),
        output_prefix=required_args.extracted_destination_prefix,
        pipeline_args=pipeline_args
    )
