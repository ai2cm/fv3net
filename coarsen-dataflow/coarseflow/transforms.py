import tempfile
import logging
import subprocess
import shutil
import logging
from pathlib import Path
import apache_beam
from apache_beam.pvalue import PCollection  # type: ignore

import coarseflow.utils as cfutils
from coarseflow.file_lister import GCSLister

from google.cloud.storage import Client, Bucket, Blob

logger = logging.getLogger(__name__)

@apache_beam.typehints.with_input_types(str)
@apache_beam.typehints.with_output_types(None)
class ExtractAndUploadTimestepWithC3072SurfaceData(apache_beam.DoFn):
    def __init__(self):
        super().__init__()

    def process(self, element):
        
        # TODO: Un-hardcode this
        output_prefix = '2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted'

        with tempfile.TemporaryDirectory() as tmpdir:

            timestep_blob = cfutils.init_blob_from_gcs_url(element)
            filename = Path(timestep_blob.name).name
            downloaded_timestep = cfutils.download_blob_to_file(timestep_blob, tmpdir, filename)
            untarred_timestep = cfutils.extract_tarball_to_path(downloaded_timestep)

            current_timestep = untarred_timestep.name
            c384_path = Path(tmpdir, 'coarsened', 'C384', current_timestep)
            c384_path.mkdir(parents=True, exist_ok=True)
            c3702_path = Path(tmpdir, 'coarsened', 'C3702', current_timestep)
            c3702_path.mkdir(parents=True, exist_ok=True)

            # move highres sfc data to separate dir
            for sfc_file in untarred_timestep.glob('*.sfc_data.*'):

                destination = c3702_path.joinpath(sfc_file.name)
                shutil.move(sfc_file, destination)

            # upload highres sfc data
            c3702_blob_prefix = c3702_path.relative_to(c3702_path.parent.parent)
            c3702_blob_prefix = str(Path(output_prefix, c3702_blob_prefix))
            cfutils.upload_dir_to_gcs('vcm-ml-data', c3702_blob_prefix, c3702_path)

            # upload pre-coarsened files to timestep
            c384_blob_prefix = c384_path.relative_to(c384_path.parent.parent)
            c384_blob_prefix = str(Path(output_prefix, c384_blob_prefix))
            cfutils.upload_dir_to_gcs('vcm-ml-data', c384_blob_prefix, untarred_timestep)