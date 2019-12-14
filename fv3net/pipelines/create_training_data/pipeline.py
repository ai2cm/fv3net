import numba
import apache_beam as beam
import argparse
import logging
import gcsfs
import os

from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions


from fv3net.pipelines.create_training_data.dataset_creator import \
    create_training_dataset, write_to_zarr
from vcm.convenience import get_timestep_from_filename

del numba

logging.basicConfig(level=logging.INFO)


class CreateTrainZarr(beam.DoFn):
    def __init__(self, args):
        self.mask_to_surface_type = args.mask_to_surface_type
        self.gcs_output_data_dir = args.gcs_output_data_dir
        self.gcs_bucket = args.gcs_bucket
        self.gcs_project = args.gcs_project

    def process(self, data_urls):
        ds = create_training_dataset(
            data_urls,
            self.mask_to_surface_type,
            project=self.gcs_project)
        zarr_file = get_timestep_from_filename(data_urls[0]) + '.zarr'
        write_to_zarr(
            ds,
            self.gcs_output_data_dir,
            zarr_file,
            bucket=self.gcs_bucket, )



def run(args, pipeline_args):
    fs = gcsfs.GCSFileSystem(project=args.gcs_project)

    zarr_dir = os.path.join(args.gcs_bucket, args.gcs_input_data_path)
    gcs_urls = sorted(fs.ls(zarr_dir))
    num_outputs = int((len(gcs_urls)-1)/args.timesteps_per_output_file)
    tstep_pairs = [
        (args.timesteps_per_output_file * i,
         args.timesteps_per_output_file * i + (args.timesteps_per_output_file + 1))
        for i in range(num_outputs)]
    data_urls = [
        gcs_urls[start_ind:stop_ind] for start_ind, stop_ind in tstep_pairs]

    print(f"Processing {len(data_urls)} subsets...")
    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (p
         | beam.Create(data_urls)
         | "CreateSubsetZarr" >> beam.ParDo(CreateTrainZarr(args))
         )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gcs-input-data-path',
        type=str,
        required=True,
        help="Location of input data in Google Cloud Storage bucket. "
             "Don't include bucket in path."
    )
    parser.add_argument(
        '--gcs-output-data-dir',
        type=str,
        required=True,
        help="Write path for train data in Google Cloud Storage bucket. "
             "Don't include bucket in path."
    )
    parser.add_argument(
        '--gcs-bucket',
        type=str,
        default='gs://vcm-ml-data',
        help="Google Cloud Storage bucket name."
    )
    parser.add_argument(
        '--gcs-project',
        type=str,
        default='vcm-ml',
        help="Project name for google cloud."
    )
    parser.add_argument(
        '--mask-to-surface-type',
        type=str,
        default=None,
        help="Mask to surface type in ['sea', 'land', 'seaice']."
    )
    parser.add_argument(
        '--timesteps-per-output-file',
        type=int,
        default=2,
        help="Number of consecutive timesteps to calculate features/targets for in "
             "a single process and save to output file."
             "When the full output is shuffled at the data generator step, these"
             "timesteps will always be in the same training data batch."

    )
    args, pipeline_args = parser.parse_known_args()

    """Main function"""
    run(args=args, pipeline_args=pipeline_args)
