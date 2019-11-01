from src import gcs, fv3
import sys
import tempfile
from datetime import datetime
from os.path import join
import os
import logging
from functools import partial

RESTART_DIR_PATTERN = "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/restart/C48/{time}/rundir"
OUTPUT_DIR_PATTERN = "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/one-step-run/C48/{time}/rundir"


def dir(time):
    return RESTART_DIR_PATTERN.format(time=time)


def output(time):
    return OUTPUT_DIR_PATTERN.format(time=time)


def diag_table_time(time: str) -> str:
    date = datetime.strptime(time, '%Y%m%d.%H%M%S')
    date_string = date.strftime('%Y %m %d %H %M %S')
    return date_string


def patch_diag_table(dir, time):
    with open(join(dir, 'rundir', 'diag_table'), 'w') as file:
        date_string = diag_table_time(time)
        file.write(f'20160801.00Z.C48.32bit.non-mono\n{date_string}')
        # add output of the grid spec for post-processing purposes (TODO replace all this with fv3config)
        file.write(
            '''
            #output files
            "grid_spec",              -1,  "months",   1, "days",  "time"
            ###
            # grid_spec
            ###
            "dynamics", "grid_lon", "grid_lon", "grid_spec", "all", .false.,  "none", 2,
            "dynamics", "grid_lat", "grid_lat", "grid_spec", "all", .false.,  "none", 2,
            "dynamics", "grid_lont", "grid_lont", "grid_spec", "all", .false.,  "none", 2,
            "dynamics", "grid_latt", "grid_latt", "grid_spec", "all", .false.,  "none", 2,
            "dynamics", "area",     "area",     "grid_spec", "all", .false.,  "none", 2,
            ''')


def main(time, rundir_transformations=()):
    with tempfile.TemporaryDirectory() as localdir:
        gcs.copy(dir(time), localdir)
        logging.info("running experiment")
        for transform in rundir_transformations:
            transform(localdir)
        try:
            fv3.run_experiment(localdir)
        except Exception as e:
            logging.critical(f"Experiment failed. Listing rundir for debugging purposes: {os.listdir(localdir)}")
            raise e
        gcs.copy(localdir + '/*', output(time))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('time')
    args = parser.parse_args()
    time = args.time

    main(args.time,
         rundir_transformations=[
             partial(patch_diag_table, time=time)
         ])
