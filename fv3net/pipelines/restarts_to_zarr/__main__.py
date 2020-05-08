import logging
import argparse
from . import funcs


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="root directory of time steps")
    parser.add_argument("output", help="Location of output zarr")
    parser.add_argument("-s", "--n-steps", default=-1, type=int)
    parser.add_argument("--no-init", action="store_true")
    args, pipeline_args = parser.parse_known_args()
    funcs.run(args, pipeline_args)
