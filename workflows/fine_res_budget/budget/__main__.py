import argparse
import logging

from .pipeline import run

logging.basicConfig(level=logging.DEBUG)

for name in ['gcsfs.core', 'urllib3.connectionpool']:
    logging.getLogger(name).setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("physics_url")
parser.add_argument("restart_url")
parser.add_argument("output_dir")

args, extra_args = parser.parse_known_args()

run(args.restart_url, args.physics_url, args.output_dir, extra_args)
