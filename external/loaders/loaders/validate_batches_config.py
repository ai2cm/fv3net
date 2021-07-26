import loaders
import argparse
import yaml


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="path of BatchesLoader configuration yaml file",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config, "r") as f:
        loaders.BatchesLoader.from_dict(yaml.safe_load(f))
