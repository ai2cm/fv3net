import argparse
from fv3net.artifacts import query
from fv3net.artifacts import report_search
from fv3net.artifacts import generate


def get_parser():
    parser = argparse.ArgumentParser(
        description="Query available experiment and report output."
    )
    subparsers = parser.add_subparsers(required=True, dest="command")
    query.register_parser(subparsers)
    report_search.register_parser(subparsers)
    generate.register_parser(subparsers)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.func(args)
