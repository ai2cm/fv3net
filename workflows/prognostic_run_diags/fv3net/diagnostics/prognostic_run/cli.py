import argparse
from fv3net.diagnostics.prognostic_run import metrics, compute
from fv3net.diagnostics.prognostic_run.views import (
    movie_stills,
    static_report,
)  # ignore: E402


def main():
    parser = argparse.ArgumentParser(description="Prognostic run diagnostics")
    subparsers = parser.add_subparsers(help="Prognostic run diagnostics")

    compute.register_parser(subparsers)
    metrics.register_parser(subparsers)
    movie_stills.register_parser(subparsers)
    static_report.register_parser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
