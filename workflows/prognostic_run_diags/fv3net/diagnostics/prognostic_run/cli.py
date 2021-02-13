import argparse
from fv3net.diagnostics.prognostic_run import metrics, compute
from fv3net.diagnostics.prognostic_run.views import (
    movie_stills,
    static_report,
)  # ignore: E402


parser = argparse.ArgumentParser(description="Prognostic run diagnostics")
subparsers = parser.add_subparsers(help="Prognostic run diagnostics")

compute.register_parser(subparsers)
metrics.register_parser(subparsers)
movie_stills.register_parser(subparsers)
static_report.register_parser(subparsers)

if __name__ == "__main__":
    parser.parse_args()
