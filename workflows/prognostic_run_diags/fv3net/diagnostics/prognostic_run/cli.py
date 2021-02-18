import argparse
from fv3net.diagnostics.prognostic_run import metrics, compute
from fv3net.diagnostics.prognostic_run.views import (
    movie_stills,
    static_report,
)  # ignore: E402


def dissoc(namespace, key):
    return argparse.Namespace(
        **{k: val for k, val in vars(namespace).items() if k != key}
    )


def main():
    parser = argparse.ArgumentParser(description="Prognostic run diagnostics")
    subparsers = parser.add_subparsers(help="Prognostic run diagnostics")

    compute.register_parser(subparsers)
    metrics.register_parser(subparsers)
    movie_stills.register_parser(subparsers)
    static_report.register_parser(subparsers)
    args = parser.parse_args()

    # need to remove the 'func' entry since some of these scripts save these
    # arguments in netCDF attributes
    args.func(dissoc(args, "func"))


if __name__ == "__main__":
    main()
