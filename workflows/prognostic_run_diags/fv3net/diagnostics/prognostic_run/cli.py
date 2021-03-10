import argparse
from fv3net.diagnostics.prognostic_run import metrics, compute
from fv3net.diagnostics.prognostic_run.views import movies, static_report


def dissoc(namespace, key):
    return argparse.Namespace(
        **{k: val for k, val in vars(namespace).items() if k != key}
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Prognostic run diagnostics")
    subparsers = parser.add_subparsers(help="Prognostic run diagnostics")

    compute.register_parser(subparsers)
    metrics.register_parser(subparsers)
    movies.register_parser(subparsers)
    static_report.register_parser(subparsers)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # need to remove the 'func' entry since some of these scripts save these
    # arguments in netCDF attributes
    args.func(dissoc(args, "func"))


if __name__ == "__main__":
    main()
