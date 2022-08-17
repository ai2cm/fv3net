import argparse
import logging

from fv3net.diagnostics.prognostic_run import compute, metrics, shell
from fv3net.diagnostics.prognostic_run.apps import log_viewer
from fv3net.diagnostics.prognostic_run.emulation import single_run
from fv3net.diagnostics.prognostic_run.views import movies, static_report


def dissoc(namespace, key):
    return argparse.Namespace(
        **{k: val for k, val in vars(namespace).items() if k != key}
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Prognostic run diagnostics")
    parser.add_argument("--log-level", type=str, default="INFO")
    subparsers = parser.add_subparsers(
        help="Prognostic run diagnostics", required=True, dest="command"
    )

    for entrypoint in [
        compute,
        log_viewer,
        metrics,
        movies,
        shell,
        single_run,
        static_report,
    ]:
        entrypoint.register_parser(subparsers)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.log_level:
        level = getattr(logging, args.log_level)
        logging.getLogger("gcsfs").setLevel(level)
        logging.basicConfig(level=level)

    # need to remove the 'func' entry since some of these scripts save these
    # arguments in netCDF attributes
    args.func(dissoc(args, "func"))


if __name__ == "__main__":
    main()
