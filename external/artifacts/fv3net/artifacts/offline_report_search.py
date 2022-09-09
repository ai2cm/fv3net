import os
from typing import Optional

from .report_search import ReportIndex

REPORT_URL_DEFAULT = "gs://vcm-ml-public/offline_ml_diags"

def _get_model_url(line: str) -> Optional[str]:
    if '"model_path": "gs://' in line:
        return line.split('"model_path": ')[1].strip('",')
    else:
        return None

def main(args):
    if args.write:
        index = ReportIndex(_search_function=_get_model_url)
        index.compute(args.reports_url, recurse_once=True)
        index.dump(os.path.join(args.reports_url, "index.json"))
    index = ReportIndex.from_json(os.path.join(args.reports_url, "index.json"))
    for link in index.public_links(args.url):
        print(link)


def register_parser(subparsers):
    parser = subparsers.add_parser("offline-report", help="Search for offline ML reports.")
    parser.add_argument("url", help="An fv3fit model URL.")
    parser.add_argument(
        "-r",
        "--reports-url",
        help=(
            f"Location of offline reports. Defaults to {REPORT_URL_DEFAULT}. "
            "Search uses index at REPORTS_URL/index.json"
        ),
        default=REPORT_URL_DEFAULT,
    )
    parser.add_argument(
        "-w",
        "--write",
        help="Recompute index and write to REPORTS_URL/index.json before searching.",
        action="store_true",
    )
    parser.set_defaults(func=main)
