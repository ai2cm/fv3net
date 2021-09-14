import asyncio
import dataclasses
import json
import itertools
import os
from typing import Mapping, Optional, Sequence, Set

import gcsfs
import fsspec

from .utils import _list, _cat_file, _close_session


@dataclasses.dataclass
class ReportIndex:
    """Mapping from run urls to sequences of report urls."""

    reports_by_run: Mapping[str, Sequence[str]] = dataclasses.field(
        default_factory=dict
    )

    @property
    def reports(self) -> Set[str]:
        """The available reports."""
        _reports = [v for v in self.reports_by_run.values()]
        return set(itertools.chain.from_iterable(_reports))

    def compute(self, url, filename="index.html"):
        """Compute reports_by_run index from all reports found at url.

        Args:
            url: path to directory containing report subdirectories.
            filename: name of report html files.

        Note:
            Reports are assumed to be located at {url}/*/{filename}.
        """
        loop = asyncio.get_event_loop()
        if url.startswith("gs://"):
            fs = gcsfs.GCSFileSystem(asynchronous=True)
        else:
            fs = fsspec.filesystem("file")
        self.reports_by_run = loop.run_until_complete(
            self._get_reports(fs, url, filename)
        )
        loop.run_until_complete(_close_session(fs))

    @staticmethod
    def from_json(url: str) -> "ReportIndex":
        """Initialize from existing JSON file."""
        with fsspec.open(url) as f:
            index = ReportIndex(json.load(f))
        return index

    def public_links(self, run_url: str) -> Sequence[str]:
        """Return public links for all reports containing a run_url."""
        if run_url not in self.reports_by_run:
            print(f"Provided URL {run_url} not found in any report.")
            public_links = []
        else:
            public_links = [
                self._insert_public_domain(report_url)
                for report_url in self.reports_by_run[run_url]
            ]
        return public_links

    def dump(self, url: str):
        with fsspec.open(url, "w") as f:
            json.dump(self.reports_by_run, f, sort_keys=True, indent=4)

    async def _get_reports(self, fs, url, filename) -> Mapping[str, Sequence[str]]:
        """Generate mapping from run URL to report URLs for all reports found at
        {url}/*/{filename}."""
        out = {}
        for report_dir in await _list(fs, url):
            report_url = self._url_prefix(fs) + os.path.join(report_dir, filename)
            try:
                report_head = await _cat_file(fs, report_url, end=5 * 1024)
            except FileNotFoundError:
                pass
            else:
                report_lines = report_head.decode("UTF-8").split("\n")
                for line in report_lines:
                    run_url = _get_run_url(line)
                    if run_url:
                        out.setdefault(run_url, []).append(report_url)
        return out

    @staticmethod
    def _url_prefix(fs) -> str:
        if isinstance(fs, gcsfs.GCSFileSystem):
            return "gs://"
        elif isinstance(fs, fsspec.implementations.local.LocalFileSystem):
            return ""
        else:
            raise ValueError(f"Protocol prefix unknown for {fs}.")

    @staticmethod
    def _insert_public_domain(url) -> str:
        if url.startswith("gs://"):
            return url.replace("gs://", "https://storage.googleapis.com/")
        elif url.startswith("/"):
            return url
        else:
            raise ValueError(f"Public domain unknown for url {url}.")


def _get_run_url(line: str) -> Optional[str]:
    if "<td> gs://" in line:
        # handles older style reports
        return line.split("<td>")[1].split("</td>")[0].strip()
    elif '": "gs://' in line:
        # handles newer style reports generated after
        # https://github.com/VulcanClimateModeling/fv3net/pull/1304
        return line.split(": ")[1].strip('",')
    else:
        return None


def main(args):
    if args.write:
        index = ReportIndex()
        index.compute(args.reports_url)
        index.dump(os.path.join(args.reports_url, "index.json"))
    index = ReportIndex.from_json(os.path.join(args.reports_url, "index.json"))
    for link in index.public_links(args.url):
        print(link)


def register_parser(subparsers):
    parser = subparsers.add_parser("report", help="Search for prognostic run reports.")
    parser.add_argument("url", help="A prognostic run URL.")
    parser.add_argument(
        "-r",
        "--reports-url",
        help=(
            "Location of prognostic run reports. Defaults to gs://vcm-ml-public/argo. "
            "Search uses index at REPORTS_URL/index.json"
        ),
        default="gs://vcm-ml-public/argo",
    )
    parser.add_argument(
        "-w",
        "--write",
        help="Recompute index and write to REPORTS_URL/index.json before searching.",
        action="store_true",
    )
    parser.set_defaults(func=main)
