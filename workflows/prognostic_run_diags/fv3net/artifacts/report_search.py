import asyncio
import dataclasses
import json
import itertools
import os
from typing import Mapping, Sequence, Set

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
                report_lines = [l.replace(" ", "") for l in report_lines]
                for line in report_lines:
                    if "<td>gs://" in line:
                        run_url = line.split("<td>")[1].split("</td>")[0]
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
