import asyncio
import dataclasses
import json
import itertools
import os
from typing import Mapping, Sequence, Set

import gcsfs
import fsspec


@dataclasses.dataclass
class ReportIndex:
    fs: fsspec.AbstractFileSystem
    reports_by_run: Mapping[str, Sequence[str]]

    @property
    def reports(self) -> Set[str]:
        """The available reports."""
        _reports = [v for v in self.reports_by_run.values()]
        return set(itertools.chain.from_iterable(_reports))

    @classmethod
    def from_reports(cls, url: str, filename: str = "index.html") -> "ReportIndex":
        """Initialize from url containing prognostic reports."""
        loop = asyncio.get_event_loop()
        if url.startswith("gs://"):
            fs = gcsfs.GCSFileSystem(asynchronous=True)
        else:
            fs = fsspec.filesystem("file")
        reports_by_run = loop.run_until_complete(cls._get_reports(fs, url, filename))
        loop.run_until_complete(cls._close_session(fs))
        fs, _, _ = fsspec.get_fs_token_paths(url)  # non-async version for ReportIndex
        return ReportIndex(fs, reports_by_run)

    @staticmethod
    def from_json(url: str) -> "ReportIndex":
        """Initialize from existing JSON file."""
        fs, _, _ = fsspec.get_fs_token_paths(url)
        with fs.open(url) as f:
            index = ReportIndex(fs, json.load(f))
        return index

    def public_links(self, run_url: str):
        """Return public links for all reports containing a run_url."""
        if isinstance(self.fs, gcsfs.GCSFileSystem):
            public_domain = "https://storage.googleapis.com"
        elif isinstance(self.fs, fsspec.implementations.local.LocalFileSystem):
            public_domain = ""
        else:
            raise ValueError(f"Public domain unknown for given filesystem {self.fs}.")

        if run_url not in self.reports_by_run:
            print(f"Provided URL {run_url} not found in any report.")
            public_links = []
        else:
            public_links = [
                os.path.join(public_domain, r) for r in self.reports_by_run[run_url]
            ]
        return public_links

    def dump(self, url: str):
        with self.fs.open(url, "w") as f:
            json.dump(self.reports_by_run, f)

    @classmethod
    async def _get_reports(cls, fs, url, filename):
        out = {}
        for report_dir in await cls._list(fs, url):
            report_url = os.path.join(report_dir, filename)
            try:
                report_head = await cls._cat_file(fs, report_url, end=5 * 1024)
            except FileNotFoundError:
                pass
            else:
                report_lines = report_head.decode("UTF-8").split("\n")
                for line in report_lines[15:]:
                    if "gs://" in line:
                        run_url = line.strip(" ").split(" ")[1]
                        out.setdefault(run_url, []).append(report_url)
        return out

    @staticmethod
    async def _close_session(fs):
        """Handle local and GCS filesystems"""
        try:
            await fs.session.close()
        except AttributeError:
            pass

    @staticmethod
    async def _list(fs: fsspec.AbstractFileSystem, path):
        """Handle local and GCS filesystems"""
        try:
            return await fs._ls(path)
        except AttributeError:
            return fs.ls(path)

    @staticmethod
    async def _cat_file(fs: fsspec.AbstractFileSystem, path, **kwargs):
        """Handle local and GCS filesystems"""
        try:
            return await fs._cat_file(path, **kwargs)
        except AttributeError:
            return fs.cat_file(path, **kwargs)
