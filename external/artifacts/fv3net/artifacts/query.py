from typing import Optional, Container, List, Awaitable, Sequence
import asyncio
import dataclasses
import datetime
import os
import pathlib

import fsspec
import gcsfs

from .utils import _list, _close_session


STEPS = [
    "fv3gfs_run",
    "fv3gfs_run_diagnostics",
    "offline_diags",
    "trained_models",
]


@dataclasses.dataclass
class Step:
    fs: fsspec.AbstractFileSystem
    path: pathlib.Path
    date_created: datetime.date
    project: str
    tag: str
    step: str


async def flat_gather(promises: Sequence[Awaitable[Step]]) -> List[Step]:
    list_of_list_of_steps = await asyncio.gather(*promises)
    return sum(list_of_list_of_steps, [])


async def _get_runs_with_tag(f, tag, date, project):
    out = []
    for step in await _list(f, tag):
        path = pathlib.Path(step)
        if path.name in STEPS:
            out.append(
                Step(
                    f,
                    path,
                    date,
                    pathlib.Path(project).name,
                    pathlib.Path(tag).name,
                    path.name,
                )
            )
    return out


async def _get_runs_in_date(f, path_with_date, project):
    path = pathlib.Path(path_with_date)

    try:
        date = datetime.date.fromisoformat(path.name)
    except ValueError:
        return []

    return await flat_gather(
        [
            _get_runs_with_tag(f, tag, date, project)
            for tag in await _list(f, path.as_posix())
        ]
    )


async def _get_runs(f, bucket, projects):

    top_level = [os.path.join(bucket, project) for project in set(projects)]

    files = []

    for project in top_level:
        total = await flat_gather(
            [_get_runs_in_date(f, obj, project) for obj in await _list(f, project)]
        )

        files.extend(total)
    return files


def get_artifacts(bucket, projects):
    loop = asyncio.get_event_loop()
    if bucket.startswith("gs://"):
        f = gcsfs.GCSFileSystem(asynchronous=True)
    else:
        f = fsspec.filesystem("file")

    ans = loop.run_until_complete(_get_runs(f, bucket, projects or ["default"]))
    loop.run_until_complete(_close_session(f))
    return ans


def matches_step(step: Step, name: Container[str]) -> bool:
    return step.step in name or not name


def matches_tag(step: Step, tag: Optional[str]) -> bool:
    if tag is None:
        return True
    else:
        return tag in step.tag


def list(args):
    for run in get_artifacts(args.bucket, args.project):
        if matches_step(run, args.step) and matches_tag(run, args.tag):
            print(
                run.project,
                run.date_created,
                run.tag,
                run.step,
                "gs://" + run.path.as_posix(),
            )


def register_parser(subparsers):
    parser = subparsers.add_parser("ls", help="Query the experiments buckets.")
    parser.add_argument("step", nargs="*", help="One of " + ", ".join(STEPS))
    parser.add_argument(
        "-b",
        "--bucket",
        help="Bucket. Default: gs://vcm-ml-experiments",
        default="gs://vcm-ml-experiments",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="Include project in list. Can be passed multiple times. Default: default",
        action="append",
        default=[],
    )
    parser.add_argument(
        "-t",
        "--tag",
        help="Subtring of experiment tag. Any artifacts with tags containing this "
        "will be printed.",
        default=None,
    )
    parser.set_defaults(func=list)
