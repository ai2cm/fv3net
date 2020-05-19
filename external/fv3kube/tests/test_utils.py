import pytest

import re
from typing import Mapping

from fv3kube import (
    get_base_fv3config,
    job_failed,
    job_complete,
    get_alphanumeric_unique_tag,
)
from fv3kube.utils import _handle_jobs
import kubernetes

from kubernetes.client import V1Job, V1JobStatus, V1ObjectMeta


failed_condition = kubernetes.client.V1JobCondition(type="Failed", status="True")

succesful_condition = kubernetes.client.V1JobCondition(type="Complete", status="True")

failed_status = V1JobStatus(active=None, conditions=[failed_condition])

complete_status = V1JobStatus(active=None, conditions=[succesful_condition])

inprogress_status = V1JobStatus(
    active=1, completion_time=None, conditions=None, failed=None, succeeded=None
)


def test_get_base_fv3config():

    config = get_base_fv3config("v0.3")
    assert isinstance(config, Mapping)


def test_get_base_fv3config_bad_version():

    with pytest.raises(KeyError):
        get_base_fv3config("nonexistent_fv3gfs_version_key")


@pytest.mark.parametrize(
    "func, status, expected",
    [
        (job_failed, complete_status, False),
        (job_failed, inprogress_status, False),
        (job_failed, failed_status, True),
        (job_complete, complete_status, True),
        (job_complete, inprogress_status, False),
        (job_complete, failed_status, False),
    ],
)
def test_job_failed(func, status, expected):
    job = V1Job(status=status)
    assert func(job) == expected


@pytest.mark.parametrize(
    "statuses, raise_on_fail, expected",
    [
        ([complete_status, complete_status, complete_status], True, True),
        ([complete_status, inprogress_status, complete_status], True, False),
        ([complete_status, inprogress_status, failed_status], False, False),
        ([inprogress_status, inprogress_status, inprogress_status], True, False),
    ],
)
def test__handle_jobs_completed(statuses, raise_on_fail, expected):
    jobs = [
        V1Job(metadata=V1ObjectMeta(name=str(k)), status=status)
        for k, status in enumerate(statuses)
    ]
    assert _handle_jobs(jobs, raise_on_fail) == expected


def test__handle_jobs_raises_error():
    statuses = [complete_status, inprogress_status, failed_status]
    jobs = [
        V1Job(metadata=V1ObjectMeta(name=str(k)), status=status)
        for k, status in enumerate(statuses)
    ]

    with pytest.raises(ValueError):
        _handle_jobs(jobs, True)


def test_alphanumeric_unique_tag_length():

    tlen = 8
    tag = get_alphanumeric_unique_tag(tlen)
    assert len(tag) == tlen

    with pytest.raises(ValueError):
        get_alphanumeric_unique_tag(0)


def test_alphanumeric_uniq_tag_is_lowercase_alphanumeric():
    """
    Generate a really long tag to be reasonably certain character restrictions
    are enforced.
    """

    tag = get_alphanumeric_unique_tag(250)
    pattern = "^[a-z0-9]+$"
    res = re.match(pattern, tag)
    assert res is not None
