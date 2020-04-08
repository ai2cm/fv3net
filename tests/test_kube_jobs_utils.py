import pytest

from typing import Mapping

from fv3net.pipelines.kube_jobs import (
    wait_for_complete,
    get_base_fv3config,
    job_failed,
    job_complete,
)
from fv3net.pipelines.kube_jobs.utils import _handle_jobs
import kubernetes

from kubernetes.client import V1Job, V1JobStatus, V1ObjectMeta

import datetime


failed_condition = kubernetes.client.V1JobCondition(type="Failed", status="True")

succesful_condition = kubernetes.client.V1JobCondition(type="Complete", status="True")

failed_status = V1JobStatus(active=None, conditions=[failed_condition])

complete_status = V1JobStatus(active=None, conditions=[succesful_condition])

inprogress_status = V1JobStatus(
    active=1, completion_time=None, conditions=None, failed=None, succeeded=None
)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MockBatchV1ApiResponse(object):
    """
    Mock of the response dictionary object from BatchV1Api
    """

    def __init__(self, items):
        self.items = items

    @classmethod
    def from_args(cls, num_jobs: int, num_successful: int, labels: Mapping[str, str]):
        """
        Args:
            num_jobs: Number of fake jobs to create
            active_jobs: Number of jobs to mark as active. Should be > 1
            num_successful: Number of jobs to mark as successful.
            labels: Labels to apply to all generated jobs.
        """

        items = []
        for i in range(num_jobs):

            job_name = f"job{i}"
            success = int(i + 1 <= num_successful)
            info = cls._gen_job_info(job_name, success, labels)
            items.append(info)

        return cls(items)

    @staticmethod
    def _gen_job_info(
        job_name: str, success: bool, labels: Mapping[str, str],
    ):

        info = dotdict(
            metadata=dotdict(labels=labels, name=job_name, namespace="default",),
            status=dotdict(
                active=1,
                failed=(1 if not success else None),
                succeeded=(1 if success else None),
            ),
        )

        return info

    def delete_job_item(self, job_name, job_namespace):

        for i, job_item in enumerate(self.items):
            curr_job_name = job_item.metadata.name
            curr_job_namespace = job_item.metadata.namespace
            if job_name == curr_job_name and job_namespace == curr_job_namespace:
                break

        del self.items[i]

    def make_jobs_inactive(self):

        for job_item in self.items:
            job_item.status.active = None

    def get_job_and_namespace_tuples(self):

        results = []
        for job_info in self.items:
            metadata = job_info.metadata
            res_tuple = (metadata.name, metadata.namespace)
            results.append(res_tuple)

        return results

    def get_response_with_matching_labels(self, labels):

        items = []
        for job_info in self.items:
            job_labels = job_info.metadata.labels
            labels_match = self._check_labels_in_job_info(job_labels, labels)
            if labels_match:
                items.append(job_info)

        return self.__class__(items)

    @staticmethod
    def _check_labels_in_job_info(job_labels, check_labels):
        for label_key, label_value in check_labels.items():
            item = job_labels.get(label_key, None)
            if item is not None and item == label_value:
                return True

        return False


class MockBatchV1Api(object):

    """
    Mock of kubernetes.client.BatchV1Api to test kube_job.utils
    """

    def __init__(self, mock_response: MockBatchV1ApiResponse):

        self.response = mock_response
        self.num_list_calls = 0

    def list_job_for_all_namespaces(self, label_selector):

        # switch to non-active to kill the wait loop after 1 call
        if self.num_list_calls >= 1:
            self.response.make_jobs_inactive()
        elif self.num_list_calls > 5:
            raise TimeoutError("Probably stuck in a loop.")

        self.num_list_calls += 1

        label_dict = self._parse_label_selector(label_selector)
        return self.response.get_response_with_matching_labels(label_dict)

    @staticmethod
    def _parse_label_selector(label_selector):

        kv_pairs = [kv_pair.split("=") for kv_pair in label_selector.split(",")]
        labels_dict = {k: v for k, v in kv_pairs}

        return labels_dict

    def delete_namespaced_job(self, job_name, namespace):

        self.response.delete_job_item(job_name, namespace)


@pytest.fixture
def mock_batch_api():

    num_jobs = 4
    num_success = 3
    labels = {"test-group": "test-label", "group2": "grp2-label"}
    mock_response = MockBatchV1ApiResponse.from_args(num_jobs, num_success, labels)
    mock_api = MockBatchV1Api(mock_response)

    return num_jobs, num_success, mock_api, labels


@pytest.mark.skip()
def test_wait_for_complete(mock_batch_api):

    num_jobs, num_sucess, batch_client, labels = mock_batch_api
    success, fail = wait_for_complete(
        labels, batch_client=batch_client, sleep_interval=2
    )

    assert len(success) == num_sucess
    assert len(fail) == num_jobs - num_sucess


def test_get_base_fv3config():

    config = get_base_fv3config("v0.2")
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


@pytest.mark.parametrize('statuses, expected',[
    ([complete_status, complete_status, complete_status], True),
    ([complete_status, inprogress_status, complete_status], False),
    ([inprogress_status, inprogress_status, inprogress_status], False),
]
)
def test__handle_jobs_completed(statuses, expected):
    jobs = [
        V1Job(metadata=V1ObjectMeta(name=str(k)), status=status)
        for k, status in enumerate(statuses)
    ]
    assert _handle_jobs(jobs) == expected


def test__handle_jobs_raises_error():
    statuses = [complete_status, inprogress_status, failed_status]
    jobs = [
        V1Job(metadata=V1ObjectMeta(name=str(k)), status=status)
        for k, status in enumerate(statuses)
    ]

    with pytest.raises(ValueError):
        _handle_jobs(jobs)