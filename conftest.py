import pytest
import os
from distutils import dir_util

collect_ignore_glob = ["*ipynb_checkpoints*"]


@pytest.fixture
def datadir(tmpdir, request):
    """
    Fixture to help load data files into a test based on the name of the
    module containing the test function.

    For example, if the name of the test file is named
    ``path/to/test_integration.py``, then and data in
    ``path/to/test_integration/`` will be copied into the temporary directory
    returned by this fixture.

    Returns:
        tmpdir (a temporary directory)

    Credit:
        https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


@pytest.fixture(scope="module")
def datadir_module(tmpdir_factory, request):
    """Copy of above but attempt at module scoping"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    tmpdir = tmpdir_factory.mktemp("pytest_data")

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir
