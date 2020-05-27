import pytest
import os
from distutils import dir_util

ONE_STEP_ZARR_SCHEMA = "tests/loaders/one_step_zarr_schema.json"


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


class MockDatasetMapper:
    # mocks up the mappings except for the part where it opens the 
    def __init__(self, timesteps_dir):
        self._generate_mapping()

    def _generate_mapping(self):
        with open(ONE_STEP_ZARR_SCHEMA) as f:
            schema = synth.load(f)
        one_step_dataset = synth.generate(schema)
        self.mapping = {}
        for i in range(4):
            self.mapping[f"2020050{i}.000000"] = one_step_dataset

    def __getitem__(self, key: str) -> xr.Dataset:
        return self.mapping[key]

    def keys(self):
        return list(self.mapping.keys())

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())




@pytest.fixture
def test_dataset_mapper():
    return MockDatasetMapper