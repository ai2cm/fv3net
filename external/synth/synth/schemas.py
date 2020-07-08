from synth.core import DatasetSchema, load
import os

CURRENT_DIR = os.path.dirname(__file__)

def load_schema(name: str) -> DatasetSchema:
    """Load a schema by name from the current library"""
    path = os.path.join(CURRENT_DIR, "_dataset_fixtures", name)
    with open(path) as f:
        return load(f)
