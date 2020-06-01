# This module should only contain loaders we mean to be able to specify in the
# model training configuration
from ._one_step import open_one_step
from ._fine_resolution_budget import (
    open_fine_resolution_budget,
    open_fine_res_apparent_sources,
)

# TODO: replace with open mapping func for nudging data source
from ._nudged import load_nudging_batches

from ._batch import mapper_to_diagnostic_sequence, mapper_to_batches
