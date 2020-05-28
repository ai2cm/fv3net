import pytest
import os

from fv3net.pipelines.coarsen_restarts.testing import fv_core_schema


def test_fv_core_schema(regtest):
    print(fv_core_schema(48, 79), file=regtest)
