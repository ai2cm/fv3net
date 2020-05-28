import pytest
import os

from fv3net.pipelines.coarsen_restarts.testing import fv_core_schema


def test_fv_core_schema(regtest):
    names = dict(x="xaxis_1", xi="xaxis_2", y="yaxis_2", yi="yaxis_1", z="zaxis_1")
    print(fv_core_schema(48, 79, **names), file=regtest)
