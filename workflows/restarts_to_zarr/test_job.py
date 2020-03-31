from job import map_robustly
from collections import defaultdict
import pytest


def test_map_robustly_raises():
     
    count = defaultdict(lambda: 0)

    def fn(x):
        if count[x] < 3:
            count[x] += 1
            raise ValueError
        else:
            return True
        
    with pytest.raises(ValueError):
        map_robustly(fn, ['a'], retries=2)

    # returns succesfully
    map_robustly(fn, ['a'], retries=3)
