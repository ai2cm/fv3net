import pytest
from fv3fit._shared.data import train_validation_split_batches



@pytest.mark.parametrize(
    "timesteps, tsteps_per_train, tsteps_per_val, expect_train, expect_val",
    (
        [range(9), 2, 1, [0,1,2,3,4,5], [6,7,8]],
        [range(10), 3, 1, [0,1,2,3,4,5,6], [7,8,9]],
        [range(4), 4, 1, [0,1,2], [3]],
        [range(4), 5, 1, [0,1,2], [3]],
        [range(4), 2, 3, [0,1], [2,3]]
    ),
)
def test_train_validation_split_batches(timesteps, tsteps_per_train, tsteps_per_val, expect_train, expect_val):
    train, val = train_validation_split_batches(timesteps, tsteps_per_train, tsteps_per_val)
    assert set(train) == set(expect_train)
    assert set(val) == set(expect_val)
    assert len(train + val) == len(timesteps)
