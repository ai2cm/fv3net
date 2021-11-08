import numpy as np
import pytest
import xarray as xr
from runtime.steppers.combine import CombinedStepper


class MockStepper:
    def __init__(self, output_tendencies, output_diags, output_state_updates):
        self.output_tendencies = output_tendencies
        self.output_diags = output_diags
        self.output_state_updates = output_state_updates

    def __call__(self, time, state):
        return self.output_tendencies, self.output_diags, self.output_state_updates


da = xr.DataArray(np.array([0.0, 1.0, 2.0]), dims=["x"], attrs={"units": None})


@pytest.mark.parametrize(
    "stepper0_outputs, stepper1_outputs, passes",
    [
        pytest.param(
            ({"t0": da}, {"d0": da}, {}),
            ({"t1": da}, {"d1": da}, {}),
            True,
            id="no_overlap",
        ),
        pytest.param(
            ({}, {}, {}), ({"t1": da}, {"d1": da}, {}), True, id="one_empty_output"
        ),
        pytest.param(
            ({"t0": da}, {"d1": da}, {}),
            ({"t1": da}, {"d1": da}, {}),
            False,
            id="one_collision",
        ),
    ],
)
def test_CombinedStepper_raises_error_on_collision(
    stepper0_outputs, stepper1_outputs, passes
):
    stepper0 = MockStepper(*stepper0_outputs)
    stepper1 = MockStepper(*stepper1_outputs)
    combined_stepper = CombinedStepper([stepper0, stepper1])
    if passes:
        assert combined_stepper._verified_no_collisions is False
        combined_stepper(0, 0)
        assert combined_stepper._verified_no_collisions is True
    else:
        with pytest.raises(ValueError):
            combined_stepper(0, 0)


@pytest.mark.parametrize(
    "stepper_outputs, expected_outputs, passes",
    [
        pytest.param(
            [
                ({"t0": da}, {"d0": da}, {"s0": da}),
                ({"t1": da + 1.0}, {"d1": da + 1.0}, {}),
            ],
            ({"t0": da, "t1": da + 1.0}, {"d0": da, "d1": da + 1.0}, {"s0": da}),
            True,
            id="combine_2",
        ),
        pytest.param(
            [
                ({"t0": da}, {"d0": da}, {"s0": da}),
                ({"t1": da + 1.0}, {"d1": da + 1.0}, {}),
                ({}, {}, {"s2": da + 2.0}),
            ],
            (
                {"t0": da, "t1": da + 1.0},
                {"d0": da, "d1": da + 1.0},
                {"s0": da, "s2": da + 2.0},
            ),
            True,
            id="combine_3",
        ),
        pytest.param(
            [
                ({"t0": da}, {"d0": da}, {"s0": da}),
                ({"t1": da + 1.0}, {"d1": da + 1.0}, {}),
                ({}, {}, {"s0": da + 2.0}),
            ],
            (),
            False,
            id="fail_on_collision",
        ),
    ],
)
def test_CombinedStepper_merged_output(stepper_outputs, expected_outputs, passes):
    steppers = [MockStepper(*outputs) for outputs in stepper_outputs]
    combined_stepper = CombinedStepper(steppers)
    if passes:
        combined_outputs = combined_stepper(0, 0)
        for combined, expected in zip(combined_outputs, expected_outputs):
            assert set(combined) == set(expected)
            for k, v in combined.items():
                xr.testing.assert_identical(expected[k], v)
    else:
        with pytest.raises(ValueError):
            combined_stepper(0, 0)
