import pytest
import numpy as np
import vcm


def test_get_data():

    try:
        ds = vcm.cdl_to_dataset(
            """
        netcdf Some Data {
        dimensions:
            time = 3;
            x = 4;
        variables:
            int time(time);
            double a(time, x);
                a:_FillValue = 0;

        data:
            time = 1,2,3;
        }
        """
        )
    except FileNotFoundError:
        pytest.xfail("ncgen is not installed.")

    assert ds["a"].dims == ("time", "x")
    assert np.all(np.isnan(ds["a"]))
    assert ds["time"].values.tolist() == [1, 2, 3]
