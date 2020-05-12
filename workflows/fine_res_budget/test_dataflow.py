import dataflow
import joblib


def test_budget():
    data = joblib.load("2016-08-01 00:22:30/0.nc")
    b = dataflow.budget(data)
    print(b)