import dataflow
import joblib
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline


def test_budget():
    path = "2016-08-01 00:22:30/0.nc"

    with TestPipeline() as p:

        data = p | beam.Create([path]) | "Load" >> beam.Map(joblib.load)
        area = data | "Area" >> beam.Map(lambda x: x[1])
        kv = data | "KV" >> beam.Map(lambda x: x[0])

        kv | "Compute Budget" >> beam.Map(dataflow.budget, beam.pvalue.AsSingleton(area))