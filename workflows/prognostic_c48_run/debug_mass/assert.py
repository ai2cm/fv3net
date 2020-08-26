import sys
import json
import numpy


data_lines = []
with open(sys.argv[1]) as f:
    for line in f:
        start = "INFO:root:python:"
        if line.startswith(start):
            data_lines.append(json.loads(line[len(start) :]))


for metric in data_lines:
    numpy.testing.assert_allclose(
        metric["vapor_mass_change"]["value"], metric["total_mass_change"]["value"],
        atol=1e-2
    )

