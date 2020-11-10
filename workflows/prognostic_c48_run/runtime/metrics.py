import jsonschema
import json
import os

_metrics_file = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "metrics-schema.json"
)
with open(_metrics_file) as f:
    SCHEMA = json.load(f)


def validate(obj):
    return jsonschema.validate(obj, SCHEMA)
