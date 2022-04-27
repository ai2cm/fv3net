import dataclasses
import json
import os
from typing import Any, Mapping, List, Optional


@dataclasses.dataclass
class StepMetadata:
    job_type: str
    url: str
    commit: Optional[str] = os.getenv("COMMIT_SHA")
    dependencies: Optional[Mapping[str, str]] = None
    args: Optional[List[str]] = None
    env_vars: Optional[Mapping[str, str]] = None

    def print_json(self):
        print(json.dumps({"step_metadata": dataclasses.asdict(self)}))


def log_fact_json(
    data: Mapping[str, Any],
    kind: str = "metrics",
    labels: Optional[Mapping[str, str]] = None,
) -> None:

    payload = dict(json=data)

    labels = {} if labels is None else labels
    payload["logging.googleapis.com/labels"] = dict(kind=kind, **labels)

    print(json.dumps(payload))
