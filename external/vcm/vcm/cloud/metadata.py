import dataclasses
import json
from typing import Mapping, List


@dataclasses.dataclass
class StepMetadata:
    job_type: str
    commit: str
    url: str
    dependencies: Mapping[str, str] = dataclasses.field(default_factory=dict)
    args: List[str] = dataclasses.field(default_factory=list)

    def print_json(self):
        print(json.dumps(dataclasses.asdict(self)))
