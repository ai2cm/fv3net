import sys

sys.path.insert(0, "../argo")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402


name = "gscond-routed-reg-v4-prog-v1"
image = "1297c2a9c61fa50084002fdb07261e8f1afbe131"


config = load_yaml("../configs/gscond-only.yaml")
job = PrognosticJob(name=name, image_tag=image, config=config,)
jobs = [job]
submit_jobs(jobs, f"routed-regression")
