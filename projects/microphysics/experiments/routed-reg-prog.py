import sys

sys.path.insert(0, "../argo")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402


name = "gscond-routed-reg-v3-prog-v3"
image = "3cf424023bf215652f46eaacd0737d834dff9e77"


config = load_yaml("../configs/gscond-only.yaml")
job = PrognosticJob(name=name, image_tag=image, config=config,)
jobs = [job]
submit_jobs(jobs, f"routed-regression")
