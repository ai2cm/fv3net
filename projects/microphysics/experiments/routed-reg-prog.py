import sys

sys.path.insert(0, "../argo")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402


name = "gscond-routed-reg-v3-prog-v6"
image = "7250d75ac38bfc4c163d33371eb7428a4afd341f"


config = load_yaml("../configs/gscond-only.yaml")
job = PrognosticJob(name=name, image_tag=image, config=config,)
jobs = [job]
submit_jobs(jobs, f"routed-regression")
