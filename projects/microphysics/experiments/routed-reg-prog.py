import sys

sys.path.insert(0, "../argo")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402


name = "gscond-routed-reg-v3-prog-v5"
image = "2fec4c64a1326374b98f755e384e98b487ef7c2f"


config = load_yaml("../configs/gscond-only.yaml")
job = PrognosticJob(name=name, image_tag=image, config=config,)
jobs = [job]
submit_jobs(jobs, f"routed-regression")
