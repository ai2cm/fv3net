import sys

sys.path.insert(0, "../argo")

from end_to_end import EndToEndJob, load_yaml, submit_jobs  # noqa: E402

run_config = load_yaml("../configs/default.yaml")
ml_config = load_yaml("../train/rnn-v4-precpd-limit.yaml")
ml_config["nfiles"] = 3240  # same number of files as v3 data

job = EndToEndJob(
    name="precpd-lim-squash-v4data",
    ml_config=ml_config,
    prog_config=run_config,
    fv3fit_image_tag="latest",
    image_tag="latest",
)

submit_jobs([job], "precpd-limit-squash-optimized")
