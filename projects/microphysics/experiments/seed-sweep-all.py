import sys
from itertools import product

sys.path.insert(0, "../scripts")

from end_to_end import (
    PrognosticJob,
    load_yaml,
    submit_jobs,
)  # noqa: E402


init_month_paths = {
    "jul": "gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/create-training-microphysics-v4-2-6/artifacts/20160601.000000/RESTART",  # noqa
    "oct": "gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/create-training-microphysics-v4-2-9/artifacts/20160901.000000/RESTART",  # noqa
    "nov": "gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/create-training-microphysics-v4-2-10/artifacts/20161001.000000/RESTART",  # noqa
}

gscond_seed_urls = {
    i: f"gs://vcm-ml-experiments/microphysics-emulation/2023-02-10/zc-train-gscond-normfix-seed{i}-v1/model.tf"  # noqa
    for i in range(5)
}
gscond_seed_urls[
    5
] = "gs://vcm-ml-experiments/microphysics-emulation/2023-02-06/gscond-only-qvout-norm-fix-v1/model.tf"  # noqa

BASE = "gs://vcm-ml-experiments/microphysics-emulation/2022-12-16/zc-train-{model}-seed{i}-v2/model.tf"  # noqa
classify_seed_urls = {i: BASE.format(model="gscond-classify", i=i) for i in range(1, 6)}
classify_seed_urls[
    0
] = "gs://vcm-ml-experiments/microphysics-emulation/2022-10-08/gscond-only-classifier-rh-in-v1/model.tf"  # noqa

precpd_seed_urls = {i: BASE.format(model="precpd", i=i) for i in range(1, 6)}
precpd_seed_urls[
    0
] = "gs://vcm-ml-experiments/microphysics-emulation/2022-10-08/precpd-diff-only-press-rh-in-v1/model.tf"  # noqa


NAME = "zc-emu-seedsweep-gsc{}-cls{}-prpd{}-{}-v2"

# full sweep, full product
# jul, fixed sweep for pr and classifier
# oct/nov fixed sweep for gscond


def get_job(init_month: str, gsc_seed: int, cls_seed: int, prpd_seed: int):

    cfg = load_yaml("../configs/gscond-and-precpd.yaml")["config"]
    cfg["duration"] = "30d"

    cfg["initial_conditions"] = init_month_paths[init_month]
    zc_config = cfg["zhao_carr_emulation"]
    zc_config["gscond"]["path"] = gscond_seed_urls[gsc_seed]
    zc_config["gscond"]["classifier_path"] = classify_seed_urls[cls_seed]
    zc_config["model"]["path"] = precpd_seed_urls[prpd_seed]

    return PrognosticJob(
        config=cfg,
        image_tag="95218aec9ed1e6cbda0db1e1f69dc34e0d839c52",
        name=NAME.format(gsc_seed, cls_seed, prpd_seed, init_month),
    )


# full sweep, full product
# full_sweep = [get_job(*args) for args in product(init_month_paths.keys(), range(6), range(6), range(6))] # noqa
# submit_jobs(full_sweep, experiment_name="full-sweep-sensitivity")

# jul, fixed sweep for pr and classifier
args_set = set(product(["jul"], [1], range(6), [5]))
args_set.update(product(["jul"], [1], [5], range(6)))
jul_fixed = [get_job(*args) for args in sorted(args_set)]
submit_jobs(jul_fixed, experiment_name="july-fixed-sweep")

# oct/nov fixed sweep for gscond
oct_nov_args_set = set(product(["oct", "nov"], range(6), [5], [5]))
gscond_sweep = [get_job(*args) for args in sorted(oct_nov_args_set)]
submit_jobs(gscond_sweep, experiment_name="oct-nov-gscond-sweep")
