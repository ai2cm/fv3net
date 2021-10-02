from pathlib import Path
from typing import Optional
from runtime.segmented_run.prepare_config import to_fv3config
import fv3config
import fv3kube
import yaml


fv3config_path = Path(__file__).parent / "fv3config.yml"
config_root = Path(__file__).parent.parent / "config"
suite_root = config_root / "suites"


def get_config(suite: Optional[str], diagnostics: Optional[Path]):
    suite = suite or "edmf-zhaocarr"
    with open(suite_root / f"{suite}.yaml") as f:
        suite_config = yaml.safe_load(f)

    if diagnostics:
        with diagnostics.open() as f:
            diag_updates = yaml.safe_load(f)
    else:
        diag_updates = {}

    config = suite_config.copy()
    config["namelist"]["coupler_nml"] = diag_updates["namelist"]["coupler_nml"]
    config["namelist"]["diag_manager_nml"] = diag_updates["namelist"][
        "diag_manager_nml"
    ]
    config["fortran_diagnostics"] = diag_updates["fortran_diagnostics"]
    config["diagnostics"] = diag_updates["diagnostics"]

    restart_url = "gs://vcm-ml-experiments/online-emulator/2021-08-09/gfs-initialized-baseline-08/fv3gfs_run/artifacts/20160801.000000/RESTART"  # noqa

    dict_ = fv3kube.merge_fv3config_overlays(
        fv3config.enable_restart(config, restart_url)
    )

    return to_fv3config(dict_, nudging_url="")
