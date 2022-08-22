import dataclasses
from typing import Optional
from runtime.steppers.machine_learning import MachineLearningConfig
import radiation

# from radiation.preprocess import get_init_data


@dataclasses.dataclass
class RadiationConfig:
    scheme: str
    input_model: Optional[MachineLearningConfig] = None
    offline: bool = True


def get_radiation_driver(
    config: Optional[RadiationConfig],
    forcing_dir: str = ".",
    fv_core_dir: str = "./INPUT/fv_core.res.nc",
    static_config_path: str = "./rad_static_config.yaml",
) -> Optional[radiation.RadiationDriver]:
    if config and config.scheme == "python":
        driver = radiation.RadiationDriver()
    else:
        driver = None
    #     init_data = get_init_data(
    #         forcing_dir = forcing_dir,
    # fv_core_path = fv_core_path, config_path = config_path
    #     )
    #     return RadiationDriver().radinit(*init_data)
    return driver
