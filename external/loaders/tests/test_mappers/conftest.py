from synth import (  # noqa: F401
    dataset_fixtures_dir,
    data_source_name,
    one_step_dataset_path,
    nudging_dataset_path,
    fine_res_dataset_path,
    data_source_path,
    C48_SHiELD_diags_dataset_path,
    grid_dataset,
)

# test mapper fixtures
from ._mapper_fixtures import (  # noqa: F401
    training_mapper_name,
    training_mapper_data_source_path,
    training_mapper,
    diagnostic_mapper_name,
    diagnostic_mapper_data_source_path,
    diagnostic_mapper_helper_function,
    diagnostic_mapper_helper_function_kwargs,
    diagnostic_mapper,
)

from ._diagnostic_fixtures import (  # noqa: F401
    diagnostic_mapper_name,
    diagnostic_mapper_data_source_path,
    diagnostic_mapper,
)
