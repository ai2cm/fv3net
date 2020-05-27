import os
from fv3net.pipelines.coarsen_restarts.pipeline import _coarsen_data

curr_timestep = "20160801.001500"
coarsen_factor = 8

tmpdir = "test-outputs"

local_spec_dir = os.path.join(tmpdir, "local_grid_spec")

tmp_timestep_dir = os.path.join(tmpdir, "local_fine_dir")

local_coarsen_dir = os.path.join(tmpdir, "local_coarse_dir", curr_timestep)
_coarsen_data(
    local_coarsen_dir, curr_timestep, coarsen_factor, local_spec_dir, tmp_timestep_dir,
)
