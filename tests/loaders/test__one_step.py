import os
import pytest
import synth

from fv3net.regression.loaders._one_step import open_one_step


def test_open_onestep_mapping(datadir):
    with open(os.path.join(datadir, "one_step_zarr_schema.json")) as f:
        schema = synth.load(f)
    one_step_dataset = synth.generate(schema)
    timesteps = [f"2020050{i}.000000" for i in range(4)]
    for i, timestep in enumerate(timesteps):
        one_step_dataset = one_step_dataset.assign(
            {"test_var": one_step_dataset.area * 0 + i}
        )
        output_path = os.path.join(datadir, f"{timestep}.zarr")
        one_step_dataset.to_zarr(output_path, consolidated=True)
    timestep_mapper = open_one_step(str(datadir))
    assert set(timestep_mapper.keys()) == set(timesteps)
    assert timestep_mapper[timesteps[2]]["test_var"].values == pytest.approx(2)
