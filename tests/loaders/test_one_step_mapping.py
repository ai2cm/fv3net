import os
import pytest
import synth

from fv3net.regression.loaders._one_step import open_onestep_mapping


ONE_STEP_ZARR_SCHEMA = "one_step_zarr_schema.json"
TIMESTEP_LIST = [f"2020050{i}.000000" for i in range(4)]


def test_open_onestep_mapping(datadir):
    with open(os.path.join(datadir, ONE_STEP_ZARR_SCHEMA)) as f:
        schema = synth.load(f)
    one_step_dataset = synth.generate(schema)
    for i, timestep in enumerate(TIMESTEP_LIST):
        one_step_dataset = one_step_dataset.assign(
            {"test_var": one_step_dataset.area * 0 + i}
        )
        output_path = os.path.join(datadir, f"{timestep}.zarr")
        one_step_dataset.to_zarr(output_path, consolidated=True)
    timestep_mapper = open_onestep_mapping(str(datadir))
    assert set(timestep_mapper.keys()) == set(TIMESTEP_LIST)
    assert timestep_mapper[TIMESTEP_LIST[2]]["test_var"].values == pytest.approx(2)
