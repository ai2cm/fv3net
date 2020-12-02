import pytest
import os
import synth
from loaders import TIME_NAME
from loaders.mappers._nudged._nudged import open_nudge_to_fine, open_nudge_to_obs

NTIMES = 12
                        

@pytest.fixture(scope="module")
def state_before_nudging_schema(datadir_module):
    nudge_data_schema = datadir_module.join(f"state_before_nudging.json")
    with open(nudge_data_schema) as f:
        schema = synth.load(f)
        
    data = synth.generate(schema)

    return data.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def nudge_to_fine_tendencies(datadir_module):
    nudge_to_fine_tendencies_schema = datadir_module.join(f"nudge_to_fine_tendencies.json")
    with open(nudge_to_fine_tendencies_schema) as f:
        schema = synth.load(f)
        
    nudging_tend = synth.generate(schema)

    return nudging_tend.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def physics_tendencies(datadir_module):
    physics_tendencies_schema = datadir_module.join(f"physics_tendencies.json")
    with open(physics_tendencies_schema) as f:
        schema = synth.load(f)
        
    physics_tendencies = synth.generate(schema)

    return physics_tendencies.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def nudge_to_fine_data_dir(
    datadir_module, state_before_nudging_schema, physics_tendencies, nudge_to_fine_tendencies
):
    all_data = {"physics_tendencies": physics_tendencies}
    all_data.update({"nudge_to_fine_tendencies": nudge_to_fine_tendencies})
    all_data.update({"state_before_nudging": state_before_nudging_schema})

    for filestem, ds in all_data.items():
        filepath = os.path.join(datadir_module, f"{filestem}.zarr")
        ds.to_zarr(filepath)

    return str(datadir_module)


@pytest.mark.regression
def test_open_nudge_to_fine(nudge_to_fine_data_dir):

    mapper = open_nudge_to_fine(nudge_to_fine_data_dir, consolidated=False)

    key = list(mapper.keys())[0]
    mapper[key]["air_temperature"]
    mapper[key]["specific_humidity"]
    mapper[key]["dQ1"]
    mapper[key]["dQ2"]
    mapper[key]["dQxwind"]
    mapper[key]["dQywind"]
    mapper[key]["pQ1"]
    mapper[key]["pQ2"]


# @pytest.mark.regression
# def test_open_nudge_to_obs(nudged_data_dir):

#     merge_files = ("prognostic_diags.zarr", "nudging_tendencies.zarr")
#     rename_vars = {
#         "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
#         "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
#         "air_temperature_tendency_due_to_nudging": "dQ1",
#         "specific_humidity_tendency_due_to_nudging": "dQ2",
#         "x_wind_tendency_due_to_nudging": "dQxwind",
#         "y_wind_tendency_due_to_nudging": "dQywind",
#     }
#     mapper = open_nudge_to_obs(
#         nudged_data_dir, merge_files=merge_files, rename_vars=rename_vars, consolidated=False
#     )
#     key = list(mapper.keys())[0]
#     mapper[key]["air_temperature"]
#     mapper[key]["specific_humidity"]
#     mapper[key]["dQ1"]
#     mapper[key]["dQ2"]
#     mapper[key]["dQxwind"]
#     mapper[key]["dQywind"]
#     mapper[key]["pQ1"]
#     mapper[key]["pQ2"]