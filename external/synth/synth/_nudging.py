from dataclasses import dataclass
from synth.core import DatasetSchema


@dataclass
class NudgingDataset:
    tendencies_schema: DatasetSchema
    after_dynamics_schema: DatasetSchema
    after_physics_schema: DatasetSchema

    def generate(self, directory: str):
        nudging_dir = directory
        nudging_after_dynamics_zarrpath = os.path.join(
            nudging_dir, "after_dynamics.zarr"
        )
        nudging_after_dynamics_dataset = generate(
            nudging_after_dynamics_schema
        ).assign_coords(
            {
                "time": [
                    np.datetime64(timestep1_npdatetime_fmt),
                    np.datetime64(timestep2_npdatetime_fmt),
                ]
            }
        )
        nudging_after_dynamics_dataset.to_zarr(
            nudging_after_dynamics_zarrpath, consolidated=True
        )

        nudging_after_physics_zarrpath = os.path.join(nudging_dir, "after_physics.zarr")
        nudging_after_physics_dataset = generate(
            nudging_after_physics_schema
        ).assign_coords(
            {
                "time": [
                    np.datetime64(timestep1_npdatetime_fmt),
                    np.datetime64(timestep2_npdatetime_fmt),
                ]
            }
        )
        nudging_after_physics_dataset.to_zarr(
            nudging_after_physics_zarrpath, consolidated=True
        )

        nudging_tendencies_zarrpath = os.path.join(
            nudging_dir, "nudging_tendencies.zarr"
        )
        nudging_tendencies_dataset = generate(nudging_tendencies_schema).assign_coords(
            {
                "time": [
                    np.datetime64(timestep1_npdatetime_fmt),
                    np.datetime64(timestep2_npdatetime_fmt),
                ]
            }
        )
        nudging_tendencies_dataset.to_zarr(
            nudging_tendencies_zarrpath, consolidated=True
        )

    @classmethod
    def from_datadir(cls, datadir: str):
        with open(os.path.join(datadir, "after_dynamics.json")) as f:
            after_dynamics_schema = load(f)
        with open(os.path.join(datadir, "after_physics.json")) as f:
            after_physics_schema = load(f)
        with open(os.path.join(datadir, "nudging_tendencies.json")) as f:
            tendencies_schema = load(f)

        return cls(tendencies_schema, after_dynamics_schema, after_physics_schema)
