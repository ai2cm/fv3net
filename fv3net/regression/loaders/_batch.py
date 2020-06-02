import functools
import logging
import numpy as np
from numpy.random import RandomState
from typing import Iterable, Sequence, Mapping, Union
import xarray as xr
from vcm import safe
from ._sequences import FunctionOutputSequence
from ._transform import stack_dropnan_shuffle
from ..constants import TIME_NAME
from ._one_step import TimestepMapper
from ._fine_resolution_budget import FineResolutionBudgetTiles

GenericMapper = Union[TimestepMapper, FineResolutionBudgetTiles]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FV3OutMapper:
    def __init__(self, *args):
        raise NotImplementedError("Don't use the base class!")

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key: str) -> xr.Dataset:
        raise NotImplementedError()

    def keys(self):
        raise NotImplementedError()


class BatchMapper(FV3OutMapper):
    def __init__(
        self,
        dataset_mapper: GenericMapper,
        files_per_batch: int,
        num_batches: int,
        random_seed: int,
        init_time_dim_name: str,
    ):
        self._dataset_mapper = dataset_mapper
        self._generate_batches(files_per_batch, num_batches, random_seed)

    def __getitem__(self, key: int) -> Sequence[xr.Dataset]:
        return [self._dataset_mapper[timestep] for timestep in self.batches[key]]

    def keys(self):
        return list(range(len(self.batches)))

    def _generate_batches(
        self, timesteps_per_batch: int, num_batches: int, random_state: RandomState
    ) -> Sequence[Sequence[str]]:

        all_timesteps = list(self._dataset_mapper.keys())
        random_state.shuffle(all_timesteps)
        num_batches = self._validated_num_batches(
            len(all_timesteps), timesteps_per_batch, num_batches
        )
        logger.info(f"{num_batches} data batches generated.")
        timesteps_list_sequence = list(
            all_timesteps[
                batch_num * timesteps_per_batch : (batch_num + 1) * timesteps_per_batch
            ]
            for batch_num in range(num_batches)
        )

        self.batches = timesteps_list_sequence

    def _validated_num_batches(
        self, total_num_input_times, timesteps_per_batch, num_batches=None
    ):
        """ check that the number of batches (if provided) and the number of
        timesteps per batch are reasonable given the number of zarrs in the
        input data dir.

        Returns:
            Number of batches to use for training
        """
        if any(arg <= 0 for arg in [total_num_input_times, timesteps_per_batch]):
            raise ValueError(
                f"Total number of input times {total_num_input_times}, "
                f"timesteps per batch {timesteps_per_batch}"
            )
        if num_batches is not None and num_batches <= 0:
            raise ValueError(f"num batches {num_batches} cannot be 0 or negative.")
        if num_batches is None:
            if total_num_input_times >= timesteps_per_batch:
                return total_num_input_times // timesteps_per_batch
            else:
                raise ValueError(
                    f"Number of input_times {total_num_input_times} "
                    f"must be greater than timesteps_per_batch {timesteps_per_batch}"
                )
        elif num_batches * timesteps_per_batch > total_num_input_times:
            raise ValueError(
                f"Number of input_times {total_num_input_times} "
                f"cannot create {num_batches} batches of size {timesteps_per_batch}."
            )
        else:
            return num_batches


def mapper_to_batches(
    data_mapping: Mapping[str, xr.Dataset],
    variable_names: Iterable[str],
    timesteps_per_batch: int = 1,
    num_batches: int = None,
    random_seed: int = 0,
    init_time_dim_name: str = "initial_time",
    rename_variables: Mapping[str, str] = None,
) -> FunctionOutputSequence:
    """ The function returns a FunctionOutputSequence that is
    later iterated over in ..sklearn.train. When iterating over the
    output FunctionOutputSequence, the loading and transformation of data
    is applied to each batch, and it effectively becomes a Sequence[xr.Dataset].

    Args:
        data_mapping (Mapping[str, xr.Dataset]): Interface to select data for
            given timestep keys.
        variable_names (Iterable[str]): data variables to select
        timesteps_per_batch (int, optional): Defaults to 1.
        num_batches (int, optional): Defaults to None.
        random_seed (int, optional): Defaults to 0.
        init_time_dim_name (str, optional): Name of time dim in data source.
            Defaults to "initial_time".
        rename_variables (Mapping[str, str], optional): Defaults to None.

    Raises:
        TypeError: If no variable_names are provided to select the final datasets

    Returns:
        FunctionOutputSequence: When iterating over the returned object in
<<<<<<< HEAD
        sklearn.train, the loading and transformation functions are applied for
        each batch it is effectively used as a Sequence[xr.Dataset].
=======
        sklearn.train, the loading and transformation functions are applied
        for each batch it is effectively used as a Sequence[xr.Dataset].
>>>>>>> refactor/batch-loader
    """
    random_state = np.random.RandomState(random_seed)
    if rename_variables is None:
        rename_variables = {}
    if len(variable_names) == 0:
        raise TypeError("At least one value must be given for variable_names")

    batch_mapper = BatchMapper(
        data_mapping, timesteps_per_batch, num_batches, random_state, init_time_dim_name
    )
    transform = functools.partial(
        stack_dropnan_shuffle, init_time_dim_name, random_state
    )
    load_batch = functools.partial(
        _load_batch, batch_mapper, variable_names, rename_variables, init_time_dim_name,
    )

    return FunctionOutputSequence(
        lambda x: transform(load_batch(x)), batch_mapper.keys()
    )


def _load_batch(
    batch_mapper: Mapping[int, Sequence[xr.Dataset]],
    data_vars: Iterable[str],
    rename_variables: Mapping[str, str],
    init_time_dim_name: str,
    batch_index: int,
) -> xr.Dataset:

    data = batch_mapper[batch_index]
    ds = xr.concat(data, init_time_dim_name)

    # need to use standardized time dimension name
    rename_variables[init_time_dim_name] = rename_variables.get(
        init_time_dim_name, TIME_NAME
    )
    ds = ds.rename(rename_variables)
    ds = safe.get_variables(ds, data_vars)

    return ds


def mapper_to_diagnostic_sequence(
    dataset_mapper: GenericMapper,
    variable_names: Sequence[str],
    files_per_batch: int = 1,
    num_batches: int = None,
    random_seed: int = 0,
    init_time_dim_name: str = "initial_time",
    rename_variables: Mapping[str, str] = None,
) -> Sequence[xr.Dataset]:
    """Load a dataset sequence for dagnostic purposes. Uses the same batch subsetting as
    as mapper_to_batch but without transformation and stacking
    """
    random_state = np.random.RandomState(random_seed)
    if rename_variables is None:
        rename_variables = {}

    batch_mapper = BatchMapper(
        dataset_mapper, files_per_batch, num_batches, random_state, init_time_dim_name
    )

    load_sequence = functools.partial(
        _load_batch, batch_mapper, variable_names, rename_variables, init_time_dim_name,
    )

    return FunctionOutputSequence(load_sequence, batch_mapper.keys())
