import backoff
import functools
import logging
import numpy as np
from typing import Iterable, Sequence, Mapping, Union
import xarray as xr
from vcm import safe
from ._sequences import FunctionOutputSequence
from ._transform import transform_train_data
from ..constants import TIME_NAME
from ._one_step import TimestepMapper
from ._fine_resolution_budget import FineResolutionBudgetTiles

GenericMapper = Union[TimestepMapper, FineResolutionBudgetTiles]

logger = logging.getLogger()
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
        init_time_dim_name: str
    ):
        self._dataset_mapper = dataset_mapper
        self._generate_batches(
            files_per_batch,
            num_batches,
            random_seed,
            init_time_dim_name
        )
        
    def __getitem__(self, key: int) -> Sequence[xr.Dataset]:
        return [self._dataset_mapper[timestep] for timestep in self.batches[key]]
        
    def keys(self):
        return list(range(len(self.batches)))
    
    def _generate_batches(
        self,
        files_per_batch: int,
        num_batches: int,
        random_seed: int,
        init_time_dim_name: str
    ):
        print(random_seed)
        print(type(random_seed))
        all_timesteps = list(self._dataset_mapper.keys())
        random = np.random.RandomState(random_seed)
        random.shuffle(all_timesteps)
        num_batches = self._validated_num_batches(
            len(all_timesteps),
            files_per_batch,
            num_batches
        )
        logger.info(f"{num_batches} data batches generated for model training.")
        timesteps_list_sequence = list(
            all_timesteps[batch_num * files_per_batch : (batch_num + 1) * files_per_batch]
            for batch_num in range(num_batches)
        )
        
        self.batches = timesteps_list_sequence
        
    def _validated_num_batches(
        self,
        total_num_input_files: int,
        files_per_batch: int,
        num_batches: int
    ):
        """ check that the number of batches (if provided) and the number of
        files per batch are reasonable given the number of zarrs in the input data dir.

        Returns:
            Number of batches to use for training
        """
        if num_batches is None:
            if total_num_input_files >= files_per_batch:
                num_train_batches = total_num_input_files // files_per_batch
            else:
                raise ValueError(
                    f"Number of input_files {total_num_input_files} "
                    f"must be greater than files_per_batch {files_per_batch}"
                )
        elif num_batches * files_per_batch > total_num_input_files:
            raise ValueError(
                f"Number of input_files {total_num_input_files} "
                f"cannot create {num_batches} batches of size {files_per_batch}."
            )
        else:
            num_train_batches = num_batches
        return num_train_batches


def load_batches(
    data_mapping: Mapping[str, xr.Dataset],
    *variable_names: Iterable[str],
    files_per_batch: int = 1,
    num_batches: int = None,
    random_seed: int = 0,
    init_time_dim_name: str = "initial_time",
    rename_variables: Mapping[str, str] = None,
):
    if rename_variables is None:
        rename_variables = {}
    if len(variable_names) == 0:
        raise TypeError("At least one value must be given for variable_names")
        
    batch_mapper = BatchMapper(data_mapping, files_per_batch, num_batches, random_seed, init_time_dim_name)
#     batched_timesteps = _select_batch_timesteps(
#         list(data_mapping.keys()), files_per_batch, num_batches, random_seed
#     )
    transform = functools.partial(transform_train_data, init_time_dim_name, random_seed)
    output_list = []
    for data_vars in variable_names:
        load_batch = functools.partial(
            _load_batch, batch_mapper, data_vars, rename_variables, init_time_dim_name,
        )
        batch_func = _compose(transform, load_batch)
        output_list.append(FunctionOutputSequence(batch_func, batch_mapper.batches))
    if len(output_list) > 1:
        return tuple(output_list)
    else:
        return output_list[0]


def _compose(outer_func, inner_func):
    return lambda x: outer_func(inner_func(x))


def _load_batch(
    timestep_mapper,
    data_vars: Iterable[str],
    rename_variables: Mapping[str, str],
    init_time_dim_name: str,
    timestep_list: Iterable[str],
):
    print(timestep_list)
    data = _load_datasets(timestep_mapper, timestep_list)
    print(data)
    ds = xr.concat(data, init_time_dim_name)
    print(ds)
    # need to use standardized time dimension name
    rename_variables[init_time_dim_name] = rename_variables.get(
        init_time_dim_name, TIME_NAME
    )
    ds = ds.rename(rename_variables)
    print(data_vars)
    ds = safe.get_variables(ds, data_vars)
    return ds.load()


def _load_sequence(
    batch_mapper: BatchMapper,
    data_vars: Iterable[str],
    rename_variables: Mapping[str, str],
    init_time_dim_name: str,
    batch_index: int,
):
    data = batch_mapper[batch_index]
    ds = xr.concat(data, init_time_dim_name)
    # need to use standardized time dimension name
    rename_variables[init_time_dim_name] = rename_variables.get(
        init_time_dim_name, TIME_NAME
    )
    ds = ds.rename(rename_variables)
    print(data_vars)
    ds = safe.get_variables(ds, data_vars)
    return ds.load()


@backoff.on_exception(backoff.expo, (ValueError, RuntimeError), max_tries=3)
def _load_datasets(
    timestep_mapper: Mapping[str, xr.Dataset], times: Iterable[str]
) -> Iterable[xr.Dataset]:
    return_list = []
    for time in times:
        ds = timestep_mapper[time]
        return_list.append(ds)
    return return_list


def load_sequence_for_diagnostics(
    dataset_mapper: GenericMapper,
    variable_names: Sequence[str],
    files_per_batch: int = 1,
    num_batches: int = None,
    random_seed: int = 0,
    init_time_dim_name: str = "initial_time",
    rename_variables: Mapping[str, str] = None,
) -> Sequence[xr.Dataset]:
    '''Load a dataset sequence for dagnostic purposes. Uses the same batch subsetting as
    as load_batches but without transformation and stacking
    '''
    if rename_variables is None:
        rename_variables = {}
    
    batch_mapper = BatchMapper(
        dataset_mapper, files_per_batch, num_batches, random_seed, init_time_dim_name
    )
    
    load_sequence = functools.partial(
            _load_sequence, batch_mapper, variable_names, rename_variables, init_time_dim_name,
        )
    
    return FunctionOutputSequence(load_sequence, batch_mapper.keys())
    