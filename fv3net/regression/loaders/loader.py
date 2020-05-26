def load




class BatchLoader():
    def __init__(self, data_path):
        self.data_path = data_path



class OnestepMapper:
    def __init__(self, timesteps_dir):
        self._timesteps_dir = timesteps_dir
        self._fs = cloud.get_fs(timesteps_dir)
        self.zarrs = self._fs.glob(os.path.join(timesteps_dir, "*.zarr"))
        if len(self.zarrs) == 0:
            raise ValueError(f"No zarrs found in {timesteps_dir}")
            
    def __getitem__(self, key: str) -> xr.Dataset:
        zarr_path = os.path.join(self._timesteps_dir, f"{key}.zarr")
        mapper = self._fs.get_mapper(zarr_path)
        consolidated = True if ".zmetadata" in mapper else False
        return xr.open_zarr(self._fs.get_mapper(zarr_path), consolidated=consolidated)

    def keys(self):
        return [vcm.parse_timestep_str_from_path(zarr) for zarr in self.zarrs]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


def _load_batches(
    data_path: str,
    *variable_names: Iterable[str],
    files_per_batch: int = 1,
    num_batches: int = None,
    random_seed: int = 1234,
    mask_to_surface_type: str = None,
    init_time_dim_name: str = "initial_time",
    rename_variables: Mapping[str, str] = None,
    loader_function,
) -> Sequence:
    """Get a sequence of batches from one-step zarr stores.

    Args:
        data_path: location of directory containing zarr stores
        *variable_names: any number of sequences of variable names. One Sequence will be
            returned for each of the given sequences. The "sample" dimension will be
            identical across each of these sequences.
        files_per_batch: number of zarr stores used to create each batch, defaults to 1
        num_batches (optional): number of batches to create. By default, use all the
            available training data.
        random_seed (optional): seed value for random number generator
        mask_to_surface_type: mask data points to ony include the indicated surface type
        init_time_dim_name: name of the initialization time dimension
        rename_variables: mapping of variables to rename,
            from data names to standard names
    """
    if rename_variables is None:
        rename_variables = {}
    if len(variable_names) == 0:
        raise TypeError("At least one value must be given for variable_names")
    logger.info(f"Reading data from {data_path}.")
    
    timestep_mapper = TimestepMapper(data_path)
    timesteps = timestep_mapper.keys()

    logger.info(f"Number of .zarrs in GCS train data dir: {len(timestep_mapper)}.")
    random = np.random.RandomState(random_seed)
    random.shuffle(timesteps)
    num_batches = _validated_num_batches(len(timesteps), files_per_batch, num_batches)
    logger.info(f"{num_batches} data batches generated for model training.")
    timesteps_list_sequence = list(
        timesteps[batch_num * files_per_batch : (batch_num + 1) * files_per_batch]
        for batch_num in range(num_batches)
    )
    output_list = []
    for data_vars in variable_names:
        load_batch = functools.partial(
            loader_function,
            timestep_mapper,
            data_vars,
            rename_variables,
            init_time_dim_name,
        )
        input_formatted_batch = functools.partial(
            stack_and_format,
            init_time_dim_name,
            mask_to_surface_type,
            copy.deepcopy(random),  # each sequence must be shuffled the same!
        )
        output_list.append(
            FunctionOutputSequence(
                lambda x: input_formatted_batch(load_batch(x)),
                timesteps_list_sequence))
    if len(output_list) > 1:
        return tuple(output_list)
    else:
        return output_list[0]