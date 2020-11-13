import yaml
import numpy as np
from collections import Counter
from typing import Mapping, Sequence


def feature_size_to_slices(feature_sizes: Mapping[str, int]):
    """
    feature_sizes: ordered mapping of feature dimension size for each variable
    """
    slices = {}
    start = 0
    for name, size in feature_sizes.items():
        end = start + size
        slices[name] = slice(start, end)
        start = end

    return slices


def _split_on_tracer_dim(name: str, arr: np.ndarray):

    split = {}
    for i in range(arr.shape[0]):
        split[f"{name}_{i}"] = arr[i]

    return split


def split_tracer_fields(state):
    """split fields with more than two dimensions along leading axis"""

    to_remove = []
    split_update = {}
    for name, arr in state.items():
        if arr.ndim == 3:
            split_fields = _split_on_tracer_dim(name, arr)
            split_update.update(split_fields)
            to_remove.append(name)
    
    for name in to_remove:
        state.pop(name)  # remove original variable
    state.update(split_update)

    return state


def _detect_tracer_field_names(state):
    tracer_info = {}

    # look for count of leading names e.g., q1_input
    prefixes = ["_".join(name.split("_")[:2]) for name in state]
    counts = Counter(prefixes)
    for prefix, count in counts.items():
        if count > 1:
            tracer_names = [f"{prefix}_{i}" for i in range(count)]
            for name in tracer_names:
                if name not in state:
                    raise KeyError(f"Non-standard tracer naming convention... {name}")
            tracer_info[prefix] = tracer_names

    return tracer_info


def consolidate_tracers(state, tracer_field_info):

    for var_name, tracer_names in tracer_field_info.items():
        arrs = [state.pop(tname) for tname in tracer_names]
        combined = np.stack(arrs, axis=0)
        state[var_name] = combined

    return state


def _check_required_vars(state, pack_names):

    for name in pack_names:
        if name not in state:
            raise ValueError(
                f"Missing required input variable ({name}) in fortran state."
            )


class EmuArrayPacker:
    def __init__(
        self, pack_names: Sequence[str], feature_sizes: Mapping[str, int], dims=None
    ):

        ordered_features = {name: feature_sizes[name] for name in pack_names}
        self._pack_names = pack_names
        self._total_feature_size = np.sum([size for size in feature_sizes.values()])
        self._slices = feature_size_to_slices(ordered_features)
        self._tracer_fields = None

    @classmethod
    def from_packer_json(cls, packer_yaml: str):
        """Initialize an EmuArrayPacker from ArrayPacker JSON file"""
        with open(packer_yaml, "r") as f:
            packer_args = yaml.safe_load(f)
        return cls(**packer_args)

    def to_array(self, state: Mapping[str, np.ndarray]) -> np.ndarray:
        _check_required_vars(state, self._pack_names)

        # sample is always trailing from fortran
        n_samples = state[list(state.keys())[0]].shape[-1]

        arr = np.empty((n_samples, self._total_feature_size))
        for name in self._pack_names:
            var_arr = state[name]
            var_slice = self._slices[name]
            arr[:, var_slice] = var_arr.T  # switch to sample leading

        return arr

    def to_dict(self, arr: np.ndarray) -> Mapping[str, np.ndarray]:

        separated = {}
        for name, var_slice in self._slices.items():

            var_arr = arr[:, var_slice]
            # switch back to feature_leading
            separated[name] = var_arr.T

        return separated
