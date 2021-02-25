import yaml
import numpy as np
from typing import Sequence, TextIO, Mapping, Any


class ArrayStacker:
    """
    Class for stacking arrays by the feature dimension. Also handles
    subselecting along the feature dimension and returning to a full
    feature space.
    """
    
    def __init__(
        self,
        variables_to_stack: Sequence[str],
        stacked_feature_sizes: Mapping[str, int],
        original_feature_sizes: Mapping[str, int],
        feature_subselections: Mapping[str, slice] = None,
    ):
        self._variables = variables_to_stack
        self._orig_feature_sizes = original_feature_sizes
        self._stacked_feature_sizes = stacked_feature_sizes
        
        if feature_subselections is None:
            feature_subselections = {}

        self._feature_subselections = {
            name: feature_subselections.get(name, slice(None))
            for name in self._variables
        }
            
        start = end = 0
        self._stacked_feature_indices = {}
        for name in self._variables:
            end = start + self._stacked_feature_sizes[name]
            self._stacked_feature_indices[name] = slice(start, end)
            start = end
 
        self._feature_size = end
        
    def stack(self, dataset: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.hstack([
            dataset[varname][..., self._feature_subselections[varname]]
            for varname in self._variables
        ])
    
    def unstack(self, array):
        unstacked = {}
        for varname in self._variables:
            index_select = self._stacked_feature_indices[varname]
            unstacked[varname] = array[..., index_select]
        return unstacked

    def unstack_orig_featuresize(self, array):
        unstacked = self.unstack(array)

        full_size_data = {}
        for varname, data in unstacked.items():
            subselection = self._feature_subselections[varname]
            orig_feature_size = self._orig_feature_sizes[varname]
            original = _to_original_shape(data, orig_feature_size, subselection)
            full_size_data[varname] = original

        return full_size_data
    
    @property
    def feature_size(self):
        return self._feature_size

    @classmethod
    def from_data(cls, dataset, variables_to_stack, feature_subselections=None):
        init_kwargs = _get_ArrayStacker_args(
            variables_to_stack,
            dataset,
            subselections=feature_subselections
        )
        return cls(**init_kwargs)

    def dump(self, out: TextIO):
        return yaml.dump(
            {
                "variables_to_stack": self._variables,
                "stacked_feature_sizes": self._stacked_feature_sizes,
                "original_feature_sizes": self._orig_feature_sizes,
                "feature_subselections": self._feature_subselections
            },
            out
        )

    @classmethod
    def load(cls, source: TextIO):
        kwargs = yaml.unsafe_load(source.read())
        return cls(**kwargs)


def _to_original_shape(data, orig_feature_size, subselection):
    """
    Return all features to original size
    """
    current_feature_size = data.shape[-1]

    if current_feature_size != orig_feature_size:
        new_shape = list(data.shape)[:-1] + [orig_feature_size]
        full_data = np.zeros(new_shape, dtype=data.dtype)
        full_data[..., subselection] = data
    else:
        full_data = data

    return full_data


def _get_ArrayStacker_args(
    variable_names: Sequence[str],
    dataset: Mapping[str, np.ndarray],
    subselections: Mapping[str, slice],
) -> Mapping[str, Any]:

    full_sizes = {}
    stacked_sizes = {}

    for name in variable_names:
        data = dataset[name]
        subselection = subselections.get(name, slice(None))

        full_sizes[name] = data.shape[-1]
        stacked_sizes[name] = data[..., subselection].shape[-1]

    return dict(
        variables_to_stack=variable_names,
        original_feature_sizes=full_sizes,
        stacked_feature_sizes=stacked_sizes,
        feature_subselections=subselections,
    )
