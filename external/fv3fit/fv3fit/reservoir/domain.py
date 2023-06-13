import fsspec
import numpy as np
import tensorflow as tf
from typing import Sequence, Iterable
import yaml
from ._reshaping import stack_samples
import pace.util


def slice_along_axis(arr: np.ndarray, inds: slice, axis: int = 0):
    # https://stackoverflow.com/a/37729566
    # For slicing ndarray along a dynamically specified axis
    # same as np.take() but does not make a copy of the data
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[tuple(sl)]


class RankDivider:
    def __init__(
        self,
        subdomain_layout: Sequence[int],
        rank_dims: Sequence[str],
        rank_extent: Sequence[int],  # shape of full data, including overlap
        overlap: int,
    ):
        """ Divides a rank of data into subdomains for use in training.
        Args:
            subdomain_layout: layout describing subdomain grid within the rank
                ex. [2,2] means the rank is divided into 4 subdomains
            rank_dims: order of spatial dimensions in data. Do not include time.
            rank_extent: Shape of full data. This includes any halo cells from
                overlap into neighboring ranks.
            overlap: number of cells surrounding each subdomain to include when
                taking subdomain data.

        Ex. I want to train reservoirs on 4x4 subdomains with 4 cells of overlap
        across subdomains. The data is preprocessed and saved as 1 C48 tile per rank,
        with n_halo=4. I would initialize the RankDivider as
            RankDivider(
                subdomain_layout=(12, 12),
                rank_dims=["x", "y", "z"],
                rank_extent=[ 56, 56, 79],
                overlap=4,
            )
        """
        self.subdomain_layout = subdomain_layout
        if "time" in rank_dims:
            raise ValueError(
                "Do not include time in dimension information. "
                "Only spatial dimensions (x, y, z) should be provided."
            )
        if not {"x", "y"}.issubset(rank_dims):
            raise ValueError(
                "'x' and 'y' dims must be in the rank_dims of the RankDivider"
            )
        self.rank_dims = rank_dims
        self.overlap = overlap
        self.rank_extent = rank_extent
        self.n_subdomains = subdomain_layout[0] * subdomain_layout[1]

        self._x_ind = rank_dims.index("x")
        self._y_ind = rank_dims.index("y")
        self._n_features = rank_extent[-1]

        self._partitioner = pace.util.TilePartitioner(subdomain_layout)

        # dimensions of rank data without the halo points. Useful for slice calculation.
        self._rank_extent_without_overlap = self._get_rank_extent_without_overlap(
            rank_extent, overlap
        )

    @property
    def subdomain_xy_size_without_overlap(self):
        # length of one side of subdomain along x/y axes
        return (
            self._rank_extent_without_overlap[self._x_ind] // self.subdomain_layout[0]
        )

    @property
    def n_subdomain_features(self):
        # number of total features (nx * ny * nz) in one subdomain
        return int(np.prod(self.get_subdomain_extent(with_overlap=True)))

    def get_subdomain_extent(self, with_overlap: bool):
        subdomain_xy_size = self.subdomain_xy_size_without_overlap
        if with_overlap:
            subdomain_xy_size += 2 * self.overlap

        subdomain_extent = list(self.rank_extent)
        subdomain_extent[self._x_ind] = subdomain_xy_size
        subdomain_extent[self._y_ind] = subdomain_xy_size
        return tuple(subdomain_extent)

    def subdomain_slice(self, subdomain_index: int, with_overlap: bool):
        # first get the slice indices w/o overlap points for XY data without halo,
        # then calculate adjustments when the overlap cells are included
        slice_ = list(
            self._partitioner.subtile_slice(
                rank=subdomain_index,
                global_dims=self.rank_dims,
                global_extent=self._rank_extent_without_overlap,
            )
        )
        x_slice_ = slice_[self._x_ind]
        y_slice_ = slice_[self._y_ind]

        if with_overlap:
            x_slice_updated = slice(
                x_slice_.start, x_slice_.stop + 2 * self.overlap, None
            )
            y_slice_updated = slice(
                y_slice_.start, y_slice_.stop + 2 * self.overlap, None
            )

        else:
            # The data includes the overlap on the sides of the full rank, so exclude
            # the rank halo region if retrieving the subdomains without overlap cells.
            x_slice_updated = slice(
                x_slice_.start + self.overlap, x_slice_.stop + self.overlap, None
            )
            y_slice_updated = slice(
                y_slice_.start + self.overlap, y_slice_.stop + self.overlap, None
            )

        slice_[self._x_ind] = x_slice_updated
        slice_[self._y_ind] = y_slice_updated
        return tuple(slice_)

    def _get_rank_extent_without_overlap(
        self, data_shape: Sequence[int], overlap: int
    ) -> Sequence[int]:
        extent_without_halos = list(data_shape)
        extent_without_halos[self._x_ind] = (
            extent_without_halos[self._x_ind] - 2 * overlap
        )
        extent_without_halos[self._y_ind] = (
            extent_without_halos[self._y_ind] - 2 * overlap
        )
        return extent_without_halos

    def get_subdomain_tensor_slice(
        self,
        tensor_data: tf.Tensor,
        subdomain_index: int,
        with_overlap: bool,
        data_has_time_dim: bool,
    ) -> tf.Tensor:

        subdomain_slice = self.subdomain_slice(subdomain_index, with_overlap)
        x_ind, y_ind = self._x_ind, self._y_ind
        if data_has_time_dim is True:
            subdomain_slice = (slice(None, None), *subdomain_slice)
            x_ind += 1
            y_ind += 1
        tensor_data_xsliced = slice_along_axis(
            arr=tensor_data, inds=subdomain_slice[x_ind], axis=x_ind
        )
        tensor_data_xy_sliced = slice_along_axis(
            arr=tensor_data_xsliced, inds=subdomain_slice[y_ind], axis=y_ind
        )
        return tensor_data_xy_sliced

    def unstack_subdomain(self, tensor, with_overlap: bool, data_has_time_dim: bool):
        # Takes a flattened subdomain and reshapes it back into its original
        # x and y dims
        if data_has_time_dim is True:
            unstacked_shape = self.get_subdomain_extent(with_overlap=with_overlap)
            unstacked_shape = (tensor.shape[0], *unstacked_shape)
            expected_stacked_size = np.prod(unstacked_shape[1:])
        else:
            unstacked_shape = self.get_subdomain_extent(with_overlap=with_overlap)
            expected_stacked_size = np.prod(unstacked_shape)

        if tensor.shape[-1] != expected_stacked_size:
            raise ValueError(
                f"Dimension of each stacked sample {tensor.shape[-1]} expected to be "
                f"{expected_stacked_size} (product of {unstacked_shape})."
            )

        return np.reshape(tensor, unstacked_shape)

    def flatten_subdomains_to_columns(
        self, data: tf.Tensor, with_overlap: bool, keep_first_dim: bool
    ):
        # Divide into subdomains and flatten subdomains into columns.
        # Dimensions [(time), x, y, feature_orig] -> [(time), feature_new, subdomain]
        # where feature_orig is variables at each model level, and feature_new
        # is variables at each model level and xy coord.
        subdomains_to_columns = []
        for s in range(self.n_subdomains):
            subdomain_data = self.get_subdomain_tensor_slice(
                data,
                subdomain_index=s,
                with_overlap=with_overlap,
                data_has_time_dim=keep_first_dim,
            )
            subdomains_to_columns.append(stack_samples(subdomain_data, keep_first_dim))

        # Concatentate subdomain data arrays along a new subdomain axis.
        # Dimensions are now [time, feature, submdomain]
        reshaped = np.stack(subdomains_to_columns, axis=-1)
        return reshaped

    def dump(self, path):
        metadata = {
            "subdomain_layout": self.subdomain_layout,
            "rank_dims": self.rank_dims,
            "rank_extent": self.rank_extent,
            "overlap": self.overlap,
        }
        with fsspec.open(path, "w") as f:
            f.write(yaml.dump(metadata))

    @classmethod
    def load(cls, path):
        with fsspec.open(path, "r") as f:
            metadata = yaml.safe_load(f)
        return cls(**metadata)


def assure_same_dims(variable_tensors: Iterable[tf.Tensor]) -> Iterable[tf.Tensor]:
    max_dims = max(len(v.shape) for v in variable_tensors)
    reshaped_tensors = []
    for var_data in variable_tensors:
        if len(var_data.shape) == max_dims:
            reshaped_tensors.append(var_data)
        elif len(var_data.shape) == max_dims - 1:
            orig_shape = var_data.shape
            reshaped_tensors.append(tf.reshape(var_data, shape=(*orig_shape, 1)))
        else:
            raise ValueError(
                f"Tensor data has {len(var_data.shape)} dims, must either "
                f"have either {max_dims} or {max_dims-1}."
            )
    return reshaped_tensors
