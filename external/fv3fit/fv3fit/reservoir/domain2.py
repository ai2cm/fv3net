from typing import Tuple, Optional

import pace.util
import numpy as np


def _check_feature_dims_consistent(data_shape, feature_shape):
    n_feature_dims = len(feature_shape)
    feature_dims = data_shape[-n_feature_dims:]
    if feature_dims != tuple(feature_shape):
        raise ValueError(
            f"Feature dimensions of data {feature_dims}"
            " are not consistent with expected: {feature_shape}"
        )


class RankXYDivider:
    def __init__(
        self,
        subdomain_layout: Tuple[int, int],
        # shape of full x,y data, including overlap,
        # easier to initialize from halo included data
        rank_extent: Tuple[int, int],
        z_feature: Optional[int] = None,
    ):
        if len(subdomain_layout) != 2:
            raise ValueError("Rank divider only handles 2D subdomain layouts")

        self.rank_extent = rank_extent
        self._extent_for_check = rank_extent
        self.subdomain_layout = subdomain_layout
        self.n_subdomains = subdomain_layout[0] * subdomain_layout[1]
        self._partitioner = pace.util.TilePartitioner(subdomain_layout)

        self._x_rank_extent = self.rank_extent[0]
        self._y_rank_extent = self.rank_extent[1]
        self._z_feature = z_feature

        # TODO: maybe assert that subdomain decomp works
        # for extent and subdomain layout?
        self._x_subdomain_extent = self._x_rank_extent // self.subdomain_layout[0]
        self._y_subdomain_extent = self._y_rank_extent // self.subdomain_layout[1]

    @property
    def subdomain_xy_extent(self):
        return self._x_subdomain_extent, self._y_subdomain_extent

    @property
    def _rank_extent_all_features(self):
        # Used for partitioner slicing, no overlap ever
        return self._maybe_add_z_value(self.rank_extent, self._z_feature)

    @property
    def _rank_extent_for_check(self):
        # used for data consistency check, maybe overlap
        return self._rank_extent_all_features

    @property
    def _rank_dims_all_features(self):
        # TODO: y might go before "x" since that aligns
        # with row-major order (also, lat, lon, feature)
        # probably doesn't matter given x,y agnostic merge
        # subdomains function
        return self._maybe_add_z_value(["x", "y"], "z")

    def _maybe_add_z_value(self, rank_values, feature_value):
        values = list(rank_values)
        if self._z_feature is not None:
            values += [feature_value]
        return values

    def _get_subdomain_slice(self, subdomain_index):

        rank_dims = self._rank_dims_all_features
        rank_extent = self._rank_extent_all_features
        return self._partitioner.subtile_slice(subdomain_index, rank_dims, rank_extent)

    def _add_potential_leading_dim_to_slices(self, data_shape, dim_slices):

        # add leading dimensions to slice if necessary
        if len(data_shape) > len(self._rank_dims_all_features):
            dim_slices = [..., *dim_slices]

        return dim_slices

    def get_subdomain(self, data, subdomain_index):

        if subdomain_index < 0 or subdomain_index >= self.n_subdomains:
            raise ValueError(
                f"Subdomain index {subdomain_index} out of range "
                "[0, {self.n_subdomains})"
            )

        _check_feature_dims_consistent(data.shape, self._rank_extent_for_check)
        dim_slices = self._get_subdomain_slice(subdomain_index)
        dim_slices = self._add_potential_leading_dim_to_slices(data.shape, dim_slices)

        return data[dim_slices]

    def get_all_subdomains(self, data):

        new_index_dim = -1 * len(self.all_subdomains_shape)
        subdomains_with_new_dim = []
        for i in range(self.n_subdomains):
            subdomain = self.get_subdomain(data, i)
            subdomain_newaxis = np.expand_dims(subdomain, axis=new_index_dim)
            subdomains_with_new_dim.append(subdomain_newaxis)
        return np.concatenate(subdomains_with_new_dim, axis=new_index_dim)

    @property
    def subdomain_shape(self):
        shape = list(self.subdomain_xy_extent)
        if self._z_feature is not None:
            shape += [self._z_feature]
        return shape

    @property
    def flat_subdomain_shape(self):
        return [np.prod(self.subdomain_shape)]

    @property
    def all_subdomains_shape(self):
        return [self.n_subdomains, *self.subdomain_shape]

    def flatten_subdomain_features(self, data):
        feature_shape = self.subdomain_shape
        _check_feature_dims_consistent(data.shape, feature_shape)
        n_feature_dims = len(feature_shape)
        return data.reshape(list(data.shape[:-n_feature_dims]) + [-1])

    def reshape_flat_subdomain_features(self, data):
        flat_feature_shape = self.flat_subdomain_shape
        _check_feature_dims_consistent(data.shape, flat_feature_shape)
        original_shape = list(data.shape[:-1]) + self.subdomain_shape
        return data.reshape(original_shape)

    def merge_all_subdomains(self, data):
        # [leading, nsubdomains, ..., x, y, (z)]

        _check_feature_dims_consistent(data.shape, self.all_subdomains_shape)
        rank_extent = self._rank_extent_all_features
        subdomain_axis = -1 * len(self.all_subdomains_shape)
        new_shape = list(data.shape[:subdomain_axis]) + rank_extent
        merged = np.empty(new_shape, dtype=data.dtype)

        for i in range(self.n_subdomains):
            subdomain = np.take(data, i, axis=subdomain_axis)
            dim_slices = self._get_subdomain_slice(i)
            dim_slices = self._add_potential_leading_dim_to_slices(
                subdomain.shape, dim_slices
            )
            merged[dim_slices] = subdomain

        return merged


class OverlapRankXYDivider(RankXYDivider):

    """
    Adjusted rank divider to handle halo overlap regions
    """

    def __init__(
        self,
        subdomain_layout: Tuple[int, int],
        overlap_rank_extent: Tuple[int, int],
        overlap: int,
        z_feature: Optional[int] = None,
    ):
        if len(subdomain_layout) != 2:
            raise ValueError("Rank divider only handles 2D subdomain layouts")

        self.overlap_rank_extent = overlap_rank_extent
        self.subdomain_layout = subdomain_layout
        self.n_subdomains = subdomain_layout[0] * subdomain_layout[1]
        self._partitioner = pace.util.TilePartitioner(subdomain_layout)
        self.overlap = overlap

        self._x_rank_extent = self.overlap_rank_extent[0]
        self._y_rank_extent = self.overlap_rank_extent[1]
        self._z_feature = z_feature

        self.rank_extent = (
            self._x_rank_extent - 2 * self.overlap,
            self._y_rank_extent - 2 * self.overlap,
        )

        # TODO: maybe assert that subdomain decomp works
        # for extent and subdomain layout?
        self._x_subdomain_extent = (
            self.rank_extent[0] // self.subdomain_layout[0] + 2 * self.overlap
        )
        self._y_subdomain_extent = (
            self.rank_extent[1] // self.subdomain_layout[1] + 2 * self.overlap
        )

    def get_no_overlap_rank_xy_divider(self):
        return RankXYDivider(self.subdomain_layout, self.rank_extent, self._z_feature,)

    @property
    def _rank_extent_for_check(self):
        return self._maybe_add_z_value(self.overlap_rank_extent, self._z_feature)

    def _update_slices_with_overlap(self, slices):
        rank_dims = self._rank_dims_all_features

        x_ind = rank_dims.index("x")
        y_ind = rank_dims.index("y")

        slices = list(slices)
        x_sl, y_sl = slices[x_ind], slices[y_ind]
        slices[x_ind] = slice(x_sl.start, x_sl.stop + 2 * self.overlap, None)
        slices[y_ind] = slice(y_sl.start, y_sl.stop + 2 * self.overlap, None)

        return slices

    def _get_subdomain_slice(self, subdomain_index):
        slices = super()._get_subdomain_slice(subdomain_index)
        return self._update_slices_with_overlap(slices)

    def merge_all_subdomains(self, data):
        raise NotImplementedError("Merging overlapped subdomains is not supported.")
