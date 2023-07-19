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
    """
    Base rank divider class for handling subdomains on a tile
    (with no overlaps).  Useful for breaking into subdomains
    for smaller training/prediction tasks.

    Args:
        subdomain_layout: layout describing subdomain grid within the rank
            ex. [2,2] means the rank is divided into 4 subdomains
        rank_extent: Shape of the tile (e.g., [48, 48] for C48 grid)
        z_feature: Optional trailing feature dimension.  Always assumed
            to be follow the rank_extent dimensions.
    """

    def __init__(
        self,
        subdomain_layout: Tuple[int, int],
        rank_extent: Tuple[int, int],
        z_feature: Optional[int] = None,
    ):
        if len(subdomain_layout) != 2:
            raise ValueError("Rank divider only handles 2D subdomain layouts")

        if len(rank_extent) != 2:
            raise ValueError(
                "Rank divider only handles 2D rank extents. A feature dimension "
                "should be included using 'z_feature' and leading dimensions "
                "should be excluded."
            )

        self.rank_extent = rank_extent
        self._extent_for_check = rank_extent
        self.subdomain_layout = subdomain_layout
        self.n_subdomains = subdomain_layout[0] * subdomain_layout[1]
        self._partitioner = pace.util.TilePartitioner(subdomain_layout)

        self._x_rank_extent = self.rank_extent[0]
        self._y_rank_extent = self.rank_extent[1]
        self._z_feature = z_feature

        self._check_extent_divisibility()
        self._x_subdomain_extent = self._x_rank_extent // self.subdomain_layout[0]
        self._y_subdomain_extent = self._y_rank_extent // self.subdomain_layout[1]

    @property
    def subdomain_extent(self):
        return self._x_subdomain_extent, self._y_subdomain_extent

    @property
    def _rank_extent_all_features(self):
        # Fed into partitioner for slicing, no overlap should ever be used
        return self._maybe_append_feature_value(self.rank_extent, self._z_feature)

    @property
    def _rank_extent_for_check(self):
        # used for data consistency check, maybe has overlap depending on class
        return self._rank_extent_all_features

    @property
    def _rank_dims_all_features(self):
        return self._maybe_append_feature_value(["x", "y"], "z")

    @property
    def _subdomain_shape(self):
        return self._maybe_append_feature_value(self.subdomain_extent, self._z_feature)

    @property
    def flat_subdomain_len(self) -> int:
        return np.prod(self._subdomain_shape)

    @property
    def _all_subdomains_shape(self):
        return [self.n_subdomains, *self._subdomain_shape]

    @property
    def subdomain_axis(self):
        """axis dimension for decomposed subdomains"""
        return -1 * len(self._all_subdomains_shape)

    @property
    def flat_feature_subdomain_axis(self):
        return -2

    def _check_extent_divisibility(self):
        if self._x_rank_extent % self.subdomain_layout[0] != 0:
            raise ValueError(
                f"X rank extent {self._x_rank_extent} is not divisible by "
                f"subdomain layout {self.subdomain_layout[0]}"
            )
        if self._y_rank_extent % self.subdomain_layout[1] != 0:
            raise ValueError(
                f"Y rank extent {self._y_rank_extent} is not divisible by "
                f"subdomain layout {self.subdomain_layout[1]}"
            )

    def _maybe_append_feature_value(self, rank_values, feature_value):
        to_append = [] if self._z_feature is None else [feature_value]
        return [*rank_values, *to_append]

    def _get_subdomain_slice(self, subdomain_index):
        rank_dims = self._rank_dims_all_features
        rank_extent = self._rank_extent_all_features
        return self._partitioner.subtile_slice(subdomain_index, rank_dims, rank_extent)

    def _add_potential_leading_dim_to_slices(self, data_shape, dim_slices):
        if len(data_shape) > len(self._rank_dims_all_features):
            dim_slices = [..., *dim_slices]

        return dim_slices

    def get_subdomain(self, data: np.ndarray, subdomain_index: int) -> np.ndarray:
        """
        Get a subdomain from specified data.  Data must have
        trailing feature dimensions that match the rank_extent and specified
        z_feature (if any).

        Args:
            data: Data to get subdomain from with shape [..., x, y, (z)]
            subdomain_index: Index of subdomain to retrieve

        Returns:
            Subdomain of data with shape [..., x_sub, y_sub, (z)]
        """

        if subdomain_index < 0 or subdomain_index >= self.n_subdomains:
            raise ValueError(
                f"Subdomain index {subdomain_index} out of range "
                "[0, {self.n_subdomains})"
            )

        _check_feature_dims_consistent(data.shape, self._rank_extent_for_check)
        dim_slices = self._get_subdomain_slice(subdomain_index)
        dim_slices = self._add_potential_leading_dim_to_slices(data.shape, dim_slices)

        return data[tuple(dim_slices)]

    def flatten_subdomain_features(self, data: np.ndarray) -> np.ndarray:
        """
        Flatten trailing feature dimensions of subdomain into a single dimension.
        """
        feature_shape = self._subdomain_shape
        _check_feature_dims_consistent(data.shape, feature_shape)
        n_feature_dims = len(feature_shape)
        return data.reshape(list(data.shape[:-n_feature_dims]) + [-1])

    def reshape_flat_subdomain_features(self, data: np.ndarray) -> np.ndarray:
        """
        Reshape flattened trailing feature dimensions of subdomain into original
        subdomain dimensions.
        """
        flat_feature_shape = [self.flat_subdomain_len]
        _check_feature_dims_consistent(data.shape, flat_feature_shape)
        original_shape = list(data.shape[:-1]) + self._subdomain_shape
        return data.reshape(original_shape)

    def get_all_subdomains(self, data: np.ndarray) -> np.ndarray:
        """
        Retrieve all subdomains from data.  Data must have trailing feature
        dimensions that match the rank_extent and specified z_feature (if any).
        Subdomain dimension is added just before the trailing feature dimensions,
        so any leading dimensions are preserved

        Args:
            data: Data to get subdomains from with shape [..., x, y, (z)]

        Returns:
            Subdomains of data with shape [..., n_subdomains, x_sub, y_sub, (z)]
        """

        subdomains_with_new_dim = []

        for i in range(self.n_subdomains):
            subdomain = self.get_subdomain(data, i)
            subdomain_newaxis = np.expand_dims(subdomain, axis=self.subdomain_axis)
            subdomains_with_new_dim.append(subdomain_newaxis)

        return np.concatenate(subdomains_with_new_dim, axis=self.subdomain_axis)

    def merge_all_subdomains(self, data: np.ndarray) -> np.ndarray:
        """
        Merge separated subdomains into original rank extent shape. For
        example, data [..., subdomain, x_sub, y_sub, (z)] will be merged into
        [..., x, y, (z)].
        """

        _check_feature_dims_consistent(data.shape, self._all_subdomains_shape)
        rank_extent = self._rank_extent_all_features
        subdomain_axis = -1 * len(self._all_subdomains_shape)
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

    def get_all_subdomains_with_flat_feature(self, data: np.ndarray) -> np.ndarray:
        """
        A convenience function that separates into subdomains and flattens
        the feature dimension.
        """
        subdomains = self.get_all_subdomains(data)
        return self.flatten_subdomain_features(subdomains)

    def merge_all_flat_feature_subdomains(self, data: np.ndarray) -> np.ndarray:
        """
        A convenience function that reshapes flat subdomain features to original
        features and then merges subdomains to the original rank extent shape.
        """
        orig_features = self.reshape_flat_subdomain_features(data)
        return self.merge_all_subdomains(orig_features)

    def subdomains_to_leading_axis(
        self, data: np.ndarray, flat_feature: bool = False
    ) -> np.ndarray:
        """
        Creates a sequence where the leading dimension is by subdomain.
        """
        if flat_feature:
            feature_shape = [self.n_subdomains, self.flat_subdomain_len]
            axis = self.flat_feature_subdomain_axis
        else:
            feature_shape = [self.n_subdomains, *self._subdomain_shape]
            axis = self.subdomain_axis

        _check_feature_dims_consistent(data.shape, feature_shape)
        return np.moveaxis(data, axis, 0)


class OverlapRankXYDivider(RankXYDivider):

    """
    Rank divider class for handling subdomains on a tile with
    a halo and specified subdomain overlap. Useful for breaking
    reservoir model inputs into subdomains for smaller training/prediction
    tasks.

    Args:
        subdomain_layout: layout describing subdomain grid within the rank
            ex. [2,2] means the rank is divided into 4 subdomains
        rank_extent: Shape of the tile including the halo (e.g., [52, 52] for a
            C48 grid with overlap 2.)
        overlap: Number of overlap points to include in each subdomain.  Adds
            2 * overlap to the extent of the subdomain expects the same for the
            overall rank extent.
        z_feature: Optional trailing feature dimension.  Always assumed
            to be follow the rank_extent dimensions.
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

        self._check_extent_divisibility()
        self._x_subdomain_extent = (
            self.rank_extent[0] // self.subdomain_layout[0] + 2 * self.overlap
        )
        self._y_subdomain_extent = (
            self.rank_extent[1] // self.subdomain_layout[1] + 2 * self.overlap
        )

    def get_no_overlap_rank_xy_divider(self):
        """
        Retrieves a RankXYDivider instance with the same subdomain layout and
        no overlap.
        """
        return RankXYDivider(self.subdomain_layout, self.rank_extent, self._z_feature,)

    @property
    def _rank_extent_for_check(self):
        # Uses overlap extent to check data consistency
        return self._maybe_append_feature_value(
            self.overlap_rank_extent, self._z_feature
        )

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
