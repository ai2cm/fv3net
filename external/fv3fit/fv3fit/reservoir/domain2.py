from typing import Tuple, Optional

import pace.util
import numpy as np
import fsspec
import yaml
import logging

logger = logging.getLogger(__name__)


def _check_feature_dims_consistent(data_shape, feature_shape):
    n_feature_dims = len(feature_shape)
    feature_dims = data_shape[-n_feature_dims:]
    if feature_dims != tuple(feature_shape):
        raise ValueError(
            f"Feature dimensions of data {feature_dims}"
            f" are not consistent with expected: {feature_shape}"
        )


class RankXYDivider:
    """
    Rank divider class for handling subdomains on a tile.  Useful for breaking
    into subdomains for smaller training/prediction tasks.  Assumes that the
    spatial feature dimensions are trailing.  I.e., 2D rank dimensions (e.g., [x, y])
    are last or potentially followed by a latent/vertical dimension.

    Args:
        subdomain_layout: layout describing subdomain grid within the rank
            ex. [2,2] means the rank is divided into 4 subdomains
        overlap: number of cells of overlap between subdomains
        rank_extent: Shape of the tile (e.g., [48, 48] for C48 grid) with no overlap.
            Errors if overlap_rank_extent also specified.
        overlap_rank_extent: Shape of the tile with an appended halo (e.g., [52, 52]
             for a C48 grid with a halo of 2).  Errors if rank_extent also specified.
        z_feature_size: Optional trailing feature dimension length.  Always assumed
            to be follow the rank_extent dimensions.  Usually this corresponds to a
            latent dimension from an encoding (e.g., the number of PCA components) or
            a vertical dimension (e.g., the number of vertical levels).
    """

    def __init__(
        self,
        subdomain_layout: Tuple[int, int],
        overlap: int,
        rank_extent: Optional[Tuple[int, int]] = None,
        overlap_rank_extent: Optional[Tuple[int, int]] = None,
        z_feature_size: Optional[int] = None,
    ):
        if len(subdomain_layout) != 2:
            raise ValueError("Rank divider only handles 2D subdomain layouts")

        if overlap < 0:
            raise ValueError("Overlap must be non-negative")

        self._rank_dims = ["x", "y"]

        self.overlap = overlap
        self.subdomain_layout = subdomain_layout
        self.n_subdomains = subdomain_layout[0] * subdomain_layout[1]
        self._partitioner = pace.util.TilePartitioner(
            self._subdomain_layout_for_partitioner
        )

        self._init_rank_extent(rank_extent, overlap_rank_extent)
        self._x_rank_extent = self.rank_extent[0]
        self._y_rank_extent = self.rank_extent[1]
        self._z_feature_size = z_feature_size

        self._check_extent_divisibility()
        self._x_subdomain_extent = (
            self._x_rank_extent // self.subdomain_layout[0] + 2 * self.overlap
        )
        self._y_subdomain_extent = (
            self._y_rank_extent // self.subdomain_layout[1] + 2 * self.overlap
        )

    @property
    def subdomain_extent(self):
        return self._x_subdomain_extent, self._y_subdomain_extent

    @property
    def _rank_extent_for_partitioner(self):
        # Fed into partitioner for slicing, no overlap should ever be given
        return self._maybe_append_feature_value(self.rank_extent, self._z_feature_size)

    @property
    def _subdomain_layout_for_partitioner(self):
        # partitioner expects layout in y, x order:
        x_ind = self._rank_dims.index("x")
        y_ind = self._rank_dims.index("y")
        return self.subdomain_layout[y_ind], self.subdomain_layout[x_ind]

    @property
    def _rank_extent_all_features(self):
        # used for data consistency checks
        return self._maybe_append_feature_value(
            self.overlap_rank_extent, self._z_feature_size
        )

    @property
    def _rank_dims_all_features(self):
        return self._maybe_append_feature_value(self._rank_dims, "z")

    @property
    def _subdomain_shape(self):
        return self._maybe_append_feature_value(
            self.subdomain_extent, self._z_feature_size
        )

    @property
    def flat_subdomain_len(self) -> int:
        """length of flattened trailing feature dimensions of subdomain"""
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
        """axis dimension for decomposed subdomains with flattened features"""
        return -2

    def __eq__(self, other):
        if isinstance(other, RankXYDivider):
            return (
                self.subdomain_layout == other.subdomain_layout
                and self.rank_extent == other.rank_extent
                and self._z_feature_size == other._z_feature_size
            )
        else:
            return False

    def get_new_zdim_rank_divider(self, z_feature_size: int):
        return RankXYDivider(
            subdomain_layout=self.subdomain_layout,
            overlap=self.overlap,
            rank_extent=self.rank_extent,
            z_feature_size=z_feature_size,
        )

    def get_no_overlap_rank_divider(self):
        if self.overlap == 0:
            return self
        else:
            return RankXYDivider(
                self.subdomain_layout, 0, self.rank_extent, None, self._z_feature_size
            )

    def _init_rank_extent(self, rank_extent, overlap_rank_extent):
        if rank_extent is None and overlap_rank_extent is None:
            raise ValueError("Must specify either rank_extent or overlap_rank_extent")
        elif rank_extent is not None and overlap_rank_extent is not None:
            raise ValueError("Cannot specify both rank_extent and overlap_rank_extent")

        if rank_extent is not None:
            if len(rank_extent) != 2:
                raise ValueError(
                    "Rank divider only handles 2D rank extents. A feature dimension "
                    "should be included using 'z_feature' and leading dimensions "
                    "should be excluded."
                )
            self.rank_extent = rank_extent
            self.overlap_rank_extent = (
                rank_extent[0] + 2 * self.overlap,
                rank_extent[1] + 2 * self.overlap,
            )
        elif overlap_rank_extent is not None:
            if len(overlap_rank_extent) != 2:
                raise ValueError(
                    "Rank divider only handles 2D rank extents. A feature dimension "
                    "should be included using 'z_feature' and leading dimensions "
                    "should be excluded."
                )
            self.overlap_rank_extent = overlap_rank_extent
            self.rank_extent = (
                overlap_rank_extent[0] - 2 * self.overlap,
                overlap_rank_extent[1] - 2 * self.overlap,
            )

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
        to_append = [] if self._z_feature_size is None else [feature_value]
        return [*rank_values, *to_append]

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
        rank_dims = self._rank_dims_all_features
        rank_extent = self._rank_extent_for_partitioner
        slices = self._partitioner.subtile_slice(
            subdomain_index, rank_dims, rank_extent
        )
        return self._update_slices_with_overlap(slices)

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

        _check_feature_dims_consistent(data.shape, self._rank_extent_all_features)
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

        if self.overlap > 0:
            raise ValueError("Cannot merge subdomains with overlap")

        _check_feature_dims_consistent(data.shape, self._all_subdomains_shape)
        rank_extent = self._rank_extent_all_features
        new_shape = list(data.shape[: self.subdomain_axis]) + rank_extent
        merged = np.empty(new_shape, dtype=data.dtype)

        for i in range(self.n_subdomains):
            subdomain = np.take(data, i, axis=self.subdomain_axis)
            dim_slices = self._get_subdomain_slice(i)
            dim_slices = self._add_potential_leading_dim_to_slices(
                subdomain.shape, dim_slices
            )
            merged[tuple(dim_slices)] = subdomain

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
            axis = self.flat_feature_subdomain_axis
        else:
            axis = self.subdomain_axis

        if data.shape[axis] != self.n_subdomains:
            raise ValueError(
                f"Data must have subdomain axis {axis} with length "
                f"{self.n_subdomains}: got {data.shape[axis]}"
            )
        return np.moveaxis(data, axis, 0)

    def trim_halo_from_rank_data(self, data: np.ndarray) -> np.ndarray:
        """
        Remove halo points (the overlap) from the rank data.
        """

        _check_feature_dims_consistent(data.shape, self._rank_extent_all_features)

        if self.overlap == 0:
            logger.debug("No overlap to trim, returning original data.")
            return data

        no_overlap_slice = slice(self.overlap, -self.overlap)
        slices = [no_overlap_slice, no_overlap_slice]
        slices = self._maybe_append_feature_value(slices, slice(None))
        slices = self._add_potential_leading_dim_to_slices(data.shape, slices)

        return data[tuple(slices)]

    def dump(self, path):
        metadata = {
            "subdomain_layout": self.subdomain_layout,
            "overlap": self.overlap,
            "rank_extent": self.rank_extent,
            "z_feature_size": self._z_feature_size,
        }
        with fsspec.open(path, "w") as f:
            f.write(yaml.dump(metadata))

    @classmethod
    def load(cls, path):
        with fsspec.open(path, "r") as f:
            metadata = yaml.safe_load(f)
        return cls(**metadata)
