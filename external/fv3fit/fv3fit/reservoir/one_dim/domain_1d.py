import numpy as np
from ..domain import slice_along_axis


def _slice(arr: np.ndarray, inds: slice, axis: int = 0):
    # https://stackoverflow.com/a/37729566
    # For slicing ndarray along a dynamically specified axis
    # same as np.take() but does not make a copy of the data
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[tuple(sl)]


class Subdomain:
    def __init__(self, data: np.ndarray, overlap: int, subdomain_axis: int = 0):
        self.overlapping = data
        self.overlap = overlap
        self.nonoverlapping = slice_along_axis(
            arr=data, inds=slice(overlap, -overlap), axis=subdomain_axis
        )


class PeriodicDomain:
    def __init__(
        self,
        data: np.ndarray,
        subdomain_size: int,
        subdomain_overlap: int,
        subdomain_axis: int = 0,
    ):
        self.data = data
        self.subdomain_size = subdomain_size
        if data.shape[subdomain_axis] % subdomain_size != 0:
            raise ValueError(f"Data size must be evenly divisible by subdomain_size")
        self.subdomain_overlap = subdomain_overlap
        self.subdomain_axis = subdomain_axis
        self.n_subdomains = data.shape[subdomain_axis] // subdomain_size
        self.index = 0

    def __len__(self) -> int:
        return self.n_subdomains

    def _pad_array_along_subdomain_axis(self, arr):
        n_dims = len(arr.shape)
        pad_widths = tuple(
            (0, 0)
            if axis != self.subdomain_axis
            else (self.subdomain_overlap, self.subdomain_overlap)
            for axis in range(n_dims)
        )
        return np.pad(arr, mode="wrap", pad_width=pad_widths)

    def __getitem__(self, index: int):
        padded = self._pad_array_along_subdomain_axis(self.data)
        start_ind = index * self.subdomain_size
        stop_ind = start_ind + self.subdomain_size + 2 * self.subdomain_overlap
        if index >= self.n_subdomains:
            raise ValueError(
                f"Cannot select subdomain with index {index}, there are "
                f"only {self.n_subdomains} subdomains."
            )
        subdomain_slice = slice_along_axis(
            arr=padded, inds=slice(start_ind, stop_ind), axis=self.subdomain_axis
        )
        return Subdomain(
            subdomain_slice,
            overlap=self.subdomain_overlap,
            subdomain_axis=self.subdomain_axis,
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.n_subdomains:
            raise StopIteration
        elem = self[self.index]
        self.index += 1
        return elem
