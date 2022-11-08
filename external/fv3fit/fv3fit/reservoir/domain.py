import numpy as np


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
        self.nonoverlapping = _slice(
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
        self.n_subdomains = int(data.shape[subdomain_axis] / subdomain_size)
        self.index = 0

    def __len__(self) -> int:
        return self.n_subdomains

    def __getitem__(self, index: int):
        padded = np.hstack(
            [
                _slice(
                    arr=self.data,
                    inds=slice(-self.subdomain_overlap, None),
                    axis=self.subdomain_axis,
                ),
                self.data,
                _slice(
                    arr=self.data,
                    inds=slice(None, self.subdomain_overlap),
                    axis=self.subdomain_axis,
                ),
            ]
        )
        start_ind = index * self.subdomain_size
        stop_ind = start_ind + self.subdomain_size + 2 * self.subdomain_overlap
        if stop_ind > len(padded):
            raise ValueError(
                f"Cannot select subdomain with index {index}, there are"
                f"only {len(self.data)/self.subdomain_size} subdomains."
            )
        subdomain_slice = _slice(
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
