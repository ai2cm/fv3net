import numpy as np


def _slice(arr: np.ndarray, inds: slice, axis: int = 0):
    # https://stackoverflow.com/a/37729566
    # For slicing ndarray along a dynamically specified axis
    # same as np.take() but does not make a copy of the data
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[tuple(sl)]


class Subdomain:
    def __init__(self, data: np.ndarray, overlap: int):
        self.overlapping = data
        self.overlap = overlap
        self.nonoverlapping = data[overlap:-overlap]


class Domain:
    def __init__(self, data, output_size, overlap, subdomain_axis: int = 0):
        self.data = data
        self.output_size = output_size
        if len(self.data) % self.output_size != 0:
            raise ValueError(f"Data size must be evenly divisible by output_size")
        self.overlap = overlap
        self.subdomain_axis = subdomain_axis

    def __len__(self) -> int:
        return len(self.data) / self.output_size

    def __getitem__(self, index: int):
        padded = np.hstack(
            [
                _slice(
                    arr=self.data,
                    inds=slice(-self.overlap, None),
                    axis=self.subdomain_axis,
                ),
                self.data,
                _slice(
                    arr=self.data,
                    inds=slice(None, self.overlap),
                    axis=self.subdomain_axis,
                ),
            ]
        )
        start_ind = index * self.output_size
        stop_ind = start_ind + self.output_size + 2 * self.overlap
        if stop_ind > len(padded):
            raise ValueError(
                f"Cannot select subdomain with index {index}, there are"
                f"only {len(self.data)/self.output_size} subdomains."
            )
        subdomain_slice = _slice(
            arr=padded, inds=slice(start_ind, stop_ind), axis=self.subdomain_axis
        )
        return Subdomain(subdomain_slice, overlap=self.overlap)
