import xarray as xr

from typing import Sequence, Union


class SerializedSequence(Sequence[xr.Dataset]):
    """
    Create a sequence over savepoints for serialized physics data.
    """

    def __init__(
        self, xr_data: Union[xr.Dataset, xr.DataArray], item_dim: str = "savepoint"
    ):
        """
        Args
        ----
        xr_data: Xarray data source
        item_dim: data dimension to use as the "get" index
        """

        self.data = xr_data
        self.item_dim = item_dim

    def __getitem__(self, item: Union[int, slice]):

        return self.data.isel({self.item_dim: item}).load()

    def __len__(self):
        return self.data.sizes[self.item_dim]
