from typing import Any, Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple, Union
from fv3fit._shared.stacking import match_prediction_to_input_coords
import xarray as xr
import numpy as np
import fv3gfs.util


class HalosBatches(Sequence[xr.Dataset]):
    def __init__(
        self, batches: Sequence[xr.Dataset], n_halo: int,
    ):
        self._batches = batches
        self._n_halo = n_halo

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, int):
            return append_halos(self._batches[idx], n_halo=self._n_halo)
        elif isinstance(idx, slice):
            return [append_halos(ds, n_halo=self._n_halo) for ds in self._batches[idx]]
        else:
            raise TypeError(
                f"Invalid argument type of {type(idx)} passed into "
                "StackedBatches.__getitem__."
            )

    def __len__(self) -> int:
        return len(self._batches)


def _create_comms(total_ranks: int) -> Sequence[fv3gfs.util.testing.DummyComm]:
    buffer_dict: Dict[Any, Any] = {}
    comms: List[fv3gfs.util.testing.DummyComm] = []
    for rank in range(total_ranks):
        comms.append(
            fv3gfs.util.testing.DummyComm(
                rank=rank, total_ranks=total_ranks, buffer_dict=buffer_dict
            )
        )
    return comms


class _NoBufferPointGridSizer(fv3gfs.util.SubtileGridSizer):
    """
    The default SubtileGridSizer adds buffer points to cell-centered variables
    so that they have the same array shape as interface variables. Here we do not
    want to do that, so we override the shape definition.
    """

    def get_shape(self, dims: Iterable[str]) -> Tuple[int, ...]:
        shape_dict = self.extra_dim_lengths.copy()
        shape_dict.update(
            {
                fv3gfs.util.constants.X_DIM: self.nx + 2 * self.n_halo,
                fv3gfs.util.constants.X_INTERFACE_DIM: self.nx + 1 + 2 * self.n_halo,
                fv3gfs.util.constants.Y_DIM: self.ny + 2 * self.n_halo,
                fv3gfs.util.constants.Y_INTERFACE_DIM: self.ny + 1 + 2 * self.n_halo,
                fv3gfs.util.constants.Z_DIM: self.nz,
                fv3gfs.util.constants.Z_INTERFACE_DIM: self.nz + 1,
            }
        )
        return tuple(shape_dict[dim] for dim in dims)


def _quantities_from_dataset(
    ds: xr.Dataset, n_halo: int
) -> Mapping[Hashable, fv3gfs.util.Quantity]:
    return_dict = {}
    nx, ny, nz = ds.dims["x"], ds.dims["y"], ds.dims["z"]
    sizer = _NoBufferPointGridSizer(
        nx=nx, ny=ny, nz=nz, n_halo=n_halo, extra_dim_lengths={**ds.sizes}
    )
    factory = fv3gfs.util.QuantityFactory(sizer=sizer, numpy=np)
    for name, da in ds.data_vars.items():
        quantity = factory.zeros(
            dims=da.dims, units=da.attrs.get("units", "unknown"), dtype=da.dtype
        )
        quantity.view[:] = da.values[:]
        return_dict[name] = quantity
    return return_dict


def _dataset_from_quantities(
    quantities: Mapping[Hashable, fv3gfs.util.Quantity]
) -> xr.Dataset:
    data_vars = {}
    for name, quantity in quantities.items():
        data_vars[name] = xr.DataArray(
            data=quantity.data, dims=quantity.dims, attrs={"units": quantity.units}
        )
    return xr.Dataset(data_vars=data_vars)


def _halo_update(
    communicators: Sequence[fv3gfs.util.CubedSphereCommunicator],
    quantities: Sequence[fv3gfs.util.Quantity],
    n_halo: int,
):
    req_list = []
    for comm, quantity in zip(communicators, quantities):
        req_list.append(comm.start_halo_update(quantity, n_points=n_halo))
    for req in req_list:
        req.wait()


def append_halos(ds: xr.Dataset, n_halo: int) -> xr.Dataset:
    """
    Given a dataset with "tile", "x", and "y" dimensions with no halo data,
    return a dataset which has n_halo halo points appended to the start and
    end of the x and y dimensions.
    """
    if any(name not in ds.dims for name in ("tile", "x", "y")):
        raise ValueError(
            f"dataset must have 'tile', 'x', and 'y' dimensions, got {ds.dims}"
        )
    n_tiles = ds.sizes["tile"]
    if n_tiles != 6:
        raise ValueError(f"dataset must have 6 tiles to halo update, got {n_tiles}")
    comms = _create_comms(total_ranks=6)
    communicators = [
        fv3gfs.util.CubedSphereCommunicator(
            comm=comm,
            partitioner=fv3gfs.util.CubedSpherePartitioner(
                tile=fv3gfs.util.TilePartitioner(layout=(1, 1))
            ),
        )
        for comm in comms
    ]
    datasets_in = [ds.isel(tile=i) for i in range(0, 6)]
    quantity_dicts = [_quantities_from_dataset(ds, n_halo=n_halo) for ds in datasets_in]
    for name in ds.data_vars.keys():
        quantities = [d[name] for d in quantity_dicts]
        if n_halo > 0:
            _halo_update(communicators, quantities, n_halo=n_halo)
    datasets_out = [_dataset_from_quantities(d) for d in quantity_dicts]
    ds_out = xr.concat(datasets_out, dim="tile")
    ds_out = match_prediction_to_input_coords(input=ds, prediction=ds_out)
    # if you don't drop coords here, the model fails to train - no idea why
    return ds_out.drop(ds_out.coords.keys())
