from typing import Any, Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple
import tensorflow as tf
from fv3fit._shared.stacking import match_prediction_to_input_coords
import xarray as xr
import numpy as np
import pace.util
from toolz.functoolz import curry


def _create_comms(total_ranks: int) -> Sequence[pace.util.testing.DummyComm]:
    buffer_dict: Dict[Any, Any] = {}
    comms: List[pace.util.testing.DummyComm] = []
    for rank in range(total_ranks):
        comms.append(
            pace.util.testing.DummyComm(
                rank=rank, total_ranks=total_ranks, buffer_dict=buffer_dict
            )
        )
    return comms


class _NoBufferPointGridSizer(pace.util.SubtileGridSizer):
    """
    The default SubtileGridSizer adds buffer points to cell-centered variables
    so that they have the same array shape as interface variables. Here we do not
    want to do that, so we override the shape definition.
    """

    def get_shape(self, dims: Iterable[str]) -> Tuple[int, ...]:
        shape_dict = self.extra_dim_lengths.copy()
        shape_dict.update(
            {
                pace.util.constants.X_DIM: self.nx + 2 * self.n_halo,
                pace.util.constants.X_INTERFACE_DIM: self.nx + 1 + 2 * self.n_halo,
                pace.util.constants.Y_DIM: self.ny + 2 * self.n_halo,
                pace.util.constants.Y_INTERFACE_DIM: self.ny + 1 + 2 * self.n_halo,
                pace.util.constants.Z_DIM: self.nz,
                pace.util.constants.Z_INTERFACE_DIM: self.nz + 1,
            }
        )
        return tuple(shape_dict[dim] for dim in dims)


def _quantities_from_dataset(
    ds: xr.Dataset, n_halo: int
) -> Mapping[Hashable, pace.util.Quantity]:
    return_dict = {}
    nx, ny, nz = ds.dims["x"], ds.dims["y"], ds.dims["z"]
    sizer = _NoBufferPointGridSizer(
        nx=nx, ny=ny, nz=nz, n_halo=n_halo, extra_dim_lengths={**ds.sizes}
    )
    factory = pace.util.QuantityFactory(sizer=sizer, numpy=np)
    for name, da in ds.data_vars.items():
        quantity = factory.zeros(
            dims=da.dims, units=da.attrs.get("units", "unknown"), dtype=da.dtype
        )
        quantity.view[:] = da.values[:]
        return_dict[name] = quantity
    return return_dict


def _dataset_from_quantities(
    quantities: Mapping[Hashable, pace.util.Quantity]
) -> xr.Dataset:
    data_vars = {}
    for name, quantity in quantities.items():
        data_vars[name] = xr.DataArray(
            data=quantity.data, dims=quantity.dims, attrs={"units": quantity.units}
        )
    return xr.Dataset(data_vars=data_vars)


def _halo_update(
    communicators: Sequence[pace.util.CubedSphereCommunicator],
    quantities: Sequence[pace.util.Quantity],
    n_halo: int,
):
    req_list = []
    for comm, quantity in zip(communicators, quantities):
        req_list.append(comm.start_halo_update(quantity, n_points=n_halo))
    for req in req_list:
        req.wait()


def _get_comm():
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    except ModuleNotFoundError:
        raise RuntimeError("mpi4py is not installed")
    return comm


def _get_cubed_sphere_communicator(comm):
    # it would be ideal to take this in externally, so configuration can occur
    # at the top level instead of hard-coding a square layout on a cube,
    # but we create the communicator internally here to avoid refactoring
    # configuration of the prognostic run
    #
    # This could possibly be changed if/when we integrate with the DSL model
    layout_1d = int((comm.total_ranks / 6.0) ** 0.5)
    return pace.util.CubedSphereCommunicator(
        comm=comm,
        partitioner=pace.util.CubedSpherePartitioner(
            tile=pace.util.TilePartitioner(layout=(layout_1d, layout_1d))
        ),
    )


def append_halos_using_mpi(ds: xr.Dataset, n_halo: int) -> xr.Dataset:
    comm = _get_comm()
    if comm.Get_size() < 6:
        raise RuntimeError("to halo update over MPI we need at least 6 ranks")
    _append_halos_using_mpi(ds=ds, n_halo=n_halo, comm=comm)


def _append_halos_using_mpi(ds: xr.Dataset, n_halo: int, comm):
    communicator = _get_cubed_sphere_communicator(comm=comm)
    quantity_dict = _quantities_from_dataset(ds, n_halo=n_halo)
    req_list = []
    for name in sorted(ds.data_vars.keys()):
        quantity = quantity_dict[name]
        req_list.append(communicator.start_halo_update(quantity, n_points=n_halo))
    for req in req_list:
        req.wait()
    ds_out = _dataset_from_quantities(quantity_dict)
    ds_out = match_prediction_to_input_coords(input=ds, prediction=ds_out)
    return ds_out.drop(ds_out.coords.keys())


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
    communicators = [_get_cubed_sphere_communicator(comm) for comm in comms]
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


@curry
def append_halos_tensor(n_halo: int, tensor: tf.Tensor) -> tf.Tensor:
    """
    Given a tensor with
    "tile", "x", "y", and "z" dimensions and no halo data,
    return a tensor which has n_halo halo points of valid data
    appended to the start and end of the x and y dimensions.
    """
    if n_halo > 0:
        out_tensor = tf.py_function(
            _append_halos_tensor(n_halo), inp=(tensor,), Tout=tensor.dtype
        )
    else:
        out_tensor = tensor
    return out_tensor


@curry
def _append_halos_tensor(n_halo: int, tensor: tf.Tensor) -> tf.Tensor:
    comms = _create_comms(total_ranks=6)
    communicators = [_get_cubed_sphere_communicator(comm) for comm in comms]
    shape = list(tensor.shape[1:])
    shape[0] += 2 * n_halo
    shape[1] += 2 * n_halo
    quantities = [
        pace.util.Quantity(
            data=np.zeros(shape, dtype=_dtype_to_numpy[tensor.dtype]),
            dims=["x", "y", "z"],
            units="unknown",
            origin=(n_halo, n_halo, 0),
            extent=tuple(tensor.shape[1:]),
        )
        for _ in range(6)
    ]
    for i, quantity in enumerate(quantities):
        quantity.view[:] = tensor[i, :, :, :].numpy()
    _halo_update(communicators=communicators, quantities=quantities, n_halo=n_halo)
    out_tensor = tf.concat(
        [quantity.data[None, :, :, :] for quantity in quantities], axis=0
    )
    return out_tensor


_dtype_to_numpy = {
    tf.float32: np.float32,
    tf.float64: np.float64,
}
