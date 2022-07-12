import gt4py
import numpy as np
import xarray as xr
from config import *
from copy import deepcopy

import serialbox as ser


# Read serialized data at a specific tile and savepoint
def read_data(path, scheme, tile, ser_count, is_in, vars):
    """Read serialized data from fv3gfs-fortran output

    Args:
        path (str): path to serialized data
        scheme (str): Name of scheme. Either "lwrad" or "swrad"
        tile (int): current rank
        ser_count (int): current timestep
        is_in (bool): flag denoting whether to read input or output
        vars (dict): dictionary of variable names, shapes and types

    Returns:
        dict: dictionary of variable names and numpy arrays
    """

    mode_str = "in" if is_in else "out"

    if is_in:

        serializer = ser.Serializer(
            ser.OpenModeKind.Read, path, "Generator_rank" + str(tile)
        )
        savepoint = ser.Savepoint(f"{scheme}-{mode_str}-{ser_count:0>6d}")

    else:

        serializer = ser.Serializer(
            ser.OpenModeKind.Read, path, "Generator_rank" + str(tile)
        )
        savepoint = ser.Savepoint(f"{scheme}-{mode_str}-{ser_count:0>6d}")

    return data_dict_from_var_list(vars, serializer, savepoint)


# Read serialized data at a specific tile and savepoint
def read_intermediate_data(path, scheme, tile, ser_count, routine, vars):
    """Read serialized output from fortran radiation standalone

    Args:
        path (str): path to serialized data
        scheme (str): name of scheme. Either "lwrad" or "swrad"
        tile (int): current rank
        ser_count (int): current timestep
        routine (str): name of routine to get output from
        vars (dict): dictionary of variable names, shapes and types

    Returns:
        dict: dictionary of variable names and numpy arrays
    """

    serializer = ser.Serializer(
        ser.OpenModeKind.Read, path, "Serialized_rank" + str(tile)
    )
    savepoint = ser.Savepoint(f"{scheme}-{routine}-output-{ser_count:0>6d}")

    return data_dict_from_var_list(vars, serializer, savepoint)


# Read given variables from a specific savepoint in the given serializer
def data_dict_from_var_list(vars, serializer, savepoint):
    """Get dictionary of variable names and numpy arrays

    Args:
        vars (dict): dictionary of variable names, shapes and types
        serializer (serializer): serialbox Serializer object to read from
        savepoint (str): name of savepoint to read from

    Returns:
        dict: dictionary of variable names and numpy arrays
    """

    data_dict = {}

    for var in vars.keys():
        data_dict[var] = serializer.read(var, savepoint)

    searr_to_scalar(data_dict)

    return data_dict


# Convert single element arrays (searr) to scalar values of the correct
# type
def searr_to_scalar(data_dict):
    """convert size-1 numpy arrays to scalars

    Args:
        data_dict (dict): dictionary of variable names and numpy arrays
    """

    for var in data_dict:

        if data_dict[var].size == 1:

            data_dict[var] = data_dict[var][0]


# Transform a dictionary of numpy arrays into a dictionary of gt4py
# storages of shape (iie-iis+1, jje-jjs+1, kke-kks+1)
def numpy_dict_to_gt4py_dict(np_dict, vars, rank1_flag=False):
    """convert dictionary of numpy arrays from serialized data to GT4Py storages

    Args:
        np_dict (dict): dictionary of variables names and numpy arrays
        vars (dict): dictionary of variable names, shapes and types

    Returns:
        dict: dictionary of variable names and storages
    """

    gt4py_dict = {}

    for var in vars.keys():

        data = np_dict[var]
        shape = vars[var]["shape"]
        type = vars[var]["type"]

        if len(shape) == 1:
            tmp = np.tile(data[:, None], (1, 1))
            if var == "idxday":
                tmp2 = np.zeros(npts, dtype=bool)
                for n in range(npts):
                    if rank1_flag:
                        if tmp[n] > 1 and tmp[n] < 25:
                            tmp2[tmp[n] - 1] = True
                    else:
                        if tmp[n] > 0 and tmp[n] < 25:
                            tmp2[tmp[n] - 1] = True

                tmp = np.tile(tmp2[:, None], (1, 1))
        elif len(shape) == 2:
            if shape[1] == nlay:
                data = np.insert(data, 0, 0, axis=1)
                tmp = np.tile(data[:, None, :], (1, 1, 1))
            elif shape[1] == nlp1:
                tmp = np.tile(data[:, None, :], (1, 1, 1))
            elif shape[1] == 4:
                tmp = np.tile(data[:, None, None, :], (1, 1, nlp1, 1))
        elif len(shape) == 3:
            data = np.insert(data, 0, 0, axis=1)
            tmp = np.tile(data[:, None, :, :], (1, 1, 1, 1))
        elif len(shape) == 4:
            data = np.insert(data, 0, 0, axis=1)
            tmp = np.tile(data[:, None, :, :, :], (1, 1, 1, 1, 1))
        else:
            tmp = data

        if data.size > 1:
            if tmp.ndim == 2:
                gt4py_dict[var] = create_storage_from_array(
                    tmp, backend, shape_2D, type
                )
            else:
                gt4py_dict[var] = create_storage_from_array(
                    tmp, backend, shape_nlp1, type
                )
        else:
            gt4py_dict[var] = deepcopy(data)

    return gt4py_dict


def create_gt4py_dict_zeros(indict):
    """create a dictionary of GT4Py storages initialized to zeros

    Args:
        indict (dict): dictionary of variable names, shapes and types

    Returns:
        dict: dictionart of variable names and storages
    """
    gt4py_dict = {}

    for var in indict.keys():
        gt4py_dict[var] = create_storage_zeros(
            backend, indict[var]["shape"], indict[var]["type"]
        )

    return gt4py_dict


# Cast a dictionary of gt4py storages into dictionary of numpy arrays
def view_gt4py_storage(gt4py_dict):
    """convert dictionary of storages to numpy arrays

    Args:
        gt4py_dict (dict): dictionary of variable names and storages

    Returns:
        dict: dictionary of variable names and numpy arrays
    """

    np_dict = {}

    for var in gt4py_dict:

        data = gt4py_dict[var]

        np_dict[var] = np.squeeze(data.view(np.ndarray))

    return np_dict


def convert_gt4py_output_for_validation(datadict, infodict):
    """reshape stencil output to compare to fortran output

    Args:
        datadict (dict): dictionary of variable names and numpy arrays
        infodict (dict): dictionary of variable names and shapes from fortran

    Returns:
        dict: dictionary of variable names and numpy arrays
    """
    npdict = view_gt4py_storage(datadict)

    outdict = dict()

    for var in infodict.keys():

        if var != "laytrop":
            data = npdict[var]
            target_shape = infodict[var]["fortran_shape"]

            if data.shape != target_shape:
                if data.shape[1] == target_shape[1] + 1:
                    outdict[var] = data[:, 1:, ...]
                elif data.shape[1] == target_shape[1] - 1:
                    outdict[var] = np.append(data, np.zeros((npts, 1)), axis=1)
                elif target_shape[1] not in [nlay, nlp1] and len(target_shape) == 2:
                    outdict[var] = data[:, 0, :].squeeze()
                elif data.shape[1] in [nlay, nlp1] and target_shape[2] in [nlay, nlp1]:
                    outdict[var] = np.transpose(data, (0, 2, 1))
                    if outdict[var].shape[2] == target_shape[2] + 1:
                        outdict[var] = outdict[var][:, :, 1:, ...]
            else:
                outdict[var] = data
        else:
            outdict[var] = (
                npdict[var][:, 1:]
                .squeeze()
                .view(np.ndarray)
                .astype(DTYPE_INT)
                .sum(axis=1)
            )

    return outdict


def compare_data(data, ref_data, explicit=True, blocking=True):
    """test whether stencil output matches fortran output

    Args:
        data (dict): dictionary of variable names and stencil output
        ref_data (dict): dictionary of variable names and fortran output
        explicit (bool, optional): Flag to print result. Defaults to True.
        blocking (bool, optional): Flag to make failure block progress. Defaults to True.
    """

    wrong = []
    flag = True

    for var in data:

        if not np.allclose(
            data[var], ref_data[var], rtol=1e-11, atol=1.0e-13, equal_nan=True
        ):

            wrong.append(var)
            flag = False

        else:

            if explicit:
                print(f"Successfully validated {var}!")

    if blocking:
        assert flag, f"Output data does not match reference data for field {wrong}!"
    else:
        if not flag:
            print(f"Output data does not match reference data for field {wrong}!")


def create_storage_from_array(
    var, backend, shape, dtype, default_origin=default_origin
):
    """Create GT4Py storage from numpy array

    Args:
        var (ndarray): array of data
        backend (str): GT4Py backend
        shape (tuple): shape of array
        dtype (type): type of storage
        default_origin (tuple, optional): origin of storage. Defaults to default_origin.

    Returns:
        Storage: gt4py storage containing data
    """
    out = gt4py.storage.from_array(
        var, backend=backend, default_origin=default_origin, shape=shape, dtype=dtype
    )
    return out


def create_storage_zeros(backend, shape, dtype):
    """Create GT4Py storage initialized to zeros

    Args:
        backend (str): GT4Py backend
        shape (tuple): shape of data
        dtype (type): type of data

    Returns:
        Storage: storage containing zeros
    """
    out = gt4py.storage.zeros(
        backend=backend, default_origin=default_origin, shape=shape, dtype=dtype
    )
    return out


def create_storage_ones(backend, shape, dtype):
    """Create GT4Py storage initialized to ones

    Args:
        backend (str): GT4Py backend
        shape (tuple): shape of data
        dtype (type): type of data

    Returns:
        Storage: storage containing ones
    """
    out = gt4py.storage.ones(
        backend=backend, default_origin=default_origin, shape=shape, dtype=dtype
    )
    return out


def loadlookupdata(name, scheme):
    """Load lookup table data for the given subroutine
    This is a workaround for now, in the future this could change to a dictionary
    or some kind of map object when gt4py gets support for lookup tables

    Args:
        name (str): name of fortran module that contained the data originally
        scheme (str): name of radiation scheme, "radlw" or "radsw"

    Returns:
        dict: dictionary containing storages with the data
    """
    ds = xr.open_dataset(os.path.join(LOOKUP_DIR, scheme + "_" + name + "_data.nc"))

    lookupdict = dict()
    lookupdict_gt4py = dict()

    for var in ds.data_vars.keys():
        if len(ds.data_vars[var].shape) == 1:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :], (npts, 1, nlp1, 1)
            )
        elif len(ds.data_vars[var].shape) == 2:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :, :], (npts, 1, nlp1, 1, 1)
            )
        elif len(ds.data_vars[var].shape) == 3:
            lookupdict[var] = np.tile(
                ds[var].data[None, None, None, :, :, :], (npts, 1, nlp1, 1, 1, 1)
            )

        if len(ds.data_vars[var].shape) >= 1:
            lookupdict_gt4py[var] = create_storage_from_array(
                lookupdict[var], backend, shape_nlp1, (ds[var].dtype, ds[var].shape)
            )
        else:
            lookupdict_gt4py[var] = float(ds[var].data)

    ds2 = xr.open_dataset("../lookupdata/radlw_ref_data.nc")
    tmp = np.tile(ds2["chi_mls"].data[None, None, None, :, :], (npts, 1, nlp1, 1, 1))

    lookupdict_gt4py["chi_mls"] = create_storage_from_array(
        tmp, backend, shape_nlp1, (DTYPE_FLT, ds2["chi_mls"].shape)
    )

    return lookupdict_gt4py
