import numpy as np
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


def compare_data(data, ref_data, explicit=True, blocking=True):
    """test whether stencil output matches fortran output

    Args:
        data (dict): dictionary of variable names and stencil output
        ref_data (dict): dictionary of variable names and fortran output
        explicit (bool, optional): Flag to print result. Defaults to True.
        blocking (bool, optional): Flag to make failure block progress.
        Defaults to True.
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
