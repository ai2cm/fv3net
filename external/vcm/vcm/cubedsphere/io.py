import xarray as xr

from vcm.cubedsphere.coarsen import NUM_TILES, SUBTILE_FILE_PATTERN


def combine_subtiles(subtiles):
    """Combine subtiles of a cubed-sphere dataset
    In v.12 of xarray, with data_vars='all' by default, combined_by_coords
    broadcasts all the variables to a common set of dimensions, dramatically
    increasing the size of the dataset.  In some cases this can be avoided by
    using data_vars='minimal'; however it then fails for combining data
    variables that contain missing values and is also slow due to the fact that
    it does equality checks between variables it combines.

    To work around this issue, we combine the data variables of the dataset one
    at a time with the default data_vars='all', and then merge them back
    together.
    """
    sample_subtile = subtiles[0]
    output_vars = []
    for key in sample_subtile:
        # combine_by_coords currently does not work on DataArrays; see
        # https://github.com/pydata/xarray/issues/3248.
        subtiles_for_var = [subtile[key].to_dataset() for subtile in subtiles]
        combined = xr.combine_by_coords(subtiles_for_var)
        output_vars.append(combined)
    return xr.merge(output_vars)


def _subtile_filenames(prefix, tile, pattern=SUBTILE_FILE_PATTERN, num_subtiles=16):
    for subtile in range(num_subtiles):
        yield pattern.format(prefix=prefix, tile=tile, subtile=subtile)


def all_filenames(prefix, pattern=SUBTILE_FILE_PATTERN, num_subtiles=16):
    filenames = []
    for tile in range(1, NUM_TILES + 1):
        filenames.extend(_subtile_filenames(prefix, tile, pattern, num_subtiles))
    return filenames
