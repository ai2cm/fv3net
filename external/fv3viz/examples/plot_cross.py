"""
"Cross" plot
==========================

The routine ``vcm.cubedsphere.to_cross`` is useful for quick plots when
cartopy may not be installed.
"""

import vcm
import fsspec

url = "https://github.com/VulcanClimateModeling/vcm-ml-example-data/blob/main/fv3net/fv3viz/plot_2_plot_cube_prognostic_diags.nc?raw=true"  # noqa
fs = fsspec.get_fs_token_paths(url)[0]
ds = vcm.open_remote_nc(fs, url).load()

cross = vcm.cubedsphere.to_cross(ds["h500_time_mean_value"], x="x", y="y", tile="tile")
cross.plot.contourf(levels=31)

# need this to avoid hdf5 related error for some reason
del ds
