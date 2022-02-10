"""Routines for working with the UCAR Common Data Language (CDL)

https://www.unidata.ucar.edu/software/netcdf/workshops/most-recent/nc3model/Cdl.html
"""
import os
import subprocess
import tempfile

import xarray


def cdl_to_dataset(cdl: str) -> xarray.Dataset:
    """Convert a CDL string into a xarray dataset

    Useful for generating synthetic data for testing
    
    CDL is a human readable format with the same data model as netCDF.  CDL can
    be translated to binary netCDF using the `ncgen` command line tool bundled
    with netCDF. CDL is very compact.

    Notes:
        Requires the command line tool ``ncgen``
    
    """
    with tempfile.TemporaryDirectory() as d:
        cdl_path = os.path.join(d, "cdl")
        nc_path = os.path.join(d, "nc")

        with open(cdl_path, "w") as f:
            f.write(cdl)

        subprocess.check_output(["ncgen", "-o", nc_path, cdl_path])
        return xarray.open_dataset(nc_path).load()
