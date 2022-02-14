import xarray


def gscond_tendency(data: xarray.Dataset, field: str, source: str) -> xarray.DataArray:
    if field == "cloud_water" and source == "emulator":
        return -data[f"tendency_of_specific_humidity_due_to_gscond_{source}"]
    else:
        return data[f"tendency_of_{field}_due_to_gscond_{source}"]


def total_tendency(data: xarray.Dataset, field: str, source: str) -> xarray.DataArray:
    return data[f"tendency_of_{field}_due_to_zhao_carr_{source}"]


def precpd_tendency(data: xarray.Dataset, field: str, source: str) -> xarray.DataArray:
    return total_tendency(data, field, source) - gscond_tendency(data, field, source)
