from vcm import mask_to_surface_type
import xarray as xr

SOIL_THICKNESS = xr.DataArray([0.1, 0.3, 0.6, 1.0], dims=["soil_layer"])


def remove_suffixes(ds, suffixes=["_coarse", "sfc"]):
    rename_dict = {}
    for variable in ds.variables:
        for suffix in suffixes:
            if suffix in variable:
                rename_dict[variable] = variable.split(suffix)[0]
    return ds.rename(rename_dict)


def drop_uninformative_coords(
    ds, coord_list=["grid_x", "grid_y", "grid_xt", "grid_yt"]
):
    drop_coords = [coord for coord in ds.coords if coord in coord_list]
    return ds.drop_vars(names=drop_coords)


def sum_soil_moisture(ds, thicknesses=SOIL_THICKNESS):
    sm_total = (1000 * SOIL_THICKNESS * ds["smc"]).sum("soil_layer")
    sm_total = sm_total.assign_attrs(
        {"long_name": "column-integrated soil moisture storage", "units": "kg/m**2"}
    )
    return ds.drop_vars(names=["smc"]).assign({"SOILM": sm_total})


def mask_soilm_to_land(ds, grid):
    ds_copy = ds.copy().merge(grid)
    mask_ds = mask_to_surface_type(ds_copy, "land", surface_type_var="land_sea_mask")
    return ds.assign({"SOILM": mask_ds["SOILM"]})


def top_soil_temperature_only(ds):
    st_top = ds["stc"].isel(soil_layer=0)
    return ds.drop_vars("stc").assign({"SOILT1": st_top})


def rename_reference_vars(ds, rename_dict={"t": "TMP", "q": "SPFH"}):
    rename_vars = {}
    for data_var in ds.data_vars:
        for old_var, new_var in rename_dict.items():
            if old_var in data_var:
                rename_vars[data_var] = data_var.replace(old_var, new_var)
    return ds.rename(rename_vars)


def precip_units(ds):
    precip = 86400 * ds["PRATE"]
    precip.attrs.update({"long_name": "surface_precipitation", "units": "mm/day"})
    return ds.assign({"PRATE": precip})


def concat_and_differences(ds1, ds2, diff_coord="derivation"):
    differences = {}
    for common_var in set(ds1.data_vars) & set(ds2.data_vars):
        differences[common_var] = ds1[common_var] - ds2[common_var]
    difference_ds = xr.Dataset(differences).expand_dims(diff_coord)
    return xr.concat([ds1, ds2, difference_ds], dim="derivation")


def global_average(da, area_da, mask_da, surface_type, average_dims=["tile", "x", "y"]):
    weights = mask_to_surface_type(
        xr.merge([area_da, mask_da]), surface_type, surface_type_var=mask_da.name
    )[area_da.name]
    return ((da * weights).sum(average_dims)) / weights.sum(average_dims)
