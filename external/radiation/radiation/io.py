import os
import xarray as xr
import numpy as np
import tarfile
import warnings
from vcm.cloud import get_fs


def load_random_numbers(lookup_dir: str, tile_number: int):
    data_dict = {}
    # File names for serialized random numbers in mcica_subcol
    if tile_number == 0:
        sw_rand_file = os.path.join(lookup_dir, "rand2d_sw.nc")
    else:
        sw_rand_file = os.path.join(
            lookup_dir, "rand2d_tile" + str(tile_number) + "_sw.nc"
        )
    lw_rand_file = os.path.join(lookup_dir, "rand2d_tile" + str(tile_number) + "_lw.nc")
    data_dict["sw_rand"] = xr.open_dataset(sw_rand_file)["rand2d"].values
    data_dict["lw_rand"] = xr.open_dataset(lw_rand_file)["rand2d"].values

    return data_dict


def load_lw(lookup_dir: str):
    # data needed in lwrad()
    lw_dict = {}
    dfile = os.path.join(lookup_dir, "totplnk.nc")
    pfile = os.path.join(lookup_dir, "radlw_ref_data.nc")
    lw_dict["totplnk"] = xr.open_dataset(dfile)["totplnk"].values
    lw_dict["preflog"] = xr.open_dataset(pfile)["preflog"].values
    lw_dict["tref"] = xr.open_dataset(pfile)["tref"].values
    lw_dict["chi_mls"] = xr.open_dataset(pfile)["chi_mls"].values

    # loading data for cldprop in lwrad()
    ds = xr.open_dataset(os.path.join(lookup_dir, "radlw_cldprlw_data.nc"))
    lw_dict["absliq1"] = ds["absliq1"].values
    lw_dict["absice0"] = ds["absice0"].values
    lw_dict["absice1"] = ds["absice1"].values
    lw_dict["absice2"] = ds["absice2"].values
    lw_dict["absice3"] = ds["absice3"].values

    # loading data for taumol
    varname_bands = [
        "radlw_kgb01",
        "radlw_kgb02",
        "radlw_kgb03",
        "radlw_kgb04",
        "radlw_kgb05",
        "radlw_kgb06",
        "radlw_kgb07",
        "radlw_kgb08",
        "radlw_kgb09",
        "radlw_kgb10",
        "radlw_kgb11",
        "radlw_kgb12",
        "radlw_kgb13",
        "radlw_kgb14",
        "radlw_kgb15",
        "radlw_kgb16",
    ]

    varnames_per_band = {
        "radlw_kgb01": [
            "selfref",
            "forref",
            "ka_mn2",
            "absa",
            "absb",
            "fracrefa",
            "fracrefb",
        ],
        "radlw_kgb02": ["selfref", "forref", "absa", "absb", "fracrefa", "fracrefb"],
        "radlw_kgb03": [
            "selfref",
            "forref",
            "ka_mn2o",
            "kb_mn2o",
            "absa",
            "absb",
            "fracrefa",
            "fracrefb",
        ],
        "radlw_kgb04": ["selfref", "forref", "absa", "absb", "fracrefa", "fracrefb"],
        "radlw_kgb05": [
            "selfref",
            "forref",
            "absa",
            "absb",
            "fracrefa",
            "fracrefb",
            "ka_mo3",
            "ccl4",
        ],
        "radlw_kgb06": [
            "selfref",
            "forref",
            "absa",
            "fracrefa",
            "ka_mco2",
            "cfc11adj",
            "cfc12",
        ],
        "radlw_kgb07": [
            "selfref",
            "forref",
            "absa",
            "absb",
            "fracrefa",
            "fracrefb",
            "ka_mco2",
            "kb_mco2",
        ],
        "radlw_kgb08": [
            "selfref",
            "forref",
            "absa",
            "absb",
            "fracrefa",
            "fracrefb",
            "ka_mo3",
            "ka_mco2",
            "kb_mco2",
            "cfc12",
            "ka_mn2o",
            "kb_mn2o",
            "cfc22adj",
        ],
        "radlw_kgb09": [
            "selfref",
            "forref",
            "absa",
            "absb",
            "fracrefa",
            "fracrefb",
            "ka_mn2o",
            "kb_mn2o",
        ],
        "radlw_kgb10": ["selfref", "forref", "absa", "absb", "fracrefa", "fracrefb"],
        "radlw_kgb11": [
            "selfref",
            "forref",
            "absa",
            "absb",
            "fracrefa",
            "fracrefb",
            "ka_mo2",
            "kb_mo2",
        ],
        "radlw_kgb12": ["selfref", "forref", "absa", "fracrefa"],
        "radlw_kgb13": [
            "selfref",
            "forref",
            "absa",
            "fracrefa",
            "fracrefb",
            "ka_mco2",
            "ka_mco",
            "kb_mo3",
        ],
        "radlw_kgb14": ["selfref", "forref", "absa", "absb", "fracrefa", "fracrefb"],
        "radlw_kgb15": ["selfref", "forref", "absa", "fracrefa", "ka_mn2"],
        "radlw_kgb16": ["selfref", "forref", "absa", "absb", "fracrefa", "fracrefb"],
    }

    for nband in varname_bands:
        data = xr.open_dataset(os.path.join(lookup_dir, nband + "_data.nc"))
        tmp = {}
        for var in varnames_per_band[nband]:
            tmp[var] = data[var].values
        lw_dict[nband] = tmp

    return lw_dict


def load_sw(lookup_dir: str):
    sw_dict = {}

    ds = xr.open_dataset(os.path.join(lookup_dir, "radsw_sflux_data.nc"))
    sw_dict["strrat"] = ds["strrat"].values
    sw_dict["specwt"] = ds["specwt"].values
    sw_dict["layreffr"] = ds["layreffr"].values
    sw_dict["ix1"] = ds["ix1"].values
    sw_dict["ix2"] = ds["ix2"].values
    sw_dict["ibx"] = ds["ibx"].values
    sw_dict["sfluxref01"] = ds["sfluxref01"].values
    sw_dict["sfluxref02"] = ds["sfluxref02"].values
    sw_dict["sfluxref03"] = ds["sfluxref03"].values
    sw_dict["scalekur"] = ds["scalekur"].values
    # data loading for setcoef
    ds = xr.open_dataset(os.path.join(lookup_dir, "radsw_ref_data.nc"))
    sw_dict["preflog"] = ds["preflog"].values
    sw_dict["tref"] = ds["tref"].values
    # load data for cldprop
    ds_cldprtb = xr.open_dataset(os.path.join(lookup_dir, "radsw_cldprtb_data.nc"))
    var_names = [
        "extliq1",
        "extliq2",
        "ssaliq1",
        "ssaliq2",
        "asyliq1",
        "asyliq2",
        "extice2",
        "ssaice2",
        "asyice2",
        "extice3",
        "ssaice3",
        "asyice3",
        "abari",
        "bbari",
        "cbari",
        "dbari",
        "ebari",
        "fbari",
        "b0s",
        "b1s",
        "b0r",
        "b0r",
        "c0s",
        "c0r",
        "a0r",
        "a1r",
        "a0s",
        "a1s",
    ]

    for var in var_names:
        sw_dict[var] = ds_cldprtb[var].values

    # loading data for taumol
    varname_bands = [
        "radsw_kgb16",
        "radsw_kgb17",
        "radsw_kgb18",
        "radsw_kgb19",
        "radsw_kgb20",
        "radsw_kgb21",
        "radsw_kgb22",
        "radsw_kgb23",
        "radsw_kgb24",
        "radsw_kgb25",
        "radsw_kgb26",
        "radsw_kgb27",
        "radsw_kgb28",
        "radsw_kgb29",
    ]

    varnames_per_band = {
        "radsw_kgb16": ["selfref", "forref", "absa", "absb", "rayl"],
        "radsw_kgb17": ["selfref", "forref", "absa", "absb", "rayl"],
        "radsw_kgb18": ["selfref", "forref", "absa", "absb", "rayl"],
        "radsw_kgb19": ["selfref", "forref", "absa", "absb", "rayl"],
        "radsw_kgb20": ["selfref", "forref", "absa", "absb", "absch4", "rayl"],
        "radsw_kgb21": ["selfref", "forref", "absa", "absb", "rayl"],
        "radsw_kgb22": ["selfref", "forref", "absa", "absb", "rayl"],
        "radsw_kgb23": ["selfref", "forref", "absa", "rayl", "givfac"],
        "radsw_kgb24": [
            "selfref",
            "forref",
            "absa",
            "absb",
            "abso3a",
            "abso3b",
            "rayla",
            "raylb",
        ],
        "radsw_kgb25": ["absa", "abso3a", "abso3b", "rayl"],
        "radsw_kgb26": ["rayl"],
        "radsw_kgb27": ["absa", "absb", "rayl"],
        "radsw_kgb28": ["absa", "absb", "rayl"],
        "radsw_kgb29": [
            "forref",
            "absa",
            "absb",
            "selfref",
            "absh2o",
            "absco2",
            "rayl",
        ],
    }

    for nband in varname_bands:
        data = xr.open_dataset(os.path.join(lookup_dir, nband + "_data.nc"))
        tmp = {}
        for var in varnames_per_band[nband]:
            tmp[var] = data[var].values
        sw_dict[nband] = tmp

    return sw_dict


def load_sigma(restart_dir, p_ref=101325.0):
    """Get sigma coordiante of vertical interfaces (approximation) used by
    radiation scheme. See https://github.com/ai2cm/fv3gfs-fortran/blob/
    5d40389e5c8f5696d165a33395660216f99c502c/FV3/gfsphysics/GFS_layer/
    GFS_driver.F90#L314
    """
    file_name = os.path.join(restart_dir, "fv_core.res.nc")
    da = xr.open_dataset(file_name)
    ak = da.ak.squeeze()
    bk = da.bk.squeeze()
    sigma = ((ak + p_ref * bk - ak[0]) / (p_ref - ak[0]))[::-1]
    return sigma.values


def load_aerosol(forcing_dir: str):
    aeros_file = os.path.join(forcing_dir, "aerosol.nc")
    var_names = [
        "kprfg",
        "kprfg",
        "idxcg",
        "cmixg",
        "denng",
        "cline",
        "iendwv",
        "haer",
        "prsref",
        "rhidext0",
        "rhidsca0",
        "rhidssa0",
        "rhidasy0",
        "rhdpext0",
        "rhdpsca0",
        "rhdpssa0",
        "rhdpasy0",
        "straext0",
    ]
    data_dict = {}
    ds = xr.open_dataset(aeros_file)
    for var in var_names:
        data_dict[var] = ds[var].values

    return data_dict


def load_astronomy(forcing_dir, isolar):
    # external solar constant data table,solarconstant_noaa_a0.txt

    if isolar == 1:  # noaa ann-tile_numberan tsi in absolute scale
        file_name = "solarconstant_noaa_a0.nc"
        solar_file = os.path.join(forcing_dir, file_name)

    elif isolar == 2:  # noaa ann-tile_numberan tsi in tim scale
        file_name = "solarconstant_noaa_an.nc"
        solar_file = os.path.join(forcing_dir, file_name)

    elif isolar == 3:  # cmip5 ann-tile_numberan tsi in tim scale
        file_name = "solarconstant_cmip_an.nc"
        solar_file = os.path.join(forcing_dir, file_name)

    elif isolar == 4:  # cmip5 mon-tile_numberan tsi in tim scale
        file_name = "solarconstant_cmip_mn.nc"
        solar_file = os.path.join(forcing_dir, file_name)

    else:
        warnings.warn(
            "- !!! ERROR in selection of solar constant data",
            f" source, ISOL = {isolar}",
        )
        raise FileNotFoundError(" !!! ERROR! Can not find solar constant file!!!")

    data = xr.open_dataset(solar_file)
    return solar_file, data


def load_sfc(forcing_dir: str):
    semis_file = os.path.join(forcing_dir, "semisdata.nc")
    data = xr.open_dataset(semis_file)
    return semis_file, data


def load_gases(forcing_dir, ictmflg):

    if ictmflg == 1:
        cfile1 = os.path.join(forcing_dir, "co2historicaldata_2016.nc")
        var_names = ["iyr", "cline", "co2g1", "co2g2", "co2dat"]
        if not os.path.isfile(cfile1):
            raise FileNotFoundError(
                "   Can not find co2 data source file",
                "*** Stopped in subroutine gas_update !!",
            )

    # Opened CO2 climatology seasonal cycle data
    elif ictmflg == 2:
        cfile1 = os.path.join(forcing_dir, "")
        var_names = ["cline", "co2g1", "co2g2", "co2dat", "gco2cyc"]
        if not os.path.isfile(cfile1):
            raise FileNotFoundError(
                "   Can not find co2 data source file",
                "*** Stopped in subroutine gas_update !!",
            )
    #  --- ...  read in co2 2-d data
    ds = xr.open_dataset(cfile1)
    data_dict = {}
    for var in var_names:
        data_dict[var] = ds[var].values

    return data_dict


def generate_random_numbers(ncolumns, nz, ngptsw, ngptlw, seed=0):
    """Get random numbers needed by cloud overlap scheme"""
    np.random.seed(seed)
    sw_rand = np.random.rand(ncolumns, nz * ngptsw)
    lw_rand = np.random.rand(ncolumns, nz * ngptlw)
    return {"sw_rand": sw_rand, "lw_rand": lw_rand}


def get_remote_tar_data(remote_filepath, local_dir):
    os.makedirs(local_dir)
    fs = get_fs(remote_filepath)
    fs.get(remote_filepath, local_dir)
    local_filepath = os.path.join(local_dir, os.path.basename(remote_filepath))
    tarfile.open(local_filepath).extractall(path=local_dir)
