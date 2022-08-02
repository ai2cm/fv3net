import os
import xarray as xr
import warnings

# this set of functions allow reading datasets needed for the radiation driver
#  Inputs:
# - directory where the data is located in str
# - tile number in int


def random_numbers(lookup_dir: str, tile_number: int):
    ##
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


def lw(lookup_dir: str):
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


def sw(lookup_dir: str):
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


def aerosol(forcing_dir: str):
    aeros_file = os.path.join(forcing_dir, "aerosol.nc")
    if os.path.isfile(aeros_file):
        print(f"Using file {aeros_file}")
    else:
        raise FileNotFoundError(
            f'Requested aerosol data file "{aeros_file}" not found!',
            "*** Stopped in subroutine aero_init !!",
        )
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


def astronomy(forcing_dir, isolar, tile_number):
    # external solar constant data table,solarconstant_noaa_a0.txt

    if tile_number == 0:
        if isolar == 1:  # noaa ann-tile_numberan tsi in absolute scale
            solar_file = "solarconstant_noaa_a0.nc"

            if os.path.isfile(os.path.join(forcing_dir, solar_file)):
                data = xr.open_dataset(os.path.join(forcing_dir, solar_file))
            else:
                warnings.warn(f'Requested solar data file "{solar_file}" not found!',)
                raise FileNotFoundError(
                    " !!! ERROR! Can not find solar constant file!!!"
                )

        elif isolar == 2:  # noaa ann-tile_numberan tsi in tim scale
            solar_file = "solarconstant_noaa_an.nc"
            if os.path.isfile(os.path.join(forcing_dir, solar_file)):
                data = xr.open_dataset(os.path.join(forcing_dir, solar_file))
            else:
                warnings.warn(f'Requested solar data file "{solar_file}" not found!',)
                raise FileNotFoundError(
                    " !!! ERROR! Can not find solar constant file!!!"
                )

        elif isolar == 3:  # cmip5 ann-tile_numberan tsi in tim scale
            solar_file = "solarconstant_cmip_an.nc"
            if os.path.isfile(os.path.join(forcing_dir, solar_file)):
                data = xr.open_dataset(os.path.join(forcing_dir, solar_file))
            else:
                warnings.warn(f'Requested solar data file "{solar_file}" not found!',)
                raise FileNotFoundError(
                    " !!! ERROR! Can not find solar constant file!!!"
                )

        elif isolar == 4:  # cmip5 mon-tile_numberan tsi in tim scale
            solar_file = "solarconstant_cmip_mn.nc"
            if os.path.isfile(os.path.join(forcing_dir, solar_file)):
                data = xr.open_dataset(os.path.join(forcing_dir, solar_file))
            else:
                warnings.warn(f'Requested solar data file "{solar_file}" not found!',)
                raise FileNotFoundError(
                    " !!! ERROR! Can not find solar constant file!!!"
                )

        else:
            warnings.warn(
                "- !!! ERROR in selection of solar constant data",
                f" source, ISOL = {isolar}",
            )
            raise FileNotFoundError(" !!! ERROR! Can not find solar constant file!!!")

    return solar_file, data


def sfc(forcing_dir: str):
    semis_file = os.path.join(forcing_dir, "semisdata.nc")
    data = xr.open_dataset(semis_file)
    return semis_file, data


def gases(forcing_dir, ictmflg):

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
