import numpy as np

## Defining variables

invars = ["idat", "jdat", "fhswr", "dtf", "lsswr"]

statein_vars = [
    "prsi",
    "prsl",
    "tgrs",
    "prslk",
    "qgrs",
]
sfcprop_vars = [
    "tsfc",
    "slmsk",
    "snowd",
    "sncovr",
    "snoalb",
    "zorl",
    "hprime",
    "alvsf",
    "alnsf",
    "alvwf",
    "alnwf",
    "facsf",
    "facwf",
    "fice",
    "tisfc",
]
model_vars = [
    "me",
    "levr",
    "levs",
    "nfxr",
    "ntrac",
    "ntcw",
    "ntiw",
    "ncld",
    "ntrw",
    "ntsw",
    "ntgl",
    "ncnd",
    "fhswr",
    "fhlwr",
    "ntoz",
    "lsswr",
    "solhr",
    "lslwr",
    "imp_physics",
    "lgfdlmprad",
    "uni_cld",
    "effr_in",
    "indcld",
    "ntclamt",
    "num_p3d",
    "npdf3d",
    "ncnvcld3d",
    "lmfdeep2",
    "sup",
    "kdt",
    "lmfshal",
    "do_sfcperts",
    "pertalb",
    "do_only_clearsky_rad",
    "swhtr",
    "solcon",
    "lprnt",
    "lwhtr",
    "lssav",
]

coupling_vars = [
    "nirbmdi",
    "nirdfdi",
    "visbmdi",
    "visdfdi",
    "nirbmui",
    "nirdfui",
    "visbmui",
    "visdfui",
    "sfcnsw",
    "sfcdsw",
    "sfcdlw",
]

radtend_vars_out = [
    "upfxc_s_lw",
    "upfx0_s_lw",
    "dnfxc_s_lw",
    "dnfx0_s_lw",
    "upfxc_s_sw",
    "upfx0_s_sw",
    "dnfxc_s_sw",
    "dnfx0_s_sw",
    "sfalb",
    "htrsw",
    "swhc",
    "semis",
    "tsflw",
    "htrlw",
    "lwhc",
]

radtend_vars = [
    "coszen",
    "coszdg",
    "sfalb",
    "htrsw",
    "swhc",
    "lwhc",
    "semis",
    "tsflw",
]

grid_vars = [
    "xlon",
    "xlat",
    "sinlat",
    "coslat",
]

diag_vars_out = [
    "fluxr",
    "upfxc_t_sw",
    "dnfxc_t_sw",
    "upfx0_t_sw",
    "upfxc_t_lw",
    "upfx0_t_lw",
]

tbd_vars = ["phy_f3d", "icsdsw", "icsdlw"]

diag_vars = ["fluxr"]

vars_dict = {
    "in": invars,
    "statein": statein_vars,
    "sfcprop": sfcprop_vars,
    "model": model_vars,
    "coupling": coupling_vars,
    "radtend_out": radtend_vars_out,
    "radtend": radtend_vars,
    "grid": grid_vars,
    "diag_out": diag_vars_out,
    "tbd": tbd_vars,
    "diag": diag_vars,
}
