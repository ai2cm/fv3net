"""
Converts a raw tar extraction with high-res surface data to an
input data directory at a specified resolution.
"""


data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile1.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile2.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile3.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile4.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile5.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_core_coarse.res.tile6.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile1.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile2.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile3.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile4.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile5.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_srf_wnd_coarse.res.tile6.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile1.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile2.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile3.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile4.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile5.nc.0001
data/extracted/20160805.170000/20160805.170000.fv_tracer_coarse.res.tile6.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile1.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile2.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile3.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile4.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile5.nc.0001
data/extracted/20160805.170000/20160805.170000.sfc_data.tile6.nc.0001

# table of source name, target name 
categories = [
    ('fv_core_coarse.res', 'fv_core')
    ('fv_src_wnd_coarse.res', 'fv_src_wnd')
    ('fv_tracer_coarse.res', 'fv_tracer')
    ('sfc_data', 'sfc_data')
]

