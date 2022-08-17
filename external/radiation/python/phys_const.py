import numpy as np

# Physical constants

# Math constants
con_pi = 4.0 * np.arctan(1.0)
con_sqrt2 = 1.414214
con_sqrt3 = 1.732051

# Geophysics/Astronomy constants

# radius of earth (m)
con_rerth = 6.3712e6
# gravity (\f$m/s^{2}\f$)
con_g = 9.80665
# ang vel of earth (\f$s^{-1}\f$)
con_omega = 7.2921e-5
# std atms pressure (pa)
con_p0 = 1.01325e5
# solar constant (\f$W/m^{2}\f$)-liu(2002)
con_solr_old = 1.3660e3
# solar constant (\f$W/m^{2}\f$)-nasa-sorce Tim(2008)
con_solr = 1.3608e3

# Thermodynamics constants

# molar gas constant (\f$J/mol/K\f$)
con_rgas = 8.314472
# gas constant air (\f$J/kg/K\f$)
con_rd = 2.8705e2
# gas constant H2O (\f$J/kg/K\f$)
con_rv = 4.6150e2
# spec heat air at p (\f$J/kg/K\f$)
con_cp = 1.0046e3
# spec heat air at v (\f$J/kg/K\f$)
con_cv = 7.1760e2
# spec heat H2O gas (\f$J/kg/K\f$)
con_cvap = 1.8460e3
# spec heat H2O liq (\f$J/kg/K\f$)
con_cliq = 4.1855e3
# spec heat H2O ice (\f$J/kg/K\f$)
con_csol = 2.1060e3
# lat heat H2O cond (\f$J/kg\f$)
con_hvap = 2.5000e6
# lat heat H2O fusion (\f$J/kg\f$)
con_hfus = 3.3358e5
# pres at H2O 3pt (Pa)
con_psat = 6.1078e2
# temp at 0C (K)
con_t0c = 2.7315e2
# temp at H2O 3pt (K)
con_ttp = 2.7316e2
# temp freezing sea (K)
con_tice = 2.7120e2
# joules per calorie
con_jcal = 4.1855
# sea water reference density (\f$kg/m^{3}\f$)
con_rhw0 = 1022.0
# min q for computing precip type
con_epsq = 1.0e-12

# Secondary constants

con_rocp = con_rd / con_cp
con_cpor = con_cp / con_rd
con_rog = con_rd / con_g
con_fvirt = con_rv / con_rd - 1.0
con_eps = con_rd / con_rv
con_epsm1 = con_rd / con_rv - 1.0
con_dldt = con_cvap - con_cliq
con_xpona = -con_dldt / con_rv
con_xponb = -con_dldt / con_rv + con_hvap / (con_rv * con_ttp)

# Other Physics/Chemistry constants (source: 2002 CODATA)

# speed of light (\f$m/s\f$)
con_c = 2.99792458e8
# planck constant (\f$J/s\f$)
con_plnk = 6.6260693e-34
# boltzmann constant (\f$J/K\f$)
con_boltz = 1.3806505e-23
# stefan-boltzmann (\f$W/m^{2}/K^{4}\f$)
con_sbc = 5.670400e-8
# avogadro constant (\f$mol^{-1}\f$)
con_avgd = 6.0221415e23
# vol of ideal gas at 273.15K, 101.325kPa (\f$m^{3}/mol\f$)
con_gasv = 22413.996e-6
# molecular wght of dry air (\f$g/mol\f$)
con_amd = 28.9644
# molecular wght of water vapor (\f$g/mol\f$)
con_amw = 18.0154
# molecular wght of o3 (\f$g/mol\f$)
con_amo3 = 47.9982
# molecular wght of co2 (\f$g/mol\f$)
con_amco2 = 44.011
# molecular wght of o2 (\f$g/mol\f$)
con_amo2 = 31.9999
# molecular wght of ch4 (\f$g/mol\f$)
con_amch4 = 16.043
# molecular wght of n2o (\f$g/mol\f$)
con_amn2o = 44.013
# temperature the H.G.Nuc. ice starts
con_thgni = -38.15

# minimum aerosol concentration
qamin = 1.0e-16

# Miscellaneous physics related constants (Moorthi - Jul 2014)
rlapse = 0.65e-2
b2mb = 10.0
pa2mb = 0.01
# for wsm6
rhowater = 1000.0  # density of water (kg/m^3)
rhosnow = 100.0  # density of snow (kg/m^3)
rhoair = 1.28  # density of air near surface (kg/m^3)

# For min/max hourly rh02m and t02m
PQ0 = 379.90516
A2A = 17.2693882
A3 = 273.16
A4 = 35.86
RHmin = 1.0e-6

# Derived constants
amdw = con_amd / con_amw
amdo3 = con_amd / con_amo3
