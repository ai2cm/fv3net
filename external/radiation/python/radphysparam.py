#  ==========================================================  !!!!!
#                    module physparam description              !!!!!
#  ==========================================================  !!!!!
#                                                                      !
#     This module defines commonly used control variables/parameters   !
#     in physics related programs.                                     !
#                                                                      !
#     Section 1 contains control variables defined in the form of      !
#     parameter. They are pre-determined choices and not adjustable    !
#     during model's run-time.                                         !
#                                                                      !
#     Section 2 contains control variables defined as module variables.!
#     They are more flexible to be changed during run-time by either   !
#     through input namelist, or trough model environment condition.  !
#     They are preassigned here as the default values.                 !
#                                                                      !
#!!!!  ==========================================================  !!!!!
#!
#! Those variables are grouped  together in accordance with functionaity
#! and are given brief descriptions and value specifications. There are
#! two types of attributes (parameters vs. save) designated for the
#! control variables. Those with a "parameter" attribute are prescribed
#! with a preferred option value, while the ones with a "save" attribute
#! are given a default value but could be changed at the model's
#! execution-time (usually through an input of name-list file or through
#! run scripts).
# ========================================!

# ==================================================================================
#  Section - 1 -
#     control flags are pre-set as run-time non-adjuztable parameters.
# ==================================================================================#
# ............................................. !
# Control flags for SW radiation
# ............................................. !

# SW heating rate unit control flag: =1:k/day; =2:k/second.
iswrate = 2

# SW minor gases effect control flag (CH4 and O2): =0:no; =1:yes.
#   =0: minor gases' effects are not included in calculations
#   =1: minor gases' effects are included in calculations
iswrgas = 1

# SW optical property for liquid clouds
#   =0:input cld opt depth, ignoring iswcice setting
#   =1:cloud optical property scheme based on Hu and Stamnes(1993)
#   =2:cloud optical property scheme based on Hu and Stamnes(1993) -updated
iswcliq = 1

# SW optical property for ice clouds (only iswcliq>0)
#   =1:optical property scheme based on Ebert and Curry (1992)
#   =2:optical property scheme based on Streamer v3.0
#   =3:optical property scheme based on Fu's method (1996)
iswcice = 3

# SW control flag for scattering process approximation
#   =1:two-stream delta-eddington    (Joseph et al. 1976
#   =2:two-stream PIFM               (Zdunkowski et al. 1980
#   =3:discrete ordinates (Liou, 1973
iswmode = 2

# ............................................. !
# -1.2- Control flags for LW radiation
# ............................................. !

# LW heating rate unit: =1:k/day; =2:k/second.
ilwrate = 2

# LW minor gases effect control flag (CH4,N2O,O2,and some CFCs):
#   =0: minor gases' effects are not included in calculations
#   =1: minor gases' effects are included in calculations
ilwrgas = 1

# LW optical property scheme for liquid clouds
#   =0:input cloud optical properties directly, not computed within
#   =1:input cwp,rew, use Hu and Stamnes(1993)
ilwcliq = 1

# LW optical property scheme for ice clouds (only ilwcliq>0)
#   =1:optical property scheme based on Ebert and Curry (1992)
#   =2:optical property scheme based on Streamer v3
#   =3:optical property scheme use Fu's method (1998)
ilwcice = 3

# ............................................. !
#   -1.3- Control flag for LW aerosol property

# selects 1 band or multi bands for LW aerosol properties
#   =.true.:aerosol properties calculated in 1 broad LW band
#   =.false.:aerosol properties calculated in all LW bands
#   variable names diff in Opr CFS
lalw1bd = False

# ==================================================================================
#  Section - 2 -
#     values of control flags might be re-set in initialization subroutines
#       (may be adjusted at run time based on namelist input or run condition)
# ==================================================================================

# ............................................. !
#   -2.1- For module radiation_astronomy
# ............................................. !

# solar constant scheme control flag
#   =0:fixed value=1366.0\f$W/m^2\f$(old standard)
#   =10:fixed value=1360.8\f$W/m^2\f$(new standard)
#   =1:NOAA ABS-scale TSI table (yearly) w 11-yr cycle approx
#   =2:NOAA TIM-scale TSI table (yearly) w 11-yr cycle approx
#   =3:CMIP5 TIM-scale TSI table (yearly) w 11-yr cycle approx
#   =4:CMIP5 TIM-scale TSI table (monthly) w 11-yr cycle approx
#   see ISOL in run scripts: Opr GFS=2; Opr CFS=1
isolar = 0

# external solar constant data table,solarconstant_noaa_a0.txt
solar_file = "solarconstant_noaa_an.nc"

# ............................................. !
#   -2.2- For module radiation_aerosols
# ............................................. !

# aerosol model scheme control flag
#   =0:seasonal global distributed OPAC aerosol climatology
#   =1:monthly global distributed GOCART aerosol climatology
#   =2: GOCART prognostic aerosol model
#   =5: OPAC climatoloy with new band mapping
#   Opr GFS=0; Opr CFS=n/a
iaermdl = 0

# aerosol effect control flag
#   3-digit flag 'abc':
#   a-stratospheric volcanic aerols
#   b-tropospheric aerosols for LW
#   c-tropospheric aerosols for SW
#   =0:aerosol effect is not included; =1:aerosol effect is included
#   Opr GFS/CFS =111; see IAER in run scripts
iaerflg = 0

# external aerosols data file: aerosol.dat
aeros_file = "aerosol.nc"

# ............................................. !
#   -2.3- For module radiation_gases
# ............................................. !

# co2 data source control flag
#   =0:prescribed value(380 ppmv)
#   =1:yearly global averaged annual mean from observations
#   =2:monthly 15 degree horizontal resolution from observations
#   Opr GFS/CFS=2; see ICO2 in run scripts
ico2flg = 0

# controls external data at initial time and data usage during
# forecast time
#   =-2:as in 0,but superimpose with seasonal climatology cycle
#   =-1:use user data,no extrapolation in overtime
#   =0:use IC time to select data,no extrapolation in overtime
#   =1:use forecast time to select data,extrapolate when necessary
#   =yyyy0:use yyyy year of data, no extrapolation
#   =yyyy1:use yyyy year of data, extrapolate when necessary
#   Opr GFS/CFS=1; see ICTM in run scripts
ictmflg = 0

# ozone data source control flag
#   =0:use seasonal climatology ozone data
#   >0:use prognostic ozone scheme (also depend on other model control
#      variable at initial time)
ioznflg = 1

# external co2 2d monthly obsv data table: co2historicaldata_2004.txt
# external co2 global annual mean data tb: co2historicaldata_glob.txt
# external co2 user defined data table: co2userdata.txt
# external co2 clim monthly cycle data tb: co2monthlycyc.txt
co2dat_file = "co2historicaldata_2004.nc"
co2gbl_file = "co2historicaldata_glob.txt"
co2usr_file = "co2userdata.txt"
co2cyc_file = "co2monthlycyc.txt"

# ............................................. !
#   -2.4- For module radiation_clouds
# ............................................. !

# cloud optical property scheme control flag
#   =0:use diagnostic cloud scheme for cloud cover and mean optical properties
#   =1:use prognostic cloud scheme for cloud cover and cloud properties
icldflg = 1

# cloud overlapping control flag for SW
#   =0:use random cloud overlapping method
#   =1:use maximum-random cloud overlapping method
#   =2:use maximum cloud overlapping method
#   =3:use decorrelation length overlapping method
#   Opr GFS/CFS=1; see IOVR_SW in run scripts
iovrsw = 1

# cloud overlapping control flag for LW
#   =0:use random cloud overlapping method
#   =1:use maximum-random cloud overlapping method
#   =2:use maximum cloud overlapping method
#   =3:use decorrelation length overlapping method
#   Opr GFS/CFS=1; see IOVR_LW in run scripts
iovrlw = 1

# sub-column cloud approx flag in SW radiation
#   =0:no McICA approximation in SW radiation
#   =1:use McICA with precribed permutation seeds (test mode)
#   =2:use McICA with randomly generated permutation seeds
#   Opr GFS/CFS=2; see ISUBC_SW in run scripts
isubcsw = 0

# sub-column cloud approx flag in LW radiation
#   =0:no McICA approximation in LW radiation
#   =1:use McICA with prescribed permutation seeds (test mode)
#   =2:use McICA with randomly generatedo
#   Opr GFS/CFS=2; see ISUBC_LW in run scripts
isubclw = 0

# eliminating CRICK control flag
lcrick = False
# in-cld condensate control flag
lcnorm = False
# precip effect on radiation flag (Ferrier microphysics)
lnoprec = False
# shallow convetion flag
lsashal = False

# ............................................. !
#   -2.5- For module radiation_surface
# ............................................. !

# surface albedo scheme control flag
#   =0:vegetation type based climatological albedo scheme
#   =1:seasonal albedo derived from MODIS measurements
ialbflg = 0

# surface emissivity scheme control flag
#   =0:black-body surface emissivity(=1.0)
#   =1:vegetation type based climatology emissivity(<1.0)
#   Opr GFS/CFS=1; see IEMS in run scripts
iemsflg = 0

# external sfc emissivity data table: sfc_emissivity_idx.txt
semis_file = "semisdata.nc"

# ............................................. !
#   -2.6- general purpose
# ............................................. !

# vertical profile indexing flag
ivflip = 1

# initial permutaion seed for mcica radiation
ipsd0 = 0
ipsdlim = 1
