import numpy as np
from scipy.special import gamma

grav = 9.80665  # gfs: acceleration due to gravity
rhow = 1.0e3  # density of cloud water (kg/m^3)
rhoi = 9.17e2  # density of cloud ice (kg/m^3)
rhor = 1.0e3  # density of rain (kg/m^3)
rhos = 0.1e3  # density of cloud snow (kg/m^3)
rhog = 0.4e3  # density of cloud graupel (kg/m^3)
rdgas = 287.05  # gfs: gas constant for dry air
rvgas = 461.50  # gfs: gas constant for water vapor
zvir = rvgas / rdgas - 1.0  # 0.6077338443
pi = 3.1415926535897931  # gfs: ratio of circle circumference to diameter

ccn_o = 90.0  # ccn over ocean (cm^ - 3)
ccn_l = 270.0  # ccn over land (cm^ - 3)

qcmin = 1.0e-12  # min value for cloud condensates

rewmin = 5.0
rewmax = 15.0
reimin = 10.0
reimax = 150.0
rermin = 15.0
rermax = 10000.0
resmin = 150.0
resmax = 10000.0
regmin = 150.0
regmax = 10000.0

n0r_sig = 8.0  # intercept parameter (significand) of rain (Lin et al. 1983) (1/m^4)
# (Marshall and Palmer 1948)
n0s_sig = 3.0  # intercept parameter (significand) of snow (Lin et al. 1983) (1/m^4)
# (Gunn and Marshall 1958)
n0g_sig = 4.0  # intercept parameter (significand) of graupel (Rutledge and Hobbs 1984)
# (1/m^4) (Houze et al. 1979)

n0r_exp = 6  # intercept parameter (exponent) of rain (Lin et al. 1983) (1/m^4)
# (Marshall and Palmer 1948)
n0s_exp = 6  # intercept parameter (exponent) of snow (Lin et al. 1983) (1/m^4)
# (Gunn and Marshall 1958)
n0g_exp = (
    6  # intercept parameter (exponent) of graupel (Rutledge and Hobbs 1984) (1/m^4)
)
# (Houze et al. 1979)

mur = 1.0  # shape parameter of rain in Gamma distribution (Marshall and Palmer 1948)
mus = 1.0  # shape parameter of snow in Gamma distribution (Gunn and Marshall 1958)
mug = 1.0  # shape parameter of graupel in Gamma distribution (Houze et al. 1979)

edar = (
    np.exp(-1.0 / (mur + 3) * np.log(n0r_sig))
    * (mur + 2)
    * np.exp(-n0r_exp / (mur + 3) * np.log(10.0))
)
edas = (
    np.exp(-1.0 / (mus + 3) * np.log(n0s_sig))
    * (mus + 2)
    * np.exp(-n0s_exp / (mus + 3) * np.log(10.0))
)
edag = (
    np.exp(-1.0 / (mug + 3) * np.log(n0g_sig))
    * (mug + 2)
    * np.exp(-n0g_exp / (mug + 3) * np.log(10.0))
)

edbr = np.exp(1.0 / (mur + 3) * np.log(pi * rhor * gamma(mur + 3)))
edbs = np.exp(1.0 / (mus + 3) * np.log(pi * rhos * gamma(mus + 3)))
edbg = np.exp(1.0 / (mug + 3) * np.log(pi * rhog * gamma(mug + 3)))

retab = [
    0.05000,
    0.05000,
    0.05000,
    0.05000,
    0.05000,
    0.05000,
    0.05500,
    0.06000,
    0.07000,
    0.08000,
    0.09000,
    0.10000,
    0.20000,
    0.30000,
    0.40000,
    0.50000,
    0.60000,
    0.70000,
    0.80000,
    0.90000,
    1.00000,
    1.10000,
    1.20000,
    1.30000,
    1.40000,
    1.50000,
    1.60000,
    1.80000,
    2.00000,
    2.20000,
    2.40000,
    2.60000,
    2.80000,
    3.00000,
    3.20000,
    3.50000,
    3.80000,
    4.10000,
    4.40000,
    4.70000,
    5.00000,
    5.30000,
    5.60000,
    5.92779,
    6.26422,
    6.61973,
    6.99539,
    7.39234,
    7.81177,
    8.25496,
    8.72323,
    9.21800,
    9.74075,
    10.2930,
    10.8765,
    11.4929,
    12.1440,
    12.8317,
    13.5581,
    14.2319,
    15.0351,
    15.8799,
    16.7674,
    17.6986,
    18.6744,
    19.6955,
    20.7623,
    21.8757,
    23.0364,
    24.2452,
    25.5034,
    26.8125,
    27.7895,
    28.6450,
    29.4167,
    30.1088,
    30.7306,
    31.2943,
    31.8151,
    32.3077,
    32.7870,
    33.2657,
    33.7540,
    34.2601,
    34.7892,
    35.3442,
    35.9255,
    36.5316,
    37.1602,
    37.8078,
    38.4720,
    39.1508,
    39.8442,
    40.5552,
    41.2912,
    42.0635,
    42.8876,
    43.7863,
    44.7853,
    45.9170,
    47.2165,
    48.7221,
    50.4710,
    52.4980,
    54.8315,
    57.4898,
    60.4785,
    63.7898,
    65.5604,
    71.2885,
    75.4113,
    79.7368,
    84.2351,
    88.8833,
    93.6658,
    98.5739,
    103.603,
    108.752,
    114.025,
    119.424,
    124.954,
    130.630,
    136.457,
    142.446,
    148.608,
    154.956,
    161.503,
    168.262,
    175.248,
    182.473,
    189.952,
    197.699,
    205.728,
    214.055,
    222.694,
    231.661,
    240.971,
    250.639,
]
