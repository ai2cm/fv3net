import numpy as np
import sys

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
from phys_const import *

# num of total spectral bands
nbands = 16
# num of total g-points
ngptlw = 140
# lookup table dimension
ntbl = 10000
# max num of absorbing gases
maxgas = 7
# num of halocarbon gasees
maxxsec = 4
# num of ref rates of binary species
nrates = 6
# dim for plank function table
nplnk = 181

NBDLW = nbands

# \name Number of g-point in each band
ng01 = 10
ng02 = 12
ng03 = 16
ng04 = 14
ng05 = 16
ng06 = 8
ng07 = 12
ng08 = 8
ng09 = 12
ng10 = 6
ng11 = 8
ng12 = 8
ng13 = 4
ng14 = 2
ng15 = 2
ng16 = 2

# \name Begining index of each band
ns01 = 0
ns02 = 10
ns03 = 22
ns04 = 38
ns05 = 52
ns06 = 68
ns07 = 76
ns08 = 88
ns09 = 96
ns10 = 108
ns11 = 114
ns12 = 122
ns13 = 130
ns14 = 134
ns15 = 136
ns16 = 138

# band indices for each g-point
ngb = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    13,
    13,
    13,
    13,
    14,
    14,
    15,
    15,
    16,
    16,
]

nspa = [1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9]
nspb = [1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]

eps = 1.0e-6
oneminus = 1 - eps

bpade = 1.0 / 0.278
wtdiff = 0.5
pival = 2.0 * np.arcsin(1.0)
fluxfac = pival * 2.0e4
heatfac = con_g * 1.0e-2 / con_cp


# Band spectrum structures (wavenumber is 1/cm
wvnlw1 = np.array(
    [
        10.0,
        350.0,
        500.0,
        630.0,
        700.0,
        820.0,
        980.0,
        1080.0,
        1180.0,
        1390.0,
        1480.0,
        1800.0,
        2080.0,
        2250.0,
        2380.0,
        2600.0,
    ]
)
wvnlw2 = np.array(
    [
        350.0,
        500.0,
        630.0,
        700.0,
        820.0,
        980.0,
        1080.0,
        1180.0,
        1390.0,
        1480.0,
        1800.0,
        2080.0,
        2250.0,
        2380.0,
        2600.0,
        3250.0,
    ]
)

delwave = np.array(
    [
        340.0,
        150.0,
        130.0,
        70.0,
        120.0,
        160.0,
        100.0,
        100.0,
        210.0,
        90.0,
        320.0,
        280.0,
        170.0,
        130.0,
        220.0,
        650.0,
    ]
)

ipat = [1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]

# absrain is the rain drop absorption coefficient \f$(m^{2}/g)\f$ .
absrain = 0.33e-3  # ncar coeff

# abssnow0 is the snow flake absorption coefficient (micron), fu coeff
abssnow0 = 1.5  # fu   coeff
# abssnow1 is the snow flake absorption coefficient \f$(m^{2}/g)\f$, ncar coeff
abssnow1 = 2.34e-3  # ncar coeff

cldmin = 1.0e-80

a0 = [
    1.66,
    1.55,
    1.58,
    1.66,
    1.54,
    1.454,
    1.89,
    1.33,
    1.668,
    1.66,
    1.66,
    1.66,
    1.66,
    1.66,
    1.66,
    1.66,
]
a1 = [
    0.00,
    0.25,
    0.22,
    0.00,
    0.13,
    0.446,
    -0.10,
    0.40,
    -0.006,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
]
a2 = [
    0.00,
    -12.0,
    -11.7,
    0.00,
    -0.72,
    -0.243,
    0.19,
    -0.062,
    0.414,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
]
