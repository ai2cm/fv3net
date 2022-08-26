import numpy as np

# Parameter constants for SW band structures

# band range lower index
nblow = 16
# band range upper index
nbhgh = 29
# total number of SW bands (14)
nbandssw = nbhgh - nblow + 1
# total number of g-point in all bands
ngptsw = 112
# maximum number of g-point in one band
ngmax = 16
# maximum number of absorbing gases
maxgas = 7
# index upper limit of optical depth and transmittance tables
ntbmx = 10000
# SW bands counter starting index (for compatibility with previous
# SW radiation schemes)
NSWSTR = 1

NSWEND = nbandssw
nbdsw = nbandssw

eps = 1.0e-6
oneminus = 1.0 - eps
# pade approx constant
bpade = 1.0 / 0.278
stpfac = 296.0 / 1013.0
ftiny = 1.0e-12
flimit = 1.0e-20
# internal solar constant
s0 = 1368.22

# uv-b band index
nuvb = 27

nspa = [9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1]
nspb = [1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1]

idxsfc = [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 1]  # band index for sfc flux
idxebc = [5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 5]  # band index for cld prop

eps = 1.0e-6
oneminus = 1.0 - eps
bpade = 1.0 / 0.278
stpfac = 296.0 / 1013.0
ftiny = 1.0e-12
flimit = 1.0e-20
s0 = 1368.22

zcrit = 0.9999995  # thresold for conservative scattering
zsr3 = np.sqrt(3.0)
od_lo = 0.06
eps1 = 1.0e-8

# The actual number of g-point for bands 16-29
NG16 = 6
NG17 = 12
NG18 = 8
NG19 = 8
NG20 = 10
NG21 = 10
NG22 = 2
NG23 = 10
NG24 = 8
NG25 = 6
NG26 = 6
NG27 = 8
NG28 = 6
NG29 = 12

ng = [
    NG16,
    NG17,
    NG18,
    NG19,
    NG20,
    NG21,
    NG22,
    NG23,
    NG24,
    NG25,
    NG26,
    NG27,
    NG28,
    NG29,
]

# Accumulative starting index for bands 16-29
NS16 = 0
NS17 = NS16 + NG16
NS18 = NS17 + NG17
NS19 = NS18 + NG18
NS20 = NS19 + NG19
NS21 = NS20 + NG20
NS22 = NS21 + NG21
NS23 = NS22 + NG22
NS24 = NS23 + NG23
NS25 = NS24 + NG24
NS26 = NS25 + NG25
NS27 = NS26 + NG26
NS28 = NS27 + NG27
NS29 = NS28 + NG28

# array contains values of NS16-NS29
ngs = [
    NS16,
    NS17,
    NS18,
    NS19,
    NS20,
    NS21,
    NS22,
    NS23,
    NS24,
    NS25,
    NS26,
    NS27,
    NS28,
    NS29,
]

# reverse checking of band index for each g-point
NGB = [
    16,
    16,
    16,
    16,
    16,
    16,  # band 16
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,
    17,  # band 17
    18,
    18,
    18,
    18,
    18,
    18,
    18,
    18,  # band 18
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,  # band 19
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,  # band 20
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,  # band 21
    22,
    22,  # band 22
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,  # band 23
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,  # band 24
    25,
    25,
    25,
    25,
    25,
    25,  # band 25
    26,
    26,
    26,
    26,
    26,
    26,  # band 26
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,  # band 27
    28,
    28,
    28,
    28,
    28,
    28,  # band 28
    29,
    29,
    29,
    29,
    29,
    29,
    29,
    29,
    29,
    29,
    29,
    29,
]  # band 29

# Starting/ending wavenumber for each of the SW bands
wvnum1 = np.array(
    [
        2600.0,
        3250.0,
        4000.0,
        4650.0,
        5150.0,
        6150.0,
        7700.0,
        8050.0,
        12850.0,
        16000.0,
        22650.0,
        29000.0,
        38000.0,
        820.0,
    ]
)
wvnum2 = np.array(
    [
        3250.0,
        4000.0,
        4650.0,
        5150.0,
        6150.0,
        7700.0,
        8050.0,
        12850.0,
        16000.0,
        22650.0,
        29000.0,
        38000.0,
        50000.0,
        2600.0,
    ]
)
