#!/bin/bash

bash build.sh \
     /lustre/f2/dev/$USER/2022-09-08-general/clone \
     /lustre/f2/dev/$USER/2022-09-08-general/install \
     gaea \
     gaea \
     intel \
     Unicos \
     intel \
     default \
     "CPP_FLAGS='-Duse_LARGEFILE -DMAXFIELDMETHODS_=500 -DGFS_PHYS' FCFLAGS='-g -FR -i4 -r8'" \
     gaea
