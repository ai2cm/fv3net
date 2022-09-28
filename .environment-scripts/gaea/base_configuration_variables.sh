#!/bin/bash

# NCEPlibs arguments
NCEPLIBS_PLATFORM=gaea
NCEPLIBS_COMPILER=intel

# ESMF arguments and environment variables
ESMF_OS=Unicos
ESMF_COMPILER=intel
ESMF_SITE=default
ESMF_CC=cc

# FMS environment variables
FMS_CC=cc
FMS_FC=ftn
FMS_LDFLAGS=
FMS_LOG_DRIVER_FLAGS=
FMS_CPPFLAGS='-Duse_LARGEFILE -DMAXFIELDMETHODS_=500 -DGFS_PHYS'
FMS_FCFLAGS='-FR -i4 -r8'
