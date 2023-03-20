#!/bin/bash

set -x

. /spack/share/spack/setup-env.sh
spack env activate ufs-utils-env
export LD_LIBRARY_PATH=/spack/var/spack/environments/ufs-utils-env/.spack-env/view/lib

RESTARTS=$1
DATE=$2
SOURCE_RESOLUTION=$3
TARGET_RESOLUTION=$4
TRACERS=$5
VCOORD_FILE=$6
REFERENCE_DATA=$7
DESTINATION=$8
MPI_TASKS=$9

VCOORD_FILENAME="$(basename $VCOORD_FILE)"
LOCAL_RESTARTS=/input_data
LOCAL_VCOORD_FILE=/input_data/$VCOORD_FILENAME
LOCAL_SOURCE_REFERENCE_DATA=/reference_data/$SOURCE_RESOLUTION
LOCAL_TARGET_REFERENCE_DATA=/reference_data/$TARGET_RESOLUTION

mkdir -p $LOCAL_RESTARTS
gsutil cp $RESTARTS/* $LOCAL_RESTARTS/
gsutil cp $VCOORD_FILE $LOCAL_VCOORD_FILE

mkdir -p $LOCAL_SOURCE_REFERENCE_DATA
mkdir -p $LOCAL_TARGET_REFERENCE_DATA
gsutil cp -r $REFERENCE_DATA/$SOURCE_RESOLUTION/* $LOCAL_SOURCE_REFERENCE_DATA/
gsutil cp -r $REFERENCE_DATA/$TARGET_RESOLUTION/* $LOCAL_TARGET_REFERENCE_DATA/

#----------------------------------------------------------------------------
# Set up environment paths.
#
# EXECufs - Location of ufs_utils executable directory.
# FIXfv3  - Location of target grid orography and 'grid' files.
# FIXsfc  - Location of target grid surface climatological files.
#----------------------------------------------------------------------------

EXECufs=/UFS_UTILS/install/bin
FIXfv3=$LOCAL_TARGET_REFERENCE_DATA
FIXsfc=$FIXfv3/fix_sfc

#----------------------------------------------------------------------------
# DATE - YYMMDDHH of your run.
#----------------------------------------------------------------------------

year=$(echo $DATE|cut -c1-4)
month=$(echo $DATE|cut -c5-6)
day=$(echo $DATE|cut -c7-8)
hour=$(echo $DATE|cut -c9-10)

#----------------------------------------------------------------------------
# Variables for stand-alone regional grids.
#
# REGIONAL - Set to 1 to create remove halo and create lateral boundary
#            file.  Set to 2 for lateral boundary file only.  Set to
#            0 for non-regional grids.
# HALO_BNDY  - Number of rows/cols for lateral boundaries.
# HALO_BLEND - Number of rows/cols for blending zone.
#----------------------------------------------------------------------------

REGIONAL=${REGIONAL:-0}
HALO_BNDY=${HALO_BNDY:-0}
HALO_BLEND=${HALO_BLEND:-0}

#----------------------------------------------------------------------------
# INPUT_TYPE - Input data type:
#        'restart' for tiled fv3 warm restart files.
#        'history' for tiled fv3 history files.
#        'gaussian_nemsio' for fv3 gaussian nemsio files.
#        'gaussian_netcdf' for fv3 gaussian netcdf files.
#        'grib2' for fv3gfs grib2 files.
#        'gfs_gaussain_nemsio' for spectral gfs nemsio files.
#        'gfs_sigio' for spectral gfs sigio/sfcio files.
#
# MOSAIC_FILE_INPUT_GRID - Path/Name of mosaic file for input grid.  Only
#                          used for 'history' and 'restart' INPUT_TYPE.
#                          Set to NULL otherwise.
#
# OROG_DIR_INPUT_GRID - Location of orography and grid files for input grid.
#                       Only used for 'history' and 'restart' INPUT_TYPE.
#                       Set to NULL otherwise.
#
# OROG_FILES_INPUT_GRID - List of orography files for input grid.  Only
#                         used for 'history' and 'restart' INPUT_TYPE.
#                         Set to NULL otherwise.
#----------------------------------------------------------------------------

INPUT_TYPE="restart"
MOSAIC_FILE_INPUT_GRID=$LOCAL_SOURCE_REFERENCE_DATA/${SOURCE_RESOLUTION}_mosaic.nc
OROG_DIR_INPUT_GRID=$LOCAL_SOURCE_REFERENCE_DATA
OROG_FILES_INPUT_GRID=''${SOURCE_RESOLUTION}'_oro_data.tile1.nc","'${SOURCE_RESOLUTION}'_oro_data.tile2.nc"'
OROG_FILES_INPUT_GRID=${OROG_FILES_INPUT_GRID}',"'${SOURCE_RESOLUTION}'_oro_data.tile3.nc","'${SOURCE_RESOLUTION}'_oro_data.tile4.nc"'
OROG_FILES_INPUT_GRID=${OROG_FILES_INPUT_GRID}',"'${SOURCE_RESOLUTION}'_oro_data.tile5.nc","'${SOURCE_RESOLUTION}'_oro_data.tile6.nc'

#----------------------------------------------------------------------------
# COMIN       - Location of input data
# CONVERT_ATM - Convert atmospheric fields when true
# CONVERT_SFC - Convert surface fields when true
# CONVERT_NST - Convert nst fields when true
#----------------------------------------------------------------------------

COMIN=$LOCAL_RESTARTS
CONVERT_ATM=${CONVERT_ATM:-.true.}
CONVERT_SFC=${CONVERT_SFC:-.true.}
CONVERT_NST=${CONVERT_NST:-.false.}

#----------------------------------------------------------------------------
# ATM_FILES_INPUT - Input atmospheric data file(s).  Not used for 'restart'
#                   or 'grib2' INPUT_TYPE.
#
# ATM_CORE_FILES - Input atmospheric core files.  Used for 'restart'
#                  INPUT_TYPE only.  The first six entries are the tiled
#                  files.  The seventh is the file containing the
#                  vertical coord definition.
#
# ATM_TRACER_FILES_INPUT - Input atmospheric tracer files for each tile.
#                          Used for 'restart' INPUT_TYPE only.
#
# SFC_FILES_INPUT - Input surface data file(s).  Not used for 'grib2'
#                   INPUT_TYPE.
#
# NST_FILES_INPUT - Input nst data file.  'gfs_gaussian_nemsio' INPUT_TYPE only.
#
# GRIB2_FILE_INPUT - Input gfs grib2 data file.  Only used for 'grib2'
#                    INPUT_TYPE.
#
# TRACERS_INPUT - List of input atmospheric tracer records to be processed.
#                 Not used for 'grib2' INPUT_TYPE.
#----------------------------------------------------------------------------

ATM_FILES_INPUT=NULL
ATM_CORE_FILES_INPUT='fv_core.res.tile1.nc","fv_core.res.tile2.nc","fv_core.res.tile3.nc","fv_core.res.tile4.nc","fv_core.res.tile5.nc","fv_core.res.tile6.nc","fv_core.res.nc'
ATM_TRACER_FILES_INPUT='fv_tracer.res.tile1.nc","fv_tracer.res.tile2.nc","fv_tracer.res.tile3.nc","fv_tracer.res.tile4.nc","fv_tracer.res.tile5.nc","fv_tracer.res.tile6.nc'
SFC_FILES_INPUT='sfc_data.tile1.nc","sfc_data.tile2.nc","sfc_data.tile3.nc","sfc_data.tile4.nc","sfc_data.tile5.nc","sfc_data.tile6.nc'
NST_FILES_INPUT=NULL
GRIB2_FILE_INPUT=NULL
TRACERS_INPUT="${TRACERS}"

#----------------------------------------------------------------------------
#
# VARMAP_FILE - Variable mapping table.  Only used for 'grib2' INPUT_TYPE.
#
# TRACERS_TARGET - List of target tracer records. Must corresponde with
#                  with TRACERS_INPUT.  Not used for 'grib2' INPUT_TYPE.
#
# VCOORD_FILE - File containing vertical coordinate definition for target
#               grid.
#
# MOSAIC FILE_TARGET_GRID - Mosaic file for target grid (include path).
#                           The associated 'grid' files assumed to be in
#                           FIXfv3.
#
# OROG_FILES_TARGET_GRID - Orography file(s) for target grid.  Assumed to
#                          be located in FIXfv3.
#----------------------------------------------------------------------------

VARMAP_FILE=NULL
TRACERS_TARGET="${TRACERS}"
MOSAIC_FILE_TARGET_GRID=${FIXfv3}/${TARGET_RESOLUTION}_mosaic.nc
OROG_FILES_TARGET_GRID=''${TARGET_RESOLUTION}'_oro_data.tile1.nc","'${TARGET_RESOLUTION}'_oro_data.tile2.nc"'
OROG_FILES_TARGET_GRID=${OROG_FILES_TARGET_GRID}',"'${TARGET_RESOLUTION}'_oro_data.tile3.nc","'${TARGET_RESOLUTION}'_oro_data.tile4.nc"'
OROG_FILES_TARGET_GRID=${OROG_FILES_TARGET_GRID}',"'${TARGET_RESOLUTION}'_oro_data.tile5.nc","'${TARGET_RESOLUTION}'_oro_data.tile6.nc'

#----------------------------------------------------------------------------
# OMP_NUM_THREADS - threads most useful for 'gfs_sigio' INPUT_TYPE.
#----------------------------------------------------------------------------

export OMP_NUM_THREADS=${OMP_NUM_THREADS_CH:-1}

WORK_DIRECTORY=$PWD/chgres
mkdir -p $WORK_DIRECTORY
cd $WORK_DIRECTORY || exit 99

cat << EOF > ./fort.41
 &config
  mosaic_file_target_grid="${MOSAIC_FILE_TARGET_GRID}"
  fix_dir_target_grid="${FIXsfc}"
  orog_dir_target_grid="${FIXfv3}"
  orog_files_target_grid="${OROG_FILES_TARGET_GRID}"
  vcoord_file_target_grid="${LOCAL_VCOORD_FILE}"
  mosaic_file_input_grid="${MOSAIC_FILE_INPUT_GRID}"
  orog_dir_input_grid="${OROG_DIR_INPUT_GRID}"
  orog_files_input_grid="${OROG_FILES_INPUT_GRID}"
  data_dir_input_grid="${COMIN}"
  atm_files_input_grid="${ATM_FILES_INPUT}"
  atm_core_files_input_grid="${ATM_CORE_FILES_INPUT}"
  atm_tracer_files_input_grid="${ATM_TRACER_FILES_INPUT}"
  sfc_files_input_grid="${SFC_FILES_INPUT}"
  nst_files_input_grid="${NST_FILES_INPUT}"
  grib2_file_input_grid="${GRIB2_FILE_INPUT}"
  varmap_file="${VARMAP_FILE}"
  cycle_year=$year
  cycle_mon=$month
  cycle_day=$day
  cycle_hour=$hour
  convert_atm=$CONVERT_ATM
  convert_sfc=$CONVERT_SFC
  convert_nst=$CONVERT_NST
  input_type="${INPUT_TYPE}"
  tracers=$TRACERS_TARGET
  tracers_input=$TRACERS_INPUT
  regional=$REGIONAL
  halo_bndy=$HALO_BNDY
  halo_blend=$HALO_BLEND
 /
EOF

mpiexec -n $MPI_TASKS ${EXECufs}/chgres_cube 1>&1 2>&2

iret=$?
if [ $iret -ne 0 ]; then
  echo "FATAL ERROR RUNNING CHGRES"
  exit $iret
fi

gsutil cp gfs_ctrl.nc $DESTINATION/
for tile in {1..6}
do
  gsutil cp out.atm.tile${tile}.nc $DESTINATION/gfs_data.tile${tile}.nc
  gsutil cp out.sfc.tile${tile}.nc $DESTINATION/sfc_data.tile${tile}.nc
done

exit
