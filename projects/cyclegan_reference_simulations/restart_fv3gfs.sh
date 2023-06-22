#!/bin/tcsh
#SBATCH --qos=urgent
#SBATCH --account=gfdl_w
#SBATCH --clusters=c5

# The fundamental structure of this script is based on scripts used to
# run SHiELD on NOAA HPC systems.  Examples of such scripts can be
# found in the SHiELD_build repository, which is released under the
# LGPL license: https://github.com/NOAA-GFDL/SHiELD_build.  For
# reference a copy of this license is included here.

# Something unique about this script is that it fv3config to set up run
# directories, which greatly simplifies the work done by typical SHiELD
# runscripts.

# ### GNU LESSER GENERAL PUBLIC LICENSE

# Version 3, 29 June 2007

# Copyright (C) 2007 Free Software Foundation, Inc.
# <https://fsf.org/>

# Everyone is permitted to copy and distribute verbatim copies of this
# license document, but changing it is not allowed.

# This version of the GNU Lesser General Public License incorporates the
# terms and conditions of version 3 of the GNU General Public License,
# supplemented by the additional permissions listed below.

# #### 0. Additional Definitions.

# As used herein, "this License" refers to version 3 of the GNU Lesser
# General Public License, and the "GNU GPL" refers to version 3 of the
# GNU General Public License.

# "The Library" refers to a covered work governed by this License, other
# than an Application or a Combined Work as defined below.

# An "Application" is any work that makes use of an interface provided
# by the Library, but which is not otherwise based on the Library.
# Defining a subclass of a class defined by the Library is deemed a mode
# of using an interface provided by the Library.

# A "Combined Work" is a work produced by combining or linking an
# Application with the Library. The particular version of the Library
# with which the Combined Work was made is also called the "Linked
# Version".

# The "Minimal Corresponding Source" for a Combined Work means the
# Corresponding Source for the Combined Work, excluding any source code
# for portions of the Combined Work that, considered in isolation, are
# based on the Application, and not on the Linked Version.

# The "Corresponding Application Code" for a Combined Work means the
# object code and/or source code for the Application, including any data
# and utility programs needed for reproducing the Combined Work from the
# Application, but excluding the System Libraries of the Combined Work.

# #### 1. Exception to Section 3 of the GNU GPL.

# You may convey a covered work under sections 3 and 4 of this License
# without being bound by section 3 of the GNU GPL.

# #### 2. Conveying Modified Versions.

# If you modify a copy of the Library, and, in your modifications, a
# facility refers to a function or data to be supplied by an Application
# that uses the facility (other than as an argument passed when the
# facility is invoked), then you may convey a copy of the modified
# version:

# -   a) under this License, provided that you make a good faith effort
#     to ensure that, in the event an Application does not supply the
#     function or data, the facility still operates, and performs
#     whatever part of its purpose remains meaningful, or
# -   b) under the GNU GPL, with none of the additional permissions of
#     this License applicable to that copy.

# #### 3. Object Code Incorporating Material from Library Header Files.

# The object code form of an Application may incorporate material from a
# header file that is part of the Library. You may convey such object
# code under terms of your choice, provided that, if the incorporated
# material is not limited to numerical parameters, data structure
# layouts and accessors, or small macros, inline functions and templates
# (ten or fewer lines in length), you do both of the following:

# -   a) Give prominent notice with each copy of the object code that
#     the Library is used in it and that the Library and its use are
#     covered by this License.
# -   b) Accompany the object code with a copy of the GNU GPL and this
#     license document.

# #### 4. Combined Works.

# You may convey a Combined Work under terms of your choice that, taken
# together, effectively do not restrict modification of the portions of
# the Library contained in the Combined Work and reverse engineering for
# debugging such modifications, if you also do each of the following:

# -   a) Give prominent notice with each copy of the Combined Work that
#     the Library is used in it and that the Library and its use are
#     covered by this License.
# -   b) Accompany the Combined Work with a copy of the GNU GPL and this
#     license document.
# -   c) For a Combined Work that displays copyright notices during
#     execution, include the copyright notice for the Library among
#     these notices, as well as a reference directing the user to the
#     copies of the GNU GPL and this license document.
# -   d) Do one of the following:
#     -   0) Convey the Minimal Corresponding Source under the terms of
#         this License, and the Corresponding Application Code in a form
#         suitable for, and under terms that permit, the user to
#         recombine or relink the Application with a modified version of
#         the Linked Version to produce a modified Combined Work, in the
#         manner specified by section 6 of the GNU GPL for conveying
#         Corresponding Source.
#     -   1) Use a suitable shared library mechanism for linking with
#         the Library. A suitable mechanism is one that (a) uses at run
#         time a copy of the Library already present on the user's
#         computer system, and (b) will operate properly with a modified
#         version of the Library that is interface-compatible with the
#         Linked Version.
# -   e) Provide Installation Information, but only if you would
#     otherwise be required to provide such information under section 6
#     of the GNU GPL, and only to the extent that such information is
#     necessary to install and execute a modified version of the
#     Combined Work produced by recombining or relinking the Application
#     with a modified version of the Linked Version. (If you use option
#     4d0, the Installation Information must accompany the Minimal
#     Corresponding Source and Corresponding Application Code. If you
#     use option 4d1, you must provide the Installation Information in
#     the manner specified by section 6 of the GNU GPL for conveying
#     Corresponding Source.)

# #### 5. Combined Libraries.

# You may place library facilities that are a work based on the Library
# side by side in a single library together with other library
# facilities that are not Applications and are not covered by this
# License, and convey such a combined library under terms of your
# choice, if you do both of the following:

# -   a) Accompany the combined library with a copy of the same work
#     based on the Library, uncombined with any other library
#     facilities, conveyed under the terms of this License.
# -   b) Give prominent notice with the combined library that part of it
#     is a work based on the Library, and explaining where to find the
#     accompanying uncombined form of the same work.

# #### 6. Revised Versions of the GNU Lesser General Public License.

# The Free Software Foundation may publish revised and/or new versions
# of the GNU Lesser General Public License from time to time. Such new
# versions will be similar in spirit to the present version, but may
# differ in detail to address new problems or concerns.

# Each version is given a distinguishing version number. If the Library
# as you received it specifies that a certain numbered version of the
# GNU Lesser General Public License "or any later version" applies to
# it, you have the option of following the terms and conditions either
# of that published version or of any later version published by the
# Free Software Foundation. If the Library as you received it does not
# specify a version number of the GNU Lesser General Public License, you
# may choose any version of the GNU Lesser General Public License ever
# published by the Free Software Foundation.

# If the Library as you received it specifies that a proxy can decide
# whether future versions of the GNU Lesser General Public License shall
# apply, that proxy's public statement of acceptance of any version is
# permanent authorization for you to choose that version for the
# Library.

set echo
set SCRATCH_ROOT = $1
set NAME = $2
set REFERENCE_CONFIG = $3
set SEGMENTS = $4
set EXECUTABLE = $5
set CONDA_ENV = $6
set SCRIPTS_DIR = $7
set ENVIRONMENT_ROOT = $8
set PLATFORM = $9
set REFERENCE_RESTART_DIR = ${10}

##### Set environment for running dynamically linked FV3GFS #####
set MODULES_FILE = ${ENVIRONMENT_ROOT}/fv3net/external/fv3gfs-fortran/FV3/conf/modules.fv3.${PLATFORM}
source $MODULES_FILE
module list

set ESMF_LIB = ${ENVIRONMENT_ROOT}/install/esmf/lib
set FMS_LIB = ${ENVIRONMENT_ROOT}/fv3net/external/fv3gfs-fortran/FMS/libFMS/.libs
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}
#################################################################

set WORKDIR = ${SCRATCH_ROOT}/${NAME}
set RUNDIR = ${WORKDIR}/rundir
set SEGMENT_CONFIG = ${RUNDIR}/fv3config.yml

# Set environment variables following SHiELD runscripts
setenv MPICH_ENV_DISPLAY
setenv MPICH_MPIIO_CB_ALIGN 2
setenv MALLOC_MMAP_MAX_ 0
setenv MALLOC_TRIM_THRESHOLD_ 536870912
setenv NC_BLKSZ 1M

# necessary for OpenMP when using Intel
setenv KMP_STACKSIZE 256m
setenv SLURM_CPU_BIND verbose

module load python/3.9
source /sw/gaea-c5/python/3.9/anaconda-base/etc/profile.d/conda.csh
echo "Activating conda env $CONDA_ENV"
conda activate $CONDA_ENV

echo "Active conda environment:"
conda list

set NPES = `python $SCRIPTS_DIR/get_npes.py $REFERENCE_CONFIG`
set run_cmd = "srun --export=ALL --ntasks=$NPES --cpus-per-task=1 ./$EXECUTABLE:t"

set SCRIPT_AREA = $PWD
if ( "$SLURM_JOB_NAME" == "sh" ) then
  set SCRIPT = "${SCRIPT_AREA}/$0"
else
  set SCRIPT = "${SCRIPT_AREA}/$SLURM_JOB_NAME"
endif

mkdir -p $WORKDIR/restart
set RST_COUNT = $WORKDIR/restart/rst.count

if ( -f ${RST_COUNT} ) then
  source ${RST_COUNT}
  if ( x"$num" == "x" || ${num} < 1 ) then
    set CONTINUATION_RUN = "F"
  else
    set CONTINUATION_RUN = "T"
  endif
else
  set num = 0
  set CONTINUATION_RUN = "F"
endif

set START_DATE = `python $SCRIPTS_DIR/get_date_restart.py $REFERENCE_CONFIG $REFERENCE_RESTART_DIR $num`
set END_DATE = `python $SCRIPTS_DIR/get_date_restart.py --end $REFERENCE_CONFIG $REFERENCE_RESTART_DIR $num`
echo $START_DATE
echo $END_DATE

if (${CONTINUATION_RUN} == "T") then
  rm -rf $RUNDIR
  set INITIAL_CONDITIONS = $WORKDIR/restart/$START_DATE
else
  set INITIAL_CONDITIONS = $REFERENCE_RESTART_DIR
endif
mkdir $RUNDIR
cp $REFERENCE_CONFIG $SEGMENT_CONFIG
enable_restart $SEGMENT_CONFIG $INITIAL_CONDITIONS
write_run_directory $SEGMENT_CONFIG $RUNDIR

cd $RUNDIR
ls INPUT/
ls RESTART/
cp $EXECUTABLE .

# Run the executable
${run_cmd} | tee fms.out || exit
@ num ++
echo "set num = ${num}" >! ${RST_COUNT}

# Move files out of run directory in advance of next segment
mkdir -p $WORKDIR/restart/$END_DATE
mv RESTART/* $WORKDIR/restart/$END_DATE

mkdir -p $WORKDIR/history/$START_DATE
mv *.nc $WORKDIR/history/$START_DATE

mkdir -p $WORKDIR/ascii/$START_DATE
mv *.out *.results *.yml $WORKDIR/ascii/$START_DATE

# Resubmit to run another segment if needed
if ($num < $SEGMENTS) then
  echo "resubmitting... "
  if ( "$SLURM_JOB_NAME" == "sh" ) then
    cd $SCRIPT_AREA
    ./$SCRIPT:t
  else
    cd $SCRIPT_AREA
    sbatch --nodes=$SLURM_NNODES --export=SBATCH_TIMELIMIT,SLURM_NNODES $SCRIPT:t \
	$SCRATCH_ROOT \
	$NAME \
	$REFERENCE_CONFIG \
	$SEGMENTS \
	$EXECUTABLE \
	$CONDA_ENV \
	$SCRIPTS_DIR \
	$ENVIRONMENT_ROOT \
	$PLATFORM \
        $REFERENCE_RESTART_DIR
    sleep 60
  endif
endif
