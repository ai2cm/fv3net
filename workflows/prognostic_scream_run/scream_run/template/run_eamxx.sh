#!/bin/bash -fe

# EAMxx template run script
create_newcase=${create_newcase:-true}
case_setup=${case_setup:-true}
case_build=${case_build:-true}
case_submit=${case_submit:-true}
upload_to_cloud=${upload_to_cloud:-true}
upload_to_cloud_path=${upload_to_cloud_path:-"gs://vcm-ml-scratch/scream"}
number_of_processers=${number_of_processers:-16}
output_yaml=${output_yaml:-"gs://vcm-scream/config/default.yaml"}
CASE_ROOT=${CASE_ROOT:-./}
CASE_NAME=${CASE_NAME:-F2010-SCREAMv1.ne30pg2_ne30pg2}
COMPSET=${COMPSET:-F2010-SCREAMv1}
RESOLUTION=${RESOLUTION:-ne30pg2_ne30pg2}
ATM_NCPL=${ATM_NCPL:-48}
STOP_OPTION=${STOP_OPTION:-ndays}
STOP_N=${STOP_N:-1}
RES_OPTION=${RES_OPTION:-ndays}
RES_N=${RES_N:-1}
HIST_OPTION=${HIST_OPTION:-ndays}
HIST_N=${HIST_N:-1}
RUN_STARTDATE=${RUN_STARTDATE:-2010-01-01}
MODEL_START_TYPE=${MODEL_START_TYPE:-initial}
OLD_EXECUTABLE=${OLD_EXECUTABLE:-""}

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi
  shift
done

main() {

# --- Configuration flags ----
# Code and compilation
readonly MACHINE="docker-scream"
readonly BRANCH="master"         # master, branch name, or github hash
readonly CHERRY=( )
readonly COMPILER="gnu"
readonly DEBUG_COMPILE=false

# Output stream YAML files: comma sepated list of files, or set to "" for default
readonly OUTPUT_YAML_FILES=${output_yaml}

# Additional options for 'branch' and 'hybrid'
readonly GET_REFCASE=false
readonly RUN_REFDIR=""
readonly RUN_REFCASE=""
readonly RUN_REFDATE=""   # same as MODEL_START_DATE for 'branch', can be different for 'hybrid'

# Machine specific settings
readonly CODE_ROOT="/src/E3SM"
readonly CASE_ROOT="${CASE_ROOT}/${CASE_NAME}"
mkdir -p /storage/timings

# Sub-directories
readonly CASE_BUILD_DIR=${CASE_ROOT}/build
readonly CASE_ARCHIVE_DIR=${CASE_ROOT}/archive

readonly layout="${number_of_processers}x1"
readonly run=${layout}
readonly CASE_SCRIPTS_DIR=${CASE_ROOT}/tests/${run}/case_scripts
readonly CASE_RUN_DIR=${CASE_ROOT}/tests/${run}/run
readonly PELAYOUT=${layout}
readonly WALLTIME="01:00:00"
readonly RESUBMIT=0
readonly DO_SHORT_TERM_ARCHIVING=false

# Make directories created by this script world-readable
umask 022

# Create case
create_newcase

# Setup
case_setup

# Build
case_build

# Configure runtime options
runtime_options

# Submit
case_submit

# Upload to google cloud
upload_to_google_cloud

# All done
echo $'\n----- All done -----\n'

}

# =======================
# Custom user_nl settings
# =======================

user_nl() {

    echo "+++ Configuring SCREAM for 128 vertical levels +++"
    ./xmlchange SCREAM_CMAKE_OPTIONS="SCREAM_NP 4 SCREAM_NUM_VERTICAL_LEV 128 SCREAM_NUM_TRACERS 10"

}

######################################################
### Most users won't need to change anything below ###
######################################################

#-----------------------------------------------------
create_newcase() {

    if [ "${create_newcase,,}" != "true" ]; then
        echo $'\n----- Skipping create_newcase -----\n'
        return
    fi

    echo $'\n----- Starting create_newcase -----\n'
    echo ${CASE_ROOT}
    # Base arguments
    args=" --case ${CASE_NAME} \
        --output-root ${CASE_ROOT} \
        --script-root ${CASE_SCRIPTS_DIR} \
        --handle-preexisting-dirs u \
        --compset ${COMPSET} \
        --res ${RESOLUTION} \
        --machine ${MACHINE} \
        --compiler ${COMPILER} \
        --walltime ${WALLTIME} \
        --pecount ${PELAYOUT}"

    # Oprional arguments
    if [ ! -z "${PROJECT}" ]; then
      args="${args} --project ${PROJECT}"
    fi
    if [ ! -z "${QUEUE}" ]; then
      args="${args} --queue ${QUEUE}"
    fi

    ${CODE_ROOT}/cime/scripts/create_newcase ${args}

    if [ $? != 0 ]; then
      echo $'\nNote: if create_newcase failed because sub-directory already exists:'
      echo $'  * delete old case_script sub-directory'
      echo $'  * or set newcase=false\n'
      exit 35
    fi

}

#-----------------------------------------------------
case_setup() {

    if [ "${case_setup,,}" != "true" ]; then
        echo $'\n----- Skipping case_setup -----\n'
        return
    fi

    echo $'\n----- Starting case_setup -----\n'
    pushd ${CASE_SCRIPTS_DIR}

    # Setup some CIME directories
    ./xmlchange EXEROOT=${CASE_BUILD_DIR}
    ./xmlchange RUNDIR=${CASE_RUN_DIR}

    # Short term archiving
    ./xmlchange DOUT_S=${DO_SHORT_TERM_ARCHIVING^^}
    ./xmlchange DOUT_S_ROOT=${CASE_ARCHIVE_DIR}
    ./xmlchange PIO_NETCDF_FORMAT="64bit_data"
    # Extracts input_data_dir in case it is needed for user edits to the namelist later
    local input_data_dir=`./xmlquery DIN_LOC_ROOT --value`

    # Custom user_nl
    user_nl

    # Finally, run CIME case.setup
    ./case.setup --reset

    popd
}

#-----------------------------------------------------
case_build() {

    pushd ${CASE_SCRIPTS_DIR}

    # case_build = false
    if [ "${case_build,,}" != "true" ]; then

        echo $'\n----- case_build -----\n'

        if [ "${OLD_EXECUTABLE}" == "" ]; then
            # Ues previously built executable, make sure it exists
            if [ -x ${CASE_BUILD_DIR}/e3sm.exe ]; then
                echo 'Skipping build because $case_build = '${case_build}
            else
                echo 'ERROR: $case_build = '${case_build}' but no executable exists for this case.'
                exit 297
            fi
        else
            # If absolute pathname exists and is executable, reuse pre-exiting executable
            if [ -x ${OLD_EXECUTABLE} ]; then
                echo 'Using $OLD_EXECUTABLE = '${OLD_EXECUTABLE}
                cp -fp ${OLD_EXECUTABLE} ${CASE_BUILD_DIR}/
            else
                echo 'ERROR: $OLD_EXECUTABLE = '$OLD_EXECUTABLE' does not exist or is not an executable file.'
                exit 297
            fi
        fi
        echo 'WARNING: Setting BUILD_COMPLETE = TRUE.  This is a little risky, but trusting the user.'
        ./xmlchange BUILD_COMPLETE=TRUE

    else

        echo $'\n----- Starting case_build -----\n'

        # Turn on debug compilation option if requested
        if [ "${DEBUG_COMPILE^^}" == "TRUE" ]; then
            ./xmlchange DEBUG=${DEBUG_COMPILE^^}
        fi

        # Run CIME case.build
        ./case.build

        # Some user_nl settings won't be updated to *_in files under the run directory
        # Call preview_namelists to make sure *_in and user_nl files are consistent.
        ./preview_namelists

    fi

    popd
}

#-----------------------------------------------------
runtime_options() {

    echo $'\n----- Starting runtime_options -----\n'
    pushd ${CASE_SCRIPTS_DIR}

    # Set simulation start date
    if [ ! -z "${START_DATE}" ]; then
        ./xmlchange RUN_STARTDATE=${START_DATE}
    fi

    # Segment length
    ./xmlchange STOP_OPTION=${STOP_OPTION,,},STOP_N=${STOP_N}

    # Restart frequency
    ./xmlchange REST_OPTION=${REST_OPTION,,},REST_N=${REST_N}
    ./atmchange Scorpio::model_restart::output_control::frequency_units=${REST_OPTION} \
                Scorpio::model_restart::output_control::Frequency=${REST_N}

    # Coupler history
    ./xmlchange HIST_OPTION=${HIST_OPTION,,},HIST_N=${HIST_N}

    # Coupler budgets (always on)
    ./xmlchange BUDGETS=TRUE

    ./xmlchange ATM_NCPL=${ATM_NCPL}

    # Set resubmissions
    if (( RESUBMIT > 0 )); then
        ./xmlchange RESUBMIT=${RESUBMIT}
    fi

    # Run type
    # Start from default of user-specified initial conditions
    if [ "${MODEL_START_TYPE,,}" == "initial" ]; then
        ./xmlchange RUN_TYPE="startup"
        ./xmlchange CONTINUE_RUN="FALSE"

    # Continue existing run
    elif [ "${MODEL_START_TYPE,,}" == "continue" ]; then
        ./xmlchange CONTINUE_RUN="TRUE"

    elif [ "${MODEL_START_TYPE,,}" == "branch" ] || [ "${MODEL_START_TYPE,,}" == "hybrid" ]; then
        ./xmlchange RUN_TYPE=${MODEL_START_TYPE,,}
        ./xmlchange GET_REFCASE=${GET_REFCASE}
	    ./xmlchange RUN_REFDIR=${RUN_REFDIR}
        ./xmlchange RUN_REFCASE=${RUN_REFCASE}
        ./xmlchange RUN_REFDATE=${RUN_REFDATE}
        echo 'Warning: $MODEL_START_TYPE = '${MODEL_START_TYPE}
        echo '$RUN_REFDIR = '${RUN_REFDIR}
        echo '$RUN_REFCASE = '${RUN_REFCASE}
        echo '$RUN_REFDATE = '${START_DATE}

    else
        echo 'ERROR: $MODEL_START_TYPE = '${MODEL_START_TYPE}' is unrecognized. Exiting.'
        exit 380
    fi

    # Change output stream yaml files if requested
    if [ ! -z "${OUTPUT_YAML_FILES}" ]; then
        ./atmchange output_yaml_files="${OUTPUT_YAML_FILES}"
    fi

    popd
}

#-----------------------------------------------------
case_submit() {

    if [ "${case_submit,,}" != "true" ]; then
        echo $'\n----- Skipping case_submit -----\n'
        return
    fi

    echo $'\n----- Starting case_submit -----\n'
    pushd ${CASE_SCRIPTS_DIR}

    # Run CIME case.submit
    ./case.submit

    popd
}

#-----------------------------------------------------
upload_to_google_cloud(){
    if [ "${upload_to_cloud,,}" != "true" ]; then
        echo $'\n----- Skipping upload_to_google_cloud -----\n'
        return
    fi

    echo $'\n----- Uploading to Google Cloud -----\n'
    gsutil -m cp -r ${CASE_RUN_DIR} ${upload_to_cloud_path}/${CASE_NAME}
}

#-----------------------------------------------------
# Silent versions of popd and pushd
pushd() {
    command pushd "$@" > /dev/null
}
popd() {
    command popd "$@" > /dev/null
}

# Now, actually run the script
#-----------------------------------------------------
main
