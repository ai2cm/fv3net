#!/bin/bash
#
# Implementation Notes:
#
# This program is implemented using an entity-based paradigm. This is suitable
# for distributed workflows that modify remote resources.
#
# The functions in an entity-based API only allow for operations create, read,
# update, and delete (CRUD). These operations are applied to the entities to
# define the behavior of the system. Using only CRUD operations moves the state
# of the run execution into these entities, and makes the logic less reliant on
# local state, and therefore easier to translate to a distributed context.
#
# For a segmented run "Runs" and "Segments" are the main entities. Runs have
# many segments and some global data stores.


function usage {
    echo "A segmented run is currently a location on Google Cloud Storage."
    echo "This script can create segmented runs and appending segments to them."
    echo "It also provides low level commands used for debugging."
    echo ""
    echo "Usage:"
    echo " runfv3.sh create <outputUrl> <fv3configFile> <chunks> <runFile>" > /dev/stderr 
    echo " runfv3.sh append <outputUrl>" > /dev/stderr 
    echo " runfv3.sh run-native <fv3config> <rundir> <runfile>" > /dev/stderr
    echo ""
    echo "Segmented Run Commands:"
    echo "  create      Initialize a segmented run in GCS"
    echo "  append      Add a segmented to a segmented run"
    echo ""
    echo "Low-level Commands:"
    echo "  run-native  Setup a run-directory and run the model. Used for "
    echo "              testing/debugging."
    echo ""
}

function createRun {
    url="$1"
    fv3config="$2"
    chunks="$3"
    runfile="$4"


    if [[ $# != 4 ]]
    then 
        usage
        exit 1
    fi

    if [[ "$(cat "$chunks")" == "" ]] 
    then
        echo "$chunks is empty" > /dev/stderr
        exit 1
    fi

    verifyUrlEmpty "$url"

    echo "Setting up iterations"
    gsutil cp "$fv3config" "$url/fv3config.yml"
    gsutil cp "$chunks" "$url/chunks.yaml"
    gsutil cp "$runfile" "$url/runfile.py"
}

function readRunConfig {
    url="$1"
    gsutil cat "$runURL/fv3config.yml"
}

function readRunChunks {
    url="$1"
    gsutil cat "$runURL/chunks.yaml"
}

function readRunLastSegment {
    url="$1"
    segments=($(gsutil ls "$runURL/artifacts" 2> /dev/null || echo "" ))
    len="${#segments[@]}"
    if [[ $len -gt 0 ]]
    then
        printf "%s" "${segments[-1]}"
    fi
}

function appendSegment {
    set -e
    runURL="$1"

    workingDir="$(mktemp -d)"
    echo "Iteration run=$runURL working_directory=$workingDir" > /dev/stderr
    readRunConfig "$runURL" > "$workingDir/fv3config.yml"
    readRunChunks "$runURL" > "$workingDir/chunks.yaml"
    lastSegment=$(readRunLastSegment "$runURL")

    if [[ -n "$lastSegment" ]]
    then
        echo "Continuing from segment $lastSegment"
        # lastSegment includes trailing slash
        enable_restart "$workingDir/fv3config.yml" "${lastSegment}RESTART"
        update_config_for_nudging "$workingDir/fv3config.yml"
    else
        echo "First segment in $runURL"
    fi

    rundir="$workingDir/rundir"
    postProcessedOut="$workingDir/post_processed"
    gsutil cp "$runURL/runfile.py" "$workingDir/runfile.py"

    set +e
    runSegment "$workingDir/fv3config.yml" "$rundir" "$workingDir/runfile.py"
    fv3ExitCode=$?
    set -e

    post_process_run --chunks "$workingDir/chunks.yaml" "$rundir" "$postProcessedOut"
    append_run "$postProcessedOut" "$runURL"

    echo "Cleaning up working directory"
    set +e
    rm -r "$workingDir"

    if [[ "$fv3ExitCode" -ne 0 ]]
    then
        exit 1
    fi
}


function runSegment {
    fv3config="$1"
    rundir="$2"
    runfile="$3"
    write_run_directory "$fv3config" "$rundir"
    mkdir -p "$rundir"
    cp "$runfile" "$rundir/runfile.py"
    (
        cd "$rundir"
        NUM_PROC=$(yq '.namelist.fv_core_nml.layout | .[0] *.[1] * 6' "$fv3config")
        mpirun -n "$NUM_PROC" python3 "$runfile" |& tee -a "$rundir/logs.txt"
    )
}

function verifyUrlEmpty {
    url="$1"
    set +e
    gsutil ls "$url" > /dev/null 2> /dev/null
    ret=$?
    set -e
    if [[ "$ret" -eq 0 ]]; then
        echo "The given output url (below) contains an object. Delete everything under output url and resubmit."
        echo "{{inputs.parameters.url}}"
        exit 1
    fi
}


set -e
set -o pipefail


command="$1"

if [[ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]]
then
    gcloud auth activate-service-account --key-file "$GOOGLE_APPLICATION_CREDENTIALS" 2> /dev/null
fi

case "$command" in 
    "create")
        shift
        createRun $@
        ;;
    "append")
        shift
        appendSegment "$1"
        ;;
    "run-native")
        shift
        runSegment $@
        ;;
    *)
        usage
        exit 1
        ;;
esac

