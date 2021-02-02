#!/bin/bash
# inputs needed
# chunks.yaml

function usage {
    echo "run-fv3.sh outputUrl fv3configFile nSegments runFile chunks" > /dev/stderr 
}

function runModel {
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

function updateConfigInplace {
    fv3config="$1"
    restart="$2"
    enable_restart "$fv3config" "$restart"
    update_config_for_nudging "$fv3config"
}


set -e
set -o pipefail


if [[ $# != 5 ]]
then 
    usage
    exit 1
fi

if [[ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]]
then
    gcloud auth activate-service-account --key-file "$GOOGLE_APPLICATION_CREDENTIALS"
fi

outputUrl="$1"
fv3configFile="$2"
nSegments="$3"
runFile="$4"
chunks="$5"

if [[ "$(cat $chunks)" == "" ]] 
then
    echo "$chunks is empty" > /dev/stderr
    exit 1
fi

verifyUrlEmpty "$outputUrl"

echo "Setting up iterations"
workingDir="$(mktemp -d)"
mkdir -p "$workingDir/1"
cp "$fv3configFile" "$workingDir/1/fv3config.yml"

for ((iter=1; iter <= nSegments; ++iter))
do
    rundir="$workingDir/$iter/rundir"
    postProcessedOut="$workingDir/$iter/post_processed"
    echo "Iteration $iter of $nSegments rundir=$rundir temporary_output=$postProcessedOut" > /dev/stderr

    runModel "$workingDir/$iter/fv3config.yml" "$rundir" "$runFile"
    post_process_run --chunks "$chunks" "$rundir" "$postProcessedOut"
    append_run "$postProcessedOut" "$outputUrl"
    rm -r "$workingDir/$iter/post_processed"

    # Preparing next iteration
    nextIterDir="$workingDir/$((iter + 1))/"
    mkdir -p "$nextIterDir"
    cp "$rundir/fv3config.yml" "$nextIterDir/fv3config.yml"
    updateConfigInplace "$nextIterDir/fv3config.yml" "$workingDir/$iter/rundir/RESTART"
done

