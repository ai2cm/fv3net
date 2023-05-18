set -e

scream_run prepare-config /tmp/scream_run_integration/test.yaml /tmp/test.yaml --precompiled_case=True
if grep -q "create_newcase: False" "/tmp/test.yaml"; then
    echo "prepare_scream_config added create_newcase: False"
else
    echo "precompiled_case is True, but scream_run prepare-config did not add create_newcase: False"
    exit 1;
fi
scream_run write-rundir /tmp/test.yaml /tmp/rundir
export TEST_SCREAM_RUN_CONFIG=/tmp/test.yaml
pytest /tmp/scream_run_integration/test_scream_runtime.py