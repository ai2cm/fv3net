set -e

prepare_scream_config /tmp/scream_run_integration/test.yaml /tmp/test.yaml --precompiled_case=True
if grep -q "create_newcase: False" "/tmp/test.yaml"; then
    echo "prepare_scream_config added create_newcase: False"
else
    echo "precompiled_case is True, but prepare_scream_config did not add create_newcase: False"
    exit 1;
fi
write_scream_run_directory /tmp/test.yaml /tmp/rundir
pytest /tmp/scream_run_integration/test_scream_runtime.py