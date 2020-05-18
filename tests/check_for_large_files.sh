#!/bin/bash

whiteList="external/vcm/tests/test_data/test_data.tar workflows/prognostic_run_diags/poetry.lock"

thresholdKb=250

ret=0

for file in $(git ls-files)
do
    size=$(du -k "$file" | awk -F' ' '{print $1}')

    [[ -f "$file" ]] || continue

    if [[ "$size" -ge "$thresholdKb" ]]
    then
        # check if present in whitelist
        echo "$whiteList" | grep -w "$file" > /dev/null
        inWhiteList=$?
        if [[ $inWhiteList -ne 0 ]]
        then
            echo "Large file found: \"$file\" is $size kb"
            ret=1
        fi
    fi

done

exit $ret