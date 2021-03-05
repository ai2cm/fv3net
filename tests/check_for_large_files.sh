#!/bin/bash

whiteList="\
external/vcm/tests/test_data/test_data.tar \
workflows/fine_res_budget/tests/gfsphysics.json \
workflows/fine_res_budget/tests/diag.json \
workflows/fine_res_budget/tests/restart.json \
workflows/prognostic_c48_run/tests/input_data/inputs_4x4.nc
"

thresholdKb=250

ret=0

while IFS= read -r -d $'\0' file
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

done < <(git ls-files -z)

exit $ret
