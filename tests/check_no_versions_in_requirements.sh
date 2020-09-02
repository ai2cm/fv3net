#!/bin/bash

for file in $(git ls-files)
do
    if [[ "$(basename $file)" == "requirements.txt" ]]; then
        if grep '=' $file > /dev/null ; then
            echo "Version found in $file. All version informations should be listed in constraints.txt instead."
            exit 1
        fi
    fi
done
