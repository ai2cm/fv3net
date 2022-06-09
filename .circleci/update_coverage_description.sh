#!/usr/bin/env bash

set -e -x

report_type=$1

if [[ -f htmlcov/index.html ]]; then
    workflow=$CIRCLE_WORKFLOW_ID
    pr_number=${CIRCLE_PULL_REQUEST##*/}
    body=$(curl https://api.github.com/repos/ai2cm/fv3net/pulls/${pr_number} | jq '.body' | tail -c +2 | head -c -2 | sed -e 's/\\r\\n/\r\n/g' | sed -e 's/\\t/\t/g')
    coverage_label=$(grep -oP '(?<=<span class="pc_cov">).+(?=<\/span>)' htmlcov/index.html)

    if [[ "$body" == *"${workflow}:"* ]]; then
        body=$(echo "$body" | sed -e "s/${report_type}: [.*](https:\/\/.*\/index.html)/${workflow}: [${coverage_label}](https:\/\/output.circle-artifacts.com\/output\/job\/${CIRCLE_WORKFLOW_JOB_ID}\/artifacts\/0\/tmp\/coverage\/htmlcov-${report_type}\/index.html)/g")
    else
        body=$(echo "$body" | sed -e "s/Coverage reports (updated automatically):/Coverage reports (updated automatically):\n- ${workflow}: [${coverage_label}](https:\/\/output.circle-artifacts.com\/output\/job\/${CIRCLE_WORKFLOW_JOB_ID}\/artifacts\/0\/tmp\/coverage\/htmlcov-${report_type}\/index.html)/g")
    fi

    gh pr edit --body "$body"
else
    echo "no coverage report found"
fi
