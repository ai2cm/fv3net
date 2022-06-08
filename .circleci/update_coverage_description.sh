#!/usr/bin/env bash

set -e -x

workflow=$CIRCLE_WORKFLOW_ID
pr_number=${CIRCLE_PULL_REQUEST##*/}
body=$(curl https://api.github.com/repos/ai2cm/fv3net/pulls/${pr_number} | jq '.body' | tail -c +2 | head -c -2 | sed -e 's/\\r\\n/\r\n/g' | sed -e 's/\\t/\t/g')
if [[ -f htmlcov/index.html ]]; then
    coverage_label=$(grep -oP '(?<=<span class="pc_cov">).+(?=<\/span>)' htmlcov/index.html)
else
    coverage_label="link"
fi

if [[ "$body" == *"${workflow}:"* ]]; then
    body=$(echo "$body" | sed -e "s/${workflow}: [.*](https:\/\/.*\/index.html)/${workflow}: [${coverage_label}](https:\/\/output.circle-artifacts.com\/output\/job\/${CIRCLE_WORKFLOW_JOB_ID}\/artifacts\/0\/tmp\/coverage\/htmlcov-$CIRCLE_WORKFLOW_ID\/index.html)/g")
else
    body=$(echo "$body" | sed -e "s/Coverage reports (updated automatically):/Coverage reports (updated automatically):\n- ${workflow}: [${coverage_label}](https:\/\/output.circle-artifacts.com\/output\/job\/${CIRCLE_WORKFLOW_JOB_ID}\/artifacts\/0\/tmp\/coverage\/htmlcov-$CIRCLE_WORKFLOW_ID\/index.html)/g")
fi

gh pr edit --body "$body"
