#!/usr/bin/env bash

set -e

report_type=$1

if [[ -f htmlcov/index.html ]]; then
    pr_number=${CIRCLE_PULL_REQUEST##*/}
    body=$(gh pr view | sed -e '1,13d')
    coverage_label=$(grep -oP '(?<=<span class="pc_cov">).+(?=<\/span>)' htmlcov/index.html)
    report_url="https:\/\/output.circle-artifacts.com\/output\/job\/${CIRCLE_WORKFLOW_JOB_ID}\/artifacts\/0\/tmp\/coverage\/htmlcov-${report_type}\/index.html"
    link_header="Coverage reports (updated automatically):"
    if [[ "$body" == *"${report_type}:"* ]]; then
        body=$(echo "$body" | sed -e "s/${report_type}: [.*](https:\/\/.*\/index.html)/${report_type}: [${coverage_label}](${report_url})/g")
    elif [[ "$body" == *"${link_header}"* ]]; then
        body=$(echo "$body" | sed -e "s/${link_header}/${link_header}\r\n- ${report_type}: [${coverage_label}](${report_url})/g")
    else
        body="${body}\r\n\r\n${link_header}\r\n- ${report_type}: [${coverage_label}](${report_url})"
    fi
    echo "$body"
    gh pr edit --body "$body"
else
    echo "no coverage report found"
fi
