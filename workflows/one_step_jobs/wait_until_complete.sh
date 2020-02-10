#!/bin/bash
# Job watcher that uses job labels to monitor completion
# Initial code provided by Stephen Dasbit

exp_label=$1

while true; do
  active_jobs="$(kubectl get jobs --selector "experiment_group=$exp_label" --output=json | jq --raw-output '[.items[] | select (.status.active == 1)] | .[].metadata.name')" || exit 1
  if [ -z "$active_jobs" ]; then
    printf "All kubernetes jobs finished!\n"
    break
  fi
  printf "active jobs at %s:\n%s\n" "$(date +%s)" "$active_jobs"
  sleep 30
done;

FAILED_JOBS="$(kubectl get jobs --selector "experiment_group=$exp_label" --output=json | jq --raw-output '[.items[] | select (.status.failed == 1)] | .[].metadata.name')" || exit 1
printf "failed jobs from iteration %s:\n%s\n" "$exp_label" "$FAILED_JOBS"

exit 0