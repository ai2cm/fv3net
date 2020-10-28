#!/bin/bash

kubectl apply -f https://raw.githubusercontent.com/argoproj/argo/v2.11.6/manifests/install.yaml
kubectl create secret generic gcp-key --from-file="$GOOGLE_APPLICATION_CREDENTIALS"
kubectl apply -f cluster