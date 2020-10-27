#!/bin/bash

kubectl apply -f https://raw.githubusercontent.com/argoproj/argo/stable/manifests/install.yaml
kubectl apply -f cluster
kubectl create secret generic gcp-key --from-file="$GOOGLE_APPLICATION_CREDENTIALS"
