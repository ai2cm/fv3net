#!/bin/bash

kubectl exec -c main -n kubeflow -ti $1 -- bash
