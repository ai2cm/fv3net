#!/bin/sh

minikube start
./build_docker.sh
./create-secret-local.sh
