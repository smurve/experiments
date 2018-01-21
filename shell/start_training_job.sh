#!/usr/bin/env bash
status=$(kubectl get pods --show-all --no-headers --selector=job-name=capsnet-fashion-trainer | head -n 1 | awk '{print $3}')

kubectl delete -f training_job.yml || echo "INFO: Couldn't delete. That's OK. Probably was deleted before"
kubectl create -f training_job.yml