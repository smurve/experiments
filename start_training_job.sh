#!/usr/bin/env bash
status=$(kubectl get pods --show-all --no-headers --selector=job-name=capsnet-fashion-trainer | awk '{print $3}')

cont="ABORT"
[ "${status}" = "Completed" ] &&  echo "Previous job completed. Deleting it." && kubectl delete -f training_job.yml
[ "${status}" = "" ] && echo "No previous job. Good. Will delete and re-create"
if [ "${status}" != "Completed" ] && [ "${status}" != "" ]
then
 echo "Status ${status}. Not Good. Aborting."
 exit -1
fi

kubectl create -f training_job.yml