#!/usr/bin/env bash
status=$(kubectl get pods --show-all --no-headers --selector=job-name=capsnet-fashion-trainer | head -n 1 | awk '{print $3}')

cont="ABORT"
[ "${status}" = "Completed" ] &&  echo "Previous job completed. Deleting it." && kubectl delete -f training_job.yml
[ "${status}" = "Error" ] &&  echo "Previous job failed. Deleting it." && kubectl delete -f training_job.yml
[ "${status}" = "" ] && echo "No previous job. Good. Will create one."
if [ "${status}" != "Completed" ] && [ "${status}" != "" ] && [ "${status}" != "Error" ]
then
 echo "Status ${status}. Not Good. Aborting."
 exit -1
fi

kubectl create -f training_job.yml