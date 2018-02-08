#! /bin/bash
i=$1
if [ "$i" -gt 0 ]; then
    range=$(while [ $i != 0 ]; do echo $(expr $i - 1); i=$(expr $i - 1); done)
else
    echo "usage: work.sh <num_workers>"
    exit -1
fi

for i in $range; do
    echo starting worker $i:
    python keras_distributed.py --job_name="worker" --task_index=$i > ./worker-$i.log 2>&1 &
    sleep 1
done
tail -f worker-0.log