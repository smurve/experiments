#!/usr/bin/env bash

#
#  I let it in, but it isn't working on my cluster.
#  Actually, communicaton begins, training commences, but I get a CUDNN_STATUS_INTERNAL ERROR on the master.
#  Maybe, I'll look into that issue at some point in time, later.
#  For the time being, 2x GPU on a single node is apparently working well.
#

node_type=$1 # either of ps, master, worker
task_index=$2
steps=$3
shift
shift
shift

export TF_CONFIG=$(python tf_config.py ${node_type} ${task_index})

python cifar10_main.py \
    --job-dir=/var/ellie/models/cifar10_new \
    --data-dir=/var/ellie/data/cifar10_tfr/ \
    --use-distortion-for-training=false \
    --num-gpus=2 \
    --sync \
    --train-steps=${steps} \
    "$@"
