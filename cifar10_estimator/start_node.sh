#!/usr/bin/env bash

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
