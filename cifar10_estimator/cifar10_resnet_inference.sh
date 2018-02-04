#!/usr/bin/env bash

echo "================================================================"
echo
echo    "WARNING: Not using GPUs if you don't specify --num-gpus=<num>"
echo
echo "================================================================"

python cifar10_resnet_inference.py \
    --job-dir=/var/ellie/models/cifar10_new \
    --data-dir=/var/ellie/data/cifar10_tfr/ \
    --num-gpus=0 \
    "$@"

