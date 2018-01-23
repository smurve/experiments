#!/usr/bin/env bash

echo "================================================================"
echo
echo    "Warning: This is a DEBUG version using local directories only.."
echo
echo    "WARNING: Not using GPUs if you don't specify --num-gpus=<num>"
echo
echo "================================================================"

ROOT_DIR=/Users/wgiersche/tmp

python cifar10_main.py \
    --job-dir=${ROOT_DIR}/cifar10_new \
    --data-dir=/var/ellie/data/cifar10_tfr/ \
    --use-distortion-for-training=false \
    --train-steps=100 \
    --train-batch-size=2000 \
    --eval-batch-size=2000 \
    --data-format=channels_last

