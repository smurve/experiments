#!/usr/bin/env bash
rm -rf /tmp/mnist_model && \
    python mnist.py \
        --train_epochs=1 \
        --data_dir=/var/ellie/data/mnist_fashion