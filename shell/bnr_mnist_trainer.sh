#!/usr/bin/env bash
#
# Build and run the mnist trainer container
#
docker build -t smurve/capsnet-fashion-mnist-trainer:latest -f Dockerfile-mnist-trainer .
docker run -v/Volumes/Ellie/:/var/ellie/ -e MONGO_URL=${MONGO_URL} smurve/capsnet-fashion-mnist-trainer:latest