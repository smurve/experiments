#!/usr/bin/env bash
#
# start the notebook from the venv virtual environment
#
python ./venv/bin/jupyter-notebook --notebook-dir=${ELLIE_HOME}/src --ip 127.0.0.1 --port 8888 --port-retries=1
