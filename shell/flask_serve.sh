#!/usr/bin/env bash
#
#  DEBUGGING ONLY
#
#  Start a flask server in the foreground that will auto-update its sources
#
if [ "$MONGO_URL" = "" ]
then
 echo
 echo "Illegal Configuraton"
 echo "MONGO_URL neets to be set to point to a running mongodb"
 exit -1
fi

cd ${ELLIE_HOME}/src && export FLASK_DEBUG=1 && export FLASK_APP=inference_webapp.py && python -m flask run