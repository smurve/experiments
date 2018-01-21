#!/bin/bash
echo starting container
docker run -d -p5000 -e MONGO_URL=mongodb://scylla:30017 -v /var/ellie:/var/ellie smurve/ellie_inference_webapp:latest
echo Done.
echo Will continue after 10 seconds
sleep 10
echo Getting container id.
c=$(docker ps | grep "smurve/ellie_inference_webapp:latest" | awk '{print $1}')
echo Done. Container id is ${c}.
echo Getting port
port=$(docker inspect --format "{{.NetworkSettings.Ports}}" ${c} | awk '{print $2}' | cut -d "}" -f1)
echo Done: Port number is ${port}.
echo Curling the health end point
res=$(curl localhost:${port}/health | grep 'Status: Healthy')
echo "result was '$res'" 
[ "$res" != "" ] || echo Failure!!
[ "$res" != "" ] && echo Success!!

echo done testing. Stopping container
docker stop ${c}
echo done. Removing container
docker rm ${c}
echo done. Reporting back
[ "$res" != "" ] 
