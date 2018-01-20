#!/bin/bash
echo starting container
docker run -d -p5000 -e MONGO_URL=mongodb://scylla:30017 -v /var/ellie:/var/ellie smurve/capsnet-fashion:latest
echo done. 
echo Getting container id.
c=$(docker ps | grep "smurve/capsnet-fashion:latest" | awk '{print $1}')
echo done. Container id is ${c}.
echo Getting port
port=$(docker inspect --format "{{.NetworkSettings.Ports}}" ${c} | awk '{print $2}' | cut -d "}" -f1)
echo done: Port number is ${port}.
echo curling after 2 seconds
sleep 2
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
