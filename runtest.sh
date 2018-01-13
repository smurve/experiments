#!/bin/bash
echo starting container
sh 'docker run -d -p5000 -e MONGO_URL=mongodb://scylla:30017 smurve/capsnet-fashion:test'
echo done. Getting container id.
c=$(docker ps | grep "smurve/capsnet-fashion:test" | awk '{print $1}')
echo done: $c. Getting port
port=$(docker inspect --format "{{.NetworkSettings.Ports}}" $c | awk '{print $2}' | cut -d "}" -f1)
echo done: $port. Curling
res=$(curl localhost:$port | grep 'Author')
[ "$res" != "" ] || echo Failed. Result: $res 
echo done: $res. Stopping container
docker stop $c
echo done. $removing container
docker rm $c
[ "$res" != "" ] 
