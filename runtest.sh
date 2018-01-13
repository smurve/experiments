#!/bin/bash
sh 'docker run -d -p5000 -e MONGO_URL=mongodb://scylla:30017 smurve/capsnet-fashion:test'
c=$(docker ps | grep "smurve/capsnet-fashion:test" | awk '{print $1}')
port=$(docker inspect --format "{{.NetworkSettings.Ports}}" $c | awk '{print $2}' | cut -d "}" -f1)
res=$(curl localhost:$port | grep 'Author')
docker stop $c
docker rm $c
[ "$res" != "" ] 
