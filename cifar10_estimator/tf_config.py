#!/usr/bin/env python

import json
import sys

node_type = sys.argv[1]
index = sys.argv[2]

ps_ips = ['scylla:2222', 'charybdis:2222']
master_ips = ['scylla:2223']
worker_ips = ['charybdis:2223']

cluster = {'master': master_ips,
           'ps': ps_ips,
           'worker': worker_ips}

print(json.dumps(
    {'cluster': cluster,
     'task': {'type': node_type, 'index': 0},
     'model_dir': '/var/ellie/models/cifar10_new',
     'environment': 'local'
     }))
