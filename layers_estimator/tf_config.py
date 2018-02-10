#!/usr/bin/env python

import json
import sys

node_type = sys.argv[1]
index = sys.argv[2]

ps_ips = ['localhost:2222']
worker_ips = ['localhost:2223']

cluster = {'ps': ps_ips,
           'worker': worker_ips}

print(json.dumps(
    {'cluster': cluster,
     'environment': 'cloud'
     }))
