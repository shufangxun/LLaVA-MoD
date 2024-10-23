#!/usr/bin/env python
# coding=utf-8
import os
pid = list(set(os.popen('fuser -v /dev/nvidia*').read().split()))
kill_cmd = 'kill -9 ' + ' '.join(pid)
print(kill_cmd)
os.popen(kill_cmd)