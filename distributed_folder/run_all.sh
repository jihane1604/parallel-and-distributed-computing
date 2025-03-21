#!/bin/bash
cd ~
scp -r jiji student@10.102.0.167:~/
scp -r jiji student@10.102.0.217:~/
cd jiji
mpirun -hostfile machines.txt -np 18 python distributed_ver.py