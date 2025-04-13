#!/bin/bash
cd ~
scp -r jiji student@10.102.0.169:~/
cd jiji
mpirun -hostfile machines.txt -np 12 python distributed.py