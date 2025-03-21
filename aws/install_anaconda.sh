#!/bin/bash

cd ~
sudo apt update
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
echo "installed anaconda, exiting now"
exit