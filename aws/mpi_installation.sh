#!/bin/bash

conda activate parallel
echo "Installing mpi4py..."
sudo apt-get update 
sudo apt-get install mpich
conda install -c conda-forge mpi4py mpich