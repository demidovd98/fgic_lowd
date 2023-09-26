#!/bin/bash

mkdir -p ./checkpoint
mkdir -p ./logs
mkdir -p ./models/pre_trained/
mkdir -p ./_extra/U2Net/saved_models/
mkdir -p ./output
mkdir -p ./test

module load cuda-10.2
nvcc --version

source /apps/local/anaconda2023/conda_init.sh
conda activate /l/users/cv-805/envs/ffvt_u2n_p36