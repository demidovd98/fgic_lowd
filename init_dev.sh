#!/bin/bash

mkdir -p ./checkpoint
mkdir -p ./logs
mkdir -p ./models/pre_trained/
mkdir -p ./_extra/U2Net/saved_models/
mkdir -p ./output
mkdir -p ./test

module load cuda-10.2
nvcc --version