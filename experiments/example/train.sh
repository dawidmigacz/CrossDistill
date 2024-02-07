#!/usr/bin/env bash

CONFIG=$1

module load CUDA/12.0.0
python ../../tools/switch_classifier.py