#!/bin/bash

# rm /tmp/hancache

for i in 256 128 64 32; do
    MADRONA_MWGPU_KERNEL_CACHE=/tmp/hancache python hanabi_train_timed.py --num-envs $i --num-steps 64 --num-updates 2048 --learning-rate 1e-3 --update-epochs 10 --num-minibatches 1 --madrona True --ent-coef 0.015 --anneal-lr False --hanabi-type full
done
