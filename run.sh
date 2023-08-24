#!/bin/bash

# Define the hyperparameters and their possible values
learning_rates=(0.01 0.05 0.1)
batch_sizes=(16 32 64)

# Loop over all combinations of hyperparameters
for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        # Run the training code with the current combination of hyperparameters
        python train.py --lr=$lr --batch=$bs -d vec -e "dti_$lr_$bs" --gpu=1 --num_epochs 200
    done
done




#python train.py -d vec -e dti_SGD_0.1lr --gpu=1 --hop=2 --batch=128 -b=4 --num_epochs 200 --lr 0.01 --optimizer SGD