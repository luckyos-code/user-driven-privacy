#!/bin/bash

# Script to trigger all the experiments

TS=$(date '+%Y-%m-%d_%H:%M:%S');

# Workspace must be allocated before starting the job
WORK_DIR=$PWD/results/all-$TS
mkdir -p $WORK_DIR

use_gpu=false  # Set to true if GPU is required

# Define all parameters
datasets=(
    "adult"
    "diabetes"
)

# Define percentages as an array of strings (space-separated)
percentages_list=( # O/G/M
    "0.33 0.33 0.34",
    # "1.0 0.0 0.0",
    # "0.0 1.0 0.0",
    # "0.0 0.0 1.0",
    # "0.66 0.17 0.17",
    # "0.17 0.66 0.17",
    # "0.17 0.17 0.66",
    # "0.50 0.25 0.25",
    # "0.25 0.50 0.25",
    # "0.25 0.25 0.50",
)

train_methods=(
    # "original" # always handled like no anonymization
    # "no_preprocessing"
    # "forced_generalization"
    # "specialization"
    "weighted_specialization"
    # "weighted_specialization_highest_confidence"
    #
    # "extended_weighted_specialization"
)

test_methods=(
    "original" # always handled like no anonymization
    # "no_preprocessing"
    # "forced_generalization"
    # "specialization"
    # "weighted_specialization"
    # "weighted_specialization_highest_confidence"
    #
    # "extended_weighted_specialization"
)

_filter_by_record_id=(
    false
    true
)

# Option to enable group duplicates for specific combinations
# HINT: not used anymore because only few duplicates where present
_enable_group_duplicates=(
    false
    # true
)

# GPU-specific options
gpu_flag=""
if [ "$use_gpu" = true ]; then
    gpu_flag="-n"
fi

# Loop through all cases
for enable_group_duplicates in "${_enable_group_duplicates[@]}"; do
    for filter_by_record_id in "${_filter_by_record_id[@]}"; do
        for dataset in "${datasets[@]}"; do
            for train_method in "${train_methods[@]}"; do
                for test_method in "${test_methods[@]}"; do
                    for percentages in "${percentages_list[@]}"; do
                        # Skip runs with filtering if the method does not contain "specialization"
                        if [ "$filter_by_record_id" = true ] && [[ "$train_method" != *"specialization"* && "$test_method" != *"specialization"* ]]; then
                            continue
                        fi

                        # dynamic cluster size depending on task
                        if [[ "$train_method" != *"specialization"* && "$test_method" != *"specialization"* ]]; then
                            n_workers=1
                        elif [[ "$dataset" == "diabetes" ]]; then
                            n_workers=4
                        elif [[ "$dataset" == "adult" ]]; then
                            n_workers=6
                        fi
                    
                        # Build command
                        cmd="sbatch ./job.sh -w $WORK_DIR -d $dataset -t $train_method -e $test_method -s $n_workers -p \"$percentages\""
                        
                        # group arg
                        if [ "$enable_group_duplicates" = true ]; then
                            cmd="$cmd -g"
                        fi

                        # gpu arg
                        if [ "$use_gpu" = true ]; then
                            cmd="$cmd -n"
                        fi

                        # filter arg
                        if [ "$filter_by_record_id" = true ]; then
                            cmd="$cmd -f"
                        fi
                        
                        eval $cmd
                    done
                done
            done
        done
    done
done
