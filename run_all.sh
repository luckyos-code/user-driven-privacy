#!/bin/bash

# Script to trigger all the experiments
# ATTENTION: will start vast amount if not selective with commenting out percentages/methods/datasets
# be selective by commenting out appropriate rows in the arrays
# also check the variable settings to fit your needs

TS=$(date '+%Y-%m-%d_%H:%M:%S');

# Workspace must be allocated before starting the job
WORK_DIR=$PWD/results/all-$TS
mkdir -p $WORK_DIR

 # Set to true if GPU is required
use_gpu=false
# Option to enable group duplicates for specification
# HINT: not used anymore because only few duplicates where present
enable_group_duplicates=false

# if enabled only save cache no training runs
cache_only=false
# filter or use fully specialized dataset
filter_by_record_id=true

# Batch submit controls: start N jobs then wait
# Set batch_size to 0 to disable batching (no waits)
# Example: batch_size=3; wait_seconds=3 -> start 3 runs, sleep 3s, continue
batch_size=0
wait_seconds=60

# Control how train/test method pairs are selected. This is an array so you can enable multiple modes
# in a single run. Comment out the one you don't need (same pattern as other arrays in this file).
# Examples:
# require_matching_methods=(false)        # only pairs including "original"
# require_matching_methods=(true)         # only identical pairs (A+A)
# require_matching_methods=(false true)   # run both modes in sequence
require_matching_methods=(
    true
    false
)

# Define all parameters
datasets=(
    "german"
    "diabetes"
    "adult"
    "employment"
)

# Define percentages as an array of strings (space-separated)
# comment out percentages you don't want to run
percentages_list=( # TODO
    # Original Generalized Missing
    # "1.0 0.0 0.0" # dont use, as this is already handled with "original-original" methods combination, so not actually needed
    #"0.66 0.17 0.17"
    #"0.66 0.00 0.34"
    #"0.33 0.33 0.34"
    #"0.33 0.00 0.67"
    #"0.00 0.66 0.34"

    # additional splits
    #"0.90 0.05 0.05" # for fast testing
    # "0.0 1.0 0.0" # extreme generalization case (comparable to forced generalization)
)

# comment out train and test methods you don't want to run (depending on require_matching_methods this will create jobs based on cross-product pairs)
# llm_prediction is directly evaluated using the llm module and extracted from llm_evaluation folder
train_methods=(
    ## just data
    "original" # always handled like no anonymization
    "no_preprocessing"

    ## methods
    "forced_generalization"
    "weighted_specialization"

    ## baselines
    "baseline_imputation"
    
    ## LLM-based methods
    "llm_imputation" # Loads pre-imputed data from llm_evaluation/<percentages>/ folder
    
    ## not used anymore
    # "specialization" # running with only specialization can create all caches, so perfect for a cache_only run but method not used in practice
    # "weighted_specialization_highest_confidence"
    # "extended_weighted_specialization"
)

test_methods=(
    ## just data
    "original" # always handled like no anonymization
    "no_preprocessing"

    ## methods
    "forced_generalization"
    "weighted_specialization"

    ## baselines
    "baseline_imputation"
    
    ## LLM-based methods
    "llm_imputation" # Loads pre-imputed data from llm_evaluation/<percentages>/ folder
    
    ## not used anymore
    # "specialization" # running with only specialization can create all caches, so perfect for a cache_only run but method not used in practice
    # "weighted_specialization_highest_confidence"
    # "extended_weighted_specialization"
)

# GPU-specific options
gpu_flag=""
if [ "$use_gpu" = true ]; then
    gpu_flag="-n"
fi

# Counter for launched runs (used for batching/waiting)
runs_started=0

# Loop through all cases
for dataset in "${datasets[@]}"; do
    # Track whether the original+original pair has already been executed for THIS dataset.
    # We initialize this once per dataset so it persists across different require_matching modes
    original_original_done=false  # set to true to skip original+original entirely; set to false to run it once

    for require_matching in "${require_matching_methods[@]}"; do
        # require_matching is 'true' or 'false' for this outer iteration
        for train_method in "${train_methods[@]}"; do
            for test_method in "${test_methods[@]}"; do
            # Decide whether to run this train/test pair based on $require_matching_methods:
            # - If $require_matching_methods=true: only identical train/test method pairs are allowed.
            # - If $require_matching_methods=false: only pairs where at least one side is the literal
            #   method name "original" are allowed (e.g. original+X or X+original). Non-original pairs are skipped.
            if [ "$require_matching" = true ]; then
                if [ "$train_method" != "$test_method" ]; then
                    # skip non-identical combinations
                    continue
                fi
            else
                # require at least one side to be the literal 'original'
                if [ "$train_method" != "original" ] && [ "$test_method" != "original" ]; then
                    continue
                fi
            fi

            # Special-case: when both train and test are the literal 'original', run that pair only once
            # regardless of how many percentage sets are defined (we'll use only the first percentages entry).
            both_original=false
            if [ "$train_method" = "original" ] && [ "$test_method" = "original" ]; then
                both_original=true
                # if we've already run original+original in a previous mode, skip now
                if [ "$original_original_done" = true ]; then
                    continue
                fi
            fi

            p_index=0
            for percentages in "${percentages_list[@]}"; do
                p_index=$((p_index+1))
                # If both train and test are original, only run once (first percentages entry)
                if [ "$both_original" = true ] && [ $p_index -gt 1 ]; then
                    continue
                fi

                # Determine filter_by_record_id based on methods
                if [[ "$train_method" == *"specialization"* || "$test_method" == *"specialization"* ]]; then
                    filter_by_record_id=$filter_by_record_id
                else
                    filter_by_record_id=false
                fi

                # dynamic cluster size depending on task
                if [[ "$train_method" != *"specialization"* && "$test_method" != *"specialization"* ]]; then
                    n_workers=1
                elif [[ "$dataset" == "adult" ]]; then
                    n_workers=2
                elif [[ "$dataset" == "diabetes" ]]; then
                    n_workers=1
                elif [[ "$dataset" == "german" ]]; then
                    n_workers=1
                elif [[ "$dataset" == "employment" ]]; then
                    n_workers=2 # may need adjustment based on run setup
                fi

                # Build command
                cmd="sbatch ./run.job -w $WORK_DIR -d $dataset -t $train_method -e $test_method -s $n_workers -p \"$percentages\""

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

                # cache_only arg
                if [ "$cache_only" = true ]; then
                    cmd="$cmd -c"
                fi

                eval $cmd

                # increment launched runs counter and apply batching/wait if enabled
                runs_started=$((runs_started+1))
                if [ "$batch_size" -gt 0 ] && [ $((runs_started % batch_size)) -eq 0 ]; then
                    echo "Launched $runs_started jobs, sleeping for $wait_seconds seconds..."
                    sleep "$wait_seconds"
                fi

                # Mark original+original as done so subsequent modes won't run it again
                if [ "$both_original" = true ]; then
                    original_original_done=true
                fi
            done
            done
        done
    done
done
