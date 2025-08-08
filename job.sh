#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1  
#SBATCH --mem=8G                   # was 400G before
#SBATCH --job-name=user_privacy
#SBATCH --partition=paul
#SBATCH --output=slurm_logs/%j/stdout.out
#SBATCH --error=slurm_logs/%j/stderr.err

set -x  # to print all the commands to stderr

# get needed modules and packages
module load CUDA/11.8.0
module load dask/2022.1.0-foss-2021b # Python 3.9.6
module load matplotlib/3.4.3-foss-2021b

pip install xgboost==2.0.3 h5py==3.11.0 seaborn==0.13.2 scikit-learn==1.2.2 torch==2.6.0

# Default parameters as fallback - see run_all for real parameters
DEFAULT_WORK_DIR=$PWD/results/default
DATASET="adult"
TRAIN_METHOD="original"
TEST_METHOD="original"
GROUP_DUPLICATES=false
N_WORKERS=1
USE_GPU=false
PERCENTAGES="0.33 0.33 0.34"  # space-separated string for percentages

# common paths
CODE_DIR=$PWD/src
DATA_DIR=$PWD/data

# Parse command line arguments
while getopts w:d:t:e:s:gnfp: flag
do
    case "${flag}" in
        w) WORK_DIR=${OPTARG};;
        d) DATASET=${OPTARG};;
        t) TRAIN_METHOD=${OPTARG};;
        e) TEST_METHOD=${OPTARG};;
        s) N_WORKERS=${OPTARG};;
        g) GROUP_DUPLICATES=true;;
        n) USE_GPU=true;;
        f) FILTER_BY_RECORD_ID=true;;
        p) PERCENTAGES="${OPTARG}";;
    esac
done

# check if workdir exists
if [ -z "$WORK_DIR" ] || ! [ -d "$WORK_DIR" ]; then
    echo "working directory ${WORK_DIR} does not exist. Creating and using: ${DEFAULT_WORK_DIR}"
    mkdir -p $DEFAULT_WORK_DIR
    WORK_DIR=$DEFAULT_WORK_DIR
fi

TS=$(date '+%Y-%m-%d_%H:%M:%S');
RUN_DIR="${WORK_DIR}/${DATASET}"

# Create arguments for group_duplicates if enabled
GROUP_ARGS=""
if [ "$GROUP_DUPLICATES" = true ]; then
    GROUP_ARGS="--group_duplicates"
fi

# Create arguments for use_gpu if enabled
GPU_ARGS=""
if [ "$USE_GPU" = true ]; then
    GPU_ARGS="--use_gpu"
fi

FILTER_ARGS=""
if [ "$FILTER_BY_RECORD_ID" = true ]; then
    FILTER_ARGS="--filter_by_record_id"
fi

# Track start time
start_time=$(date +%s)
start_time_human=$(date '+%Y-%m-%d %H:%M:%S')

# create run command
PYTHON_COMMAND="python3 run.py --save_dir $RUN_DIR --data_dir $DATA_DIR --dataset $DATASET --train_method $TRAIN_METHOD --test_method $TEST_METHOD --n_workers $N_WORKERS $GROUP_ARGS $GPU_ARGS $FILTER_ARGS --percentages \"$PERCENTAGES\""

# make the run
echo "START at $start_time_human"

mkdir -p $RUN_DIR
# echo $SLURM_JOB_ID > "${RUN_DIR}/slurm-job-id.txt" # added the results csv for each run instead

echo "Command: ${PYTHON_COMMAND}"

export PYTHONUNBUFFERED=1 # to get all print output in log
cd $PWD && $PYTHON_COMMAND

# Track end time
end_time=$(date +%s)
end_time_human=$(date '+%Y-%m-%d %H:%M:%S')

# Calculate runtime
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "FINISHED at $end_time_human"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"


# Add memory and general computation statistics output
echo -e "=== MEMORY USAGE SUMMARY ==="

# Get the MaxRSS value and convert to human-readable format
maxrss_kb=$(sacct -j $SLURM_JOB_ID -o MaxRSS -n | head -1 | tr -d ' ')

# Check if we got a value
if [[ -n "$maxrss_kb" && "$maxrss_kb" != "0" ]]; then
    # Remove the 'K' suffix if present
    maxrss_kb=${maxrss_kb%K}
    
    # Convert to MB and GB
    maxrss_mb=$(echo "scale=2; $maxrss_kb/1024" | bc)
    maxrss_gb=$(echo "scale=2; $maxrss_kb/1024/1024" | bc)
    
    echo "Maximum memory used: $maxrss_kb KB = $maxrss_mb MB = $maxrss_gb GB"
else
    echo "Memory usage data not available"
fi

# Also show full job statistics
echo -e "\n=== DETAILED JOB STATISTICS ==="
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,AveCPU,Elapsed,MaxDiskRead,MaxDiskWrite
