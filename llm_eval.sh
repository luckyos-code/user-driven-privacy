#!/bin/bash
# LLM Evaluation Helper Script
# Simplifies launching and merging partitioned evaluations
# currently only needed for employment dataset due to size

# bash llm_eval.sh launch "33-33-34,33-0-67,0-66-34,66-0-34,66-17-17" Employment 1 10
# bash llm_eval.sh launch "1-0-0" "German-Adult-Diabetes-Employment" 1 10

set -e

usage() {
    cat << EOF
LLM Evaluation Helper - Simplified workflow for partitioned evaluations

USAGE:
    $0 launch <percentage> <datasets> <partitions> [batch_size] [input_dir]
    $0 merge <percentage> <datasets> <partitions>
    $0 status
    $0 help

COMMANDS:
    launch      Submit partitioned jobs to SLURM
    merge       Merge completed partition results
    status      Check status of running jobs
    help        Show this help message

NOTES:
    - <percentage> can be a single value (e.g., "33-33-34") or comma-separated list (e.g., "33-33-34,33-0-67,0-66-34,66-0-34,66-17-17")
    - <datasets> can be a single dataset or hyphen-separated list (e.g., "Adult-Diabetes-German")
    - Multiple percentages will be processed sequentially
    - [batch_size] defaults to 1 if not provided

EXAMPLES:
    # Launch for multiple datasets with single percentage (no partitioning needed for smaller datasets)
    $0 launch 33-33-34 "Adult-Diabetes-German" 1 10 /work/ll95wyqa-user-driven

    # Launch 3 partitions for Employment with single percentage and batch size 10
    $0 launch 33-33-34 Employment 3 10 /work/ll95wyqa-user-driven

    # Launch with multiple percentages for multiple datasets
    $0 launch "33-33-34,33-0-67,0-66-34,66-0-34,66-17-17" "Adult-Diabetes-German" 1 10

    # Check job status
    $0 status

    # Merge results for single percentage
    $0 merge 33-33-34 Employment 3

    # Merge results for multiple percentages
    $0 merge "33-33-34,33-0-67,0-66-34,66-0-34,66-17-17" Employment 3

    # Launch with default input dir and batch size 1
    $0 launch 33-33-34 Employment 3
EOF
}

launch() {
    local percentage="$1"
    local datasets="$2"
    local partitions="$3"
    local batch_size="${4:-1}"
    local input_dir="${5:-/work/ll95wyqa-user-driven}"
    
    if [ -z "$percentage" ] || [ -z "$datasets" ] || [ -z "$partitions" ]; then
        echo "ERROR: Missing required arguments for launch"
        echo "Usage: $0 launch <percentage> <datasets> <partitions> [batch_size] [input_dir]"
        exit 1
    fi
    
    # Split datasets by hyphen and launch a separate job for each
    IFS='-' read -ra dataset_array <<< "$datasets"
    for dataset in "${dataset_array[@]}"; do
        echo "Launching $partitions partitions for $dataset ($percentage) with batch size $batch_size..."
        python llm_evaluation_launcher.py \
            --percentage "$percentage" \
            --datasets "$dataset" \
            --partitions "$partitions" \
            --batch-size "$batch_size" \
            --input-dir "$input_dir" \
            --results-base llm_evaluation
    done
}

merge() {
    local percentage="$1"
    local datasets="$2"
    local partitions="$3"
    
    if [ -z "$percentage" ] || [ -z "$datasets" ] || [ -z "$partitions" ]; then
        echo "ERROR: Missing required arguments for merge"
        echo "Usage: $0 merge <percentage> <datasets> <partitions>"
        exit 1
    fi
    
    echo "Merging results for $datasets ($percentage, $partitions partitions)..."
    python llm_evaluation_merger.py \
        --percentage "$percentage" \
        --datasets "$datasets" \
        --partitions "$partitions" \
        --results-base llm_evaluation
}

status() {
    echo "Current SLURM jobs:"
    squeue -u "$USER" -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
    
    echo ""
    echo "Recent job logs:"
    find llm_slurm_logs -type f -name "*.err" -mtime -1 -exec sh -c 'echo "---"; echo "File: {}"; tail -5 "{}"' \;
}

case "${1:-}" in
    launch)
        shift
        launch "$@"
        ;;
    merge)
        shift
        merge "$@"
        ;;
    status)
        status
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac
