# LLM Evaluation Partitioning Guide

This guide explains how to split large LLM evaluations into multiple parallel jobs and merge the results.

## Overview

For large datasets (e.g., German-Employment with 300k+ rows), running the full evaluation in one job can take extremely long. The partitioning system allows you to:

1. Split the dataset into N partitions
2. Run each partition as a separate SLURM job (parallel processing)
3. Merge all partition results into final combined results

## Components

### 1. `llm_evaluation.py` (Modified)
Main evaluation script now supports `--partition` parameter:
- Format: `--partition N/TOTAL` (e.g., `--partition 1/4` for first quarter)
- When specified, processes only that partition of the dataset
- Results are saved with partition suffix: `*_part1of4.csv`

### 2. `llm_evaluation_launcher.py` (New)
Launcher script that submits multiple SLURM jobs:
- Automatically divides dataset into partitions
- Submits one job per partition
- Tracks job IDs

### 3. `llm_evaluation_merger.py` (New)
Merger script that combines partition results:
- Concatenates all partition CSVs
- Regenerates complete evaluation summaries
- Produces final merged log with all statistics

### 4. `llm_evaluation.job` (Modified)
SLURM job file now supports `PARTITION` environment variable

## Usage

### Quick Start: Partitioned Evaluation

```bash
# Step 1: Launch partitioned jobs (e.g., 4 partitions for German-Employment)
python llm_evaluation_launcher.py \
    --percentage 33-33-34 \
    --datasets German-Employment \
    --partitions 4 \
    --input-dir /work/ll95wyqa-user-driven \
    --results-base llm_evaluation

# Step 2: Wait for all jobs to complete (check with squeue)
squeue -u $USER

# Step 3: Merge results when all jobs are done
python llm_evaluation_merger.py \
    --percentage 33-33-34 \
    --datasets German-Employment \
    --partitions 4 \
    --results-base llm_evaluation
```

### Detailed Workflow

#### 1. Launch Partitioned Jobs

```bash
python llm_evaluation_launcher.py \
    --percentage 33-33-34 \
    --datasets German-Employment \
    --partitions 4 \
    --input-dir /work/ll95wyqa-user-driven \
    --results-base llm_evaluation
```

**Output:**
```
================================================================================
LLM EVALUATION LAUNCHER
================================================================================
Configuration:
  Percentage: 33-33-34
  Datasets: German-Employment
  Partitions: 4
  Input directory: /work/ll95wyqa-user-driven
  Results base: llm_evaluation
  Job file: llm_evaluation.job
  Dry run: False
================================================================================

Submitting partition 1/4...
  ✓ Job 12345 submitted for partition 1/4
Submitting partition 2/4...
  ✓ Job 12346 submitted for partition 2/4
Submitting partition 3/4...
  ✓ Job 12347 submitted for partition 3/4
Submitting partition 4/4...
  ✓ Job 12348 submitted for partition 4/4

================================================================================
SUBMISSION COMPLETE
✓ Submitted 4 jobs: 12345, 12346, 12347, 12348

To merge results after completion, run:
  python llm_evaluation_merger.py --percentage 33-33-34 --datasets German-Employment --partitions 4
================================================================================
```

#### 2. Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job logs
tail -f llm_slurm_logs/12345/stdout.out
```

#### 3. Merge Results

Once all jobs complete successfully:

```bash
python llm_evaluation_merger.py \
    --percentage 33-33-34 \
    --datasets German-Employment \
    --partitions 4 \
    --results-base llm_evaluation
```

**Output:**
```
================================================================================
LLM EVALUATION MERGER
================================================================================
Configuration:
  Percentage: 33-33-34
  Datasets: German-Employment
  Partitions: 4
  Results directory: llm_evaluation/33-33-34_results
================================================================================

Merging imputation results for german-employment (train)...
  ✓ Loaded german-employment_train_imputation_results_part1of4.csv: 25000 records
  ✓ Loaded german-employment_train_imputation_results_part2of4.csv: 25000 records
  ✓ Loaded german-employment_train_imputation_results_part3of4.csv: 25000 records
  ✓ Loaded german-employment_train_imputation_results_part4of4.csv: 25000 records
  ✓ Merged 100000 total records from 4 partitions
  ✓ Saved to llm_evaluation/33-33-34_results/german-employment_train_imputation_results.csv

================================================================================
GERMAN-EMPLOYMENT IMPUTATION (train) EVALUATION RESULTS (MERGED)
================================================================================
Total predictions: 100000
Correct predictions: 45232
Accuracy: 45.23%
...
```

## File Naming Conventions

### Partition Files (Intermediate)
- `{dataset}_{part}_imputation_results_part{N}of{TOTAL}.csv`
- `{dataset}_{part}_prediction_results_part{N}of{TOTAL}.csv`
- `{dataset}_{part}_imputed_dataset_part{N}of{TOTAL}.csv`

Example: `german-employment_train_imputation_results_part1of4.csv`

### Merged Files (Final)
- `{dataset}_{part}_imputation_results.csv`
- `{dataset}_{part}_prediction_results.csv`
- `{dataset}_{part}_imputed_dataset.csv`
- `{dataset}_evaluation_merged.log`

Example: `german-employment_train_imputation_results.csv`

## Advanced Options

### Dry Run (Test Before Submitting)

```bash
python llm_evaluation_launcher.py \
    --percentage 33-33-34 \
    --datasets German-Employment \
    --partitions 4 \
    --dry-run
```

This shows what commands would be run without actually submitting jobs.

### Custom Job File

```bash
python llm_evaluation_launcher.py \
    --percentage 33-33-34 \
    --datasets German-Employment \
    --partitions 4 \
    --job-file my_custom_job.sh
```

### Multiple Datasets

```bash
# Submit partitioned jobs for multiple datasets
python llm_evaluation_launcher.py \
    --percentage 33-33-34 \
    --datasets Adult-Diabetes-German-Employment \
    --partitions 4

# Merge each dataset separately
python llm_evaluation_merger.py \
    --percentage 33-33-34 \
    --datasets Adult \
    --partitions 4

python llm_evaluation_merger.py \
    --percentage 33-33-34 \
    --datasets Diabetes \
    --partitions 4

python llm_evaluation_merger.py \
    --percentage 33-33-34 \
    --datasets German-Employment \
    --partitions 4
```

## Performance Considerations

### Choosing Number of Partitions

| Dataset Rows | Recommended Partitions | Reason |
|--------------|------------------------|--------|
| < 10k        | 1 (no partitioning)    | Overhead not worth it |
| 10k - 50k    | 2                      | Modest speedup |
| 50k - 200k   | 4                      | Good balance |
| 200k+        | 4-8                    | Significant speedup |

**Example timings for German-Employment (300k rows):**
- 1 partition: ~48 hours
- 4 partitions: ~12 hours each = 12 hours total (4x speedup)
- 8 partitions: ~6 hours each = 6 hours total (potential resource contention)

### Memory Considerations

Each partition processes `total_rows / partitions`, so:
- More partitions = less memory per job
- Fewer partitions = more memory per job, potentially faster

Current job file allocates 64GB, suitable for partitions of 50k-100k rows.

## Troubleshooting

### Missing Partition Files

If merger reports missing files:
```
⚠️  Warning: Expected 4 files, found 3
```

Check:
1. Did all jobs complete successfully? (`squeue -u $USER`)
2. Check job logs for errors: `llm_slurm_logs/{job_id}/stderr.err`
3. Resubmit failed partition only:
   ```bash
   sbatch --export=ALL,PERCENTAGE=33-33-34,DATASETS=German-Employment,INPUT_DIR=/work/ll95wyqa-user-driven,RESULTS_BASE=llm_evaluation,PARTITION=2/4 llm_evaluation.job
   ```

### Duplicate Records

If you accidentally run the same partition twice, the merger will include duplicates.

**Solution:** Delete duplicate partition files and keep only one copy of each partition.

### Inconsistent Results

Partitions should produce identical results to non-partitioned runs (just split differently).

If results differ:
- Check that all partitions used the same `--percentage` and `--datasets`
- Verify partition numbers are sequential (1/4, 2/4, 3/4, 4/4)

## Migration from Old Workflow

### Old (Single Job)
```bash
sbatch --export=ALL,PERCENTAGE=33-33-34,DATASETS=German-Employment,INPUT_DIR=/work/ll95wyqa-user-driven llm_evaluation.job
```

### New (Partitioned)
```bash
# Launch
python llm_evaluation_launcher.py --percentage 33-33-34 --datasets German-Employment --partitions 4 --input-dir /work/ll95wyqa-user-driven

# Wait for completion, then merge
python llm_evaluation_merger.py --percentage 33-33-34 --datasets German-Employment --partitions 4
```

The old workflow still works! Partitioning is optional via the `--partition` parameter.
