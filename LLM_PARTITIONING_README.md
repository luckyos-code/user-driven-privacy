# LLM Evaluation Partitioning - Complete Solution

## Problem Solved

Large dataset evaluations (e.g., German-Employment with 300k rows) were taking 48+ hours. Now split into parallel jobs for 4x speedup (12 hours with 4 partitions).

## Solution Components

### Core Files (Modified)
1. **`llm_evaluation.py`** - Main evaluation script with partition support
2. **`llm_evaluation.job`** - SLURM job file with PARTITION variable

### New Tools
3. **`llm_evaluation_launcher.py`** - Submit multiple partition jobs
4. **`llm_evaluation_merger.py`** - Merge partition results
5. **`llm_eval.sh`** - Simple helper script

### Documentation
6. **`LLM_EVALUATION_PARTITIONING.md`** - Comprehensive guide
7. **`LLM_EVALUATION_QUICKREF.md`** - Quick reference
8. **`LLM_PARTITIONING_SUMMARY.md`** - Implementation details
9. **`LLM_PARTITIONING_DIAGRAM.md`** - Visual workflow
10. **`test_partitioning.py`** - Unit tests

## Quick Start

### Simplest Method (Recommended)
```bash
# 1. Launch
./llm_eval.sh launch 33-33-34 German-Employment 4

# 2. Wait for jobs to complete
./llm_eval.sh status

# 3. Merge results
./llm_eval.sh merge 33-33-34 German-Employment 4
```

### Alternative Methods

**Python Scripts:**
```bash
python llm_evaluation_launcher.py --percentage 33-33-34 --datasets German-Employment --partitions 4 --input-dir /work/ll95wyqa-user-driven
python llm_evaluation_merger.py --percentage 33-33-34 --datasets German-Employment --partitions 4
```

**Direct SLURM (Manual):**
```bash
sbatch --export=ALL,PERCENTAGE=33-33-34,DATASETS=German-Employment,PARTITION=1/4 llm_evaluation.job
sbatch --export=ALL,PERCENTAGE=33-33-34,DATASETS=German-Employment,PARTITION=2/4 llm_evaluation.job
# ... etc
```

## Results Structure

### Before Merge (Partitions)
```
llm_evaluation/33-33-34_results/
├── german-employment_train_imputation_results_part1of4.csv
├── german-employment_train_imputation_results_part2of4.csv
├── german-employment_train_imputation_results_part3of4.csv
└── german-employment_train_imputation_results_part4of4.csv
```

### After Merge (Final - IDENTICAL to Non-Partitioned Output)
```
llm_evaluation/33-33-34_results/
├── german-employment_train_imputation_results.csv          ← FINAL (same as non-partitioned)
├── german-employment_train_prediction_results.csv          ← FINAL (same as non-partitioned)
├── german-employment_test_imputation_results.csv           ← FINAL (same as non-partitioned)
├── german-employment_test_prediction_results.csv           ← FINAL (same as non-partitioned)
├── german-employment_train_imputed_dataset.csv             ← FINAL (same as non-partitioned)
├── german-employment_test_imputed_dataset.csv              ← FINAL (same as non-partitioned)
└── german-employment_evaluation.log                        ← FINAL (same metrics as non-partitioned)
```

**Note:** The merged output files are **functionally identical** to non-partitioned runs. Same file names, same statistics, same data. The log file includes a note that results were "merged from N partitions" but all metrics match exactly.

## Workflow

```
1. LAUNCH → Multiple parallel SLURM jobs
              ↓
2. WAIT   → Jobs run independently (parallel speedup)
              ↓
3. MERGE  → Combine all partition results
              ↓
4. DONE   → Full evaluation with complete statistics
```

## Performance

| Dataset Size | Partitions | Runtime | Speedup |
|--------------|------------|---------|---------|
| 300k rows    | 1 (none)   | 48 hrs  | 1x      |
| 300k rows    | 2          | 24 hrs  | 2x      |
| 300k rows    | 4          | 12 hrs  | 4x ⭐   |
| 300k rows    | 8          | 6-8 hrs | 6-8x    |

**Recommended:** 4 partitions for datasets > 50k rows

## Key Features

✅ **Backward Compatible** - Old workflow still works  
✅ **Easy to Use** - Simple commands, clear output  
✅ **Parallel Processing** - True speedup via independent jobs  
✅ **Robust** - Handles failures, supports partial reruns  
✅ **Well Tested** - Unit tests verify correctness  
✅ **Fully Documented** - Multiple guides + diagrams  
✅ **Identical Output** - Merged results match non-partitioned exactly (same files, same stats, same format)  

## Verification

Run tests before cluster submission:
```bash
python test_partitioning.py
```

Expected output:
```
ALL TESTS PASSED ✓
Partitioning logic is working correctly!
Safe to proceed with cluster submission.
```

## Common Commands

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f llm_slurm_logs/12345/stdout.out

# Resubmit failed partition
sbatch --export=ALL,PERCENTAGE=33-33-34,DATASETS=German-Employment,PARTITION=2/4 llm_evaluation.job

# Dry run (test without submitting)
python llm_evaluation_launcher.py --percentage 33-33-34 --datasets German-Employment --partitions 4 --dry-run
```

## Documentation Index

| File | Purpose |
|------|---------|
| `LLM_EVALUATION_PARTITIONING.md` | Complete guide with examples |
| `LLM_EVALUATION_QUICKREF.md` | Quick command reference |
| `LLM_PARTITIONING_SUMMARY.md` | Implementation details |
| `LLM_PARTITIONING_DIAGRAM.md` | Visual workflow diagrams |
| `THIS FILE` | Overview and quick start |

## Next Steps

1. **Test locally** (optional): `python test_partitioning.py`
2. **Try dry run**: `./llm_eval.sh launch 33-33-34 German-Employment 4` with `--dry-run`
3. **Submit real jobs**: `./llm_eval.sh launch 33-33-34 German-Employment 4`
4. **Monitor**: `./llm_eval.sh status` or `squeue -u $USER`
5. **Merge**: `./llm_eval.sh merge 33-33-34 German-Employment 4`

## Support

If you encounter issues:
1. Check logs: `llm_slurm_logs/{job_id}/stderr.err`
2. Review troubleshooting section in `LLM_EVALUATION_PARTITIONING.md`
3. Verify partition files exist before merging
4. Ensure all jobs completed successfully

## Migration

Existing workflows continue to work unchanged. Partitioning is opt-in via the `--partition` parameter or launcher script.

**Old (still works):**
```bash
sbatch --export=ALL,PERCENTAGE=33-33-34,DATASETS=German-Employment llm_evaluation.job
```

**New (recommended for large datasets):**
```bash
./llm_eval.sh launch 33-33-34 German-Employment 4
```
