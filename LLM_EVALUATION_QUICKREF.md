# LLM Evaluation Quick Reference

## Easiest Method (Helper Script)

```bash
# 1. Launch partitioned jobs
./llm_eval.sh launch 33-33-34 German-Employment 4

# 2. Check status
./llm_eval.sh status

# 3. Merge when complete
./llm_eval.sh merge 33-33-34 German-Employment 4
```

## Single Job (Small Datasets < 50k rows)

```bash
# Submit job
sbatch --export=ALL,PERCENTAGE=33-33-34,DATASETS=Adult,INPUT_DIR=/work/ll95wyqa-user-driven llm_evaluation.job

# Or run locally
python llm_evaluation.py --percentage 33-33-34 --datasets Adult --n-samples 100
```

## Partitioned Jobs (Large Datasets > 50k rows)

```bash
# 1. Launch partitioned jobs (4 partitions recommended for 100k-300k rows)
python llm_evaluation_launcher.py \
    --percentage 33-33-34 \
    --datasets German-Employment \
    --partitions 4 \
    --input-dir /work/ll95wyqa-user-driven

# 2. Wait for completion
squeue -u $USER

# 3. Merge results
python llm_evaluation_merger.py \
    --percentage 33-33-34 \
    --datasets German-Employment \
    --partitions 4
```

## Common Commands

### Check Job Status
```bash
squeue -u $USER                    # All your jobs
squeue -j 12345                    # Specific job
```

### View Logs
```bash
tail -f llm_slurm_logs/12345/stdout.out    # Live output
tail -f llm_slurm_logs/12345/stderr.err    # Errors
```

### Resubmit Failed Partition
```bash
# If partition 2/4 failed
sbatch --export=ALL,PERCENTAGE=33-33-34,DATASETS=German-Employment,INPUT_DIR=/work/ll95wyqa-user-driven,PARTITION=2/4 llm_evaluation.job
```

### Dry Run (Test)
```bash
python llm_evaluation_launcher.py \
    --percentage 33-33-34 \
    --datasets German-Employment \
    --partitions 4 \
    --dry-run
```

## File Locations

### Results
```
llm_evaluation/33-33-34_results/
├── german-employment_train_imputation_results.csv
├── german-employment_train_prediction_results.csv
├── german-employment_test_imputation_results.csv
├── german-employment_test_prediction_results.csv
├── german-employment_train_imputed_dataset.csv
├── german-employment_test_imputed_dataset.csv
└── german-employment_evaluation_merged.log
```

### Partition Files (Intermediate)
```
llm_evaluation/33-33-34_results/
├── german-employment_train_imputation_results_part1of4.csv
├── german-employment_train_imputation_results_part2of4.csv
├── german-employment_train_imputation_results_part3of4.csv
└── german-employment_train_imputation_results_part4of4.csv
```

### Logs
```
llm_slurm_logs/
├── 12345/
│   ├── stdout.out
│   └── stderr.err
├── 12346/
│   ├── stdout.out
│   └── stderr.err
...
```

## Recommended Partitions by Dataset Size

| Dataset Size | Partitions | Example |
|--------------|------------|---------|
| < 10k rows   | 1          | Adult, Diabetes |
| 10k-50k rows | 2          | Small German-Employment subset |
| 50k-200k rows| 4          | Medium datasets |
| 200k+ rows   | 4-8        | Full German-Employment (300k) |

## Troubleshooting

### "No partition files found"
- Check job completion: `squeue -u $USER`
- Check logs: `tail llm_slurm_logs/*/stderr.err`
- Ensure correct `--partitions` number

### "Expected 4 files, found 3"
- One job failed, check logs
- Resubmit missing partition

### Results differ from non-partitioned run
- Verify same `--percentage` across all partitions
- Check partition sequence is complete (1/4, 2/4, 3/4, 4/4)

## Performance Tips

- **4 partitions** is the sweet spot for most large datasets
- Each partition should process ~50k-100k rows
- Don't exceed 8 partitions (diminishing returns + overhead)
- Monitor memory with `sacct -j JOBID --format=JobID,MaxRSS`
