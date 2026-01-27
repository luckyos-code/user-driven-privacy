# LLM Evaluation Partitioning - Visual Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PARTITIONED WORKFLOW                            │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 1: LAUNCH PARTITIONED JOBS                                          │
└──────────────────────────────────────────────────────────────────────────┘

  Input Dataset: german-employment_train.csv (300,000 rows)
       │
       ├─────────────────────┬─────────────────┬─────────────────┐
       │                     │                 │                 │
       v                     v                 v                 v
   Partition 1/4        Partition 2/4     Partition 3/4     Partition 4/4
   (rows 0-75k)        (rows 75k-150k)  (rows 150k-225k)  (rows 225k-300k)
       │                     │                 │                 │
       v                     v                 v                 v
   Job 12345            Job 12346          Job 12347         Job 12348
   [RUNNING]            [RUNNING]          [RUNNING]         [RUNNING]
       │                     │                 │                 │
       │ Parallel Processing (4x speedup)      │                 │
       │                     │                 │                 │
       v                     v                 v                 v
  part1of4.csv          part2of4.csv       part3of4.csv      part4of4.csv
  (75k results)         (75k results)      (75k results)     (75k results)


┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 2: MONITOR PROGRESS                                                 │
└──────────────────────────────────────────────────────────────────────────┘

  $ squeue -u $USER
  
  JOBID  PARTITION  NAME                  STATE    TIME
  12345  paul       llm_eval_p1of4        RUNNING  2:30:45
  12346  paul       llm_eval_p2of4        RUNNING  2:28:12
  12347  paul       llm_eval_p3of4        RUNNING  2:32:01
  12348  paul       llm_eval_p4of4        RUNNING  2:29:38


┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 3: MERGE RESULTS                                                    │
└──────────────────────────────────────────────────────────────────────────┘

  part1of4.csv ─┐
  part2of4.csv ─┼─> MERGER ─> german-employment_train_imputation_results.csv
  part3of4.csv ─┤              (300k merged results)
  part4of4.csv ─┘              + Full statistics & summaries


═══════════════════════════════════════════════════════════════════════════

                        FILE STRUCTURE OVERVIEW

llm_evaluation/33-33-34_results/
│
├─── Partition Files (Intermediate - can delete after merge)
│    ├── german-employment_train_imputation_results_part1of4.csv
│    ├── german-employment_train_imputation_results_part2of4.csv
│    ├── german-employment_train_imputation_results_part3of4.csv
│    ├── german-employment_train_imputation_results_part4of4.csv
│    ├── german-employment_train_prediction_results_part1of4.csv
│    ├── ... (8 total partition files for train)
│    └── ... (8 total partition files for test)
│
└─── Merged Files (Final - keep these)
     ├── german-employment_train_imputation_results.csv    ← FINAL
     ├── german-employment_train_prediction_results.csv    ← FINAL
     ├── german-employment_test_imputation_results.csv     ← FINAL
     ├── german-employment_test_prediction_results.csv     ← FINAL
     ├── german-employment_train_imputed_dataset.csv       ← FINAL
     ├── german-employment_test_imputed_dataset.csv        ← FINAL
     └── german-employment_evaluation_merged.log           ← FINAL

═══════════════════════════════════════════════════════════════════════════

                           COMMAND REFERENCE

┌─────────────────────────────────────────────────────────────────────────┐
│ METHOD 1: Helper Script (Easiest)                                       │
└─────────────────────────────────────────────────────────────────────────┘

  ./llm_eval.sh launch 33-33-34 German-Employment 4
  ./llm_eval.sh status
  ./llm_eval.sh merge 33-33-34 German-Employment 4

┌─────────────────────────────────────────────────────────────────────────┐
│ METHOD 2: Python Scripts (More Control)                                 │
└─────────────────────────────────────────────────────────────────────────┘

  python llm_evaluation_launcher.py \
      --percentage 33-33-34 \
      --datasets German-Employment \
      --partitions 4 \
      --input-dir /work/ll95wyqa-user-driven

  python llm_evaluation_merger.py \
      --percentage 33-33-34 \
      --datasets German-Employment \
      --partitions 4

┌─────────────────────────────────────────────────────────────────────────┐
│ METHOD 3: Manual SLURM (Advanced)                                       │
└─────────────────────────────────────────────────────────────────────────┘

  sbatch --export=ALL,PERCENTAGE=33-33-34,DATASETS=German-Employment,\
  INPUT_DIR=/work/ll95wyqa-user-driven,PARTITION=1/4 llm_evaluation.job

  sbatch --export=ALL,PERCENTAGE=33-33-34,DATASETS=German-Employment,\
  INPUT_DIR=/work/ll95wyqa-user-driven,PARTITION=2/4 llm_evaluation.job

  ... (repeat for 3/4 and 4/4)

═══════════════════════════════════════════════════════════════════════════

                         PERFORMANCE COMPARISON

┌─────────────────────────────────────────────────────────────────────────┐
│ Dataset: German-Employment (300,000 rows)                                │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┬──────────┬─────────┬────────────────────────┐
  │ Partitions   │ Runtime  │ Speedup │ Wall Clock             │
  ├──────────────┼──────────┼─────────┼────────────────────────┤
  │ 1 (no split) │ 48 hours │   1x    │ ████████████████████   │
  │ 2            │ 24 hours │   2x    │ ██████████             │
  │ 4            │ 12 hours │   4x    │ █████ ← RECOMMENDED    │
  │ 8            │  6 hours │   8x    │ ██                     │
  └──────────────┴──────────┴─────────┴────────────────────────┘

  Note: 8 partitions may experience diminishing returns due to overhead

═══════════════════════════════════════════════════════════════════════════

                       PARTITION SIZE GUIDELINES

  Dataset Size       │ Recommended Partitions │ Rows per Partition
  ───────────────────┼────────────────────────┼────────────────────
  < 10,000 rows      │ 1 (no partitioning)    │ All rows
  10,000 - 50,000    │ 2                      │ 5,000 - 25,000
  50,000 - 200,000   │ 4                      │ 12,500 - 50,000
  200,000 - 500,000  │ 4                      │ 50,000 - 125,000
  > 500,000          │ 4-8                    │ 62,500 - 125,000

  RULE OF THUMB: Aim for 50,000 - 100,000 rows per partition

═══════════════════════════════════════════════════════════════════════════

                       TROUBLESHOOTING FLOWCHART

  Job Failed?
      │
      ├─> Check logs: tail llm_slurm_logs/{JOB_ID}/stderr.err
      │
      ├─> Resubmit failed partition:
      │   sbatch --export=ALL,PERCENTAGE=...,PARTITION=2/4 llm_evaluation.job
      │
      └─> Continue with other partitions

  Merger Can't Find Files?
      │
      ├─> Verify all jobs completed: squeue -u $USER
      │
      ├─> Check partition count matches:
      │   ls llm_evaluation/33-33-34_results/*part*of4*
      │
      └─> Ensure correct --partitions argument

  Results Seem Wrong?
      │
      ├─> Compare partition results consistency
      │
      ├─> Verify same --percentage across all partitions
      │
      └─> Check for duplicate partition runs (delete extras)

═══════════════════════════════════════════════════════════════════════════
```
