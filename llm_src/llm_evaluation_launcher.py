#!/usr/bin/env python3
"""
LLM Evaluation Launcher - Submit multiple SLURM jobs for partitioned dataset processing

This script divides large datasets into partitions and submits separate SLURM jobs
for each partition to speed up processing.

Usage:
    python llm_evaluation_launcher.py --percentage 33-33-34 --datasets German-Employment --partitions 4
    python llm_evaluation_launcher.py --percentage 33-33-34 --datasets Adult-Diabetes --partitions 2 --dry-run
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def submit_partition_job(
    partition_num,
    total_partitions,
    percentage,
    datasets,
    input_dir,
    results_base,
    batch_size=1,
    job_file='llm.job',
    dry_run=False
):
    """Submit a single partition job to SLURM."""
    partition_str = f"{partition_num}/{total_partitions}"
    
    # Create unique job name
    job_name = f"llm_eval_p{partition_num}of{total_partitions}"
    
    # Build SLURM command
    cmd = [
        'sbatch',
        f'--job-name={job_name}',
        '--export=ALL,' + ','.join([
            f'PERCENTAGE={percentage}',
            f'DATASETS={datasets}',
            f'INPUT_DIR={input_dir}',
            f'RESULTS_BASE={results_base}',
            f'PARTITION={partition_str}',
            f'BATCH_SIZE={batch_size}'
        ]),
        job_file
    ]
    
    if dry_run:
        print(f"[DRY RUN] Would submit: {' '.join(cmd)}")
        return None
    else:
        print(f"Submitting partition {partition_num}/{total_partitions}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"  ✓ Job {job_id} submitted for partition {partition_num}/{total_partitions}")
            return job_id
        else:
            print(f"  ✗ Failed to submit partition {partition_num}/{total_partitions}")
            print(f"  Error: {result.stderr}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='Launch partitioned LLM evaluation jobs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--percentage',
        type=str,
        default='33-33-34',
        help='Dataset split percentage (single or comma-separated list, e.g., "33-33-34" or "33-33-34,66-17-17")'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='German-Employment',
        help='Hyphen-separated list of datasets'
    )
    parser.add_argument(
        '--partitions',
        type=int,
        default=4,
        help='Number of partitions to split dataset into'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data',
        help='Base input directory'
    )
    parser.add_argument(
        '--results-base',
        type=str,
        default='llm_evaluation',
        help='Base folder for results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for LLM requests'
    )
    parser.add_argument(
        '--job-file',
        type=str,
        default='llm.job',
        help='SLURM job file to use (relative to project root)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without submitting'
    )
    
    args = parser.parse_args()
    
    # Parse percentages (support comma-separated list)
    percentages = [p.strip() for p in args.percentage.split(',')]
    
    print(f"{'='*80}")
    print(f"LLM EVALUATION LAUNCHER")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Percentages: {', '.join(percentages)}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Partitions: {args.partitions}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Results base: {args.results_base}")
    print(f"  Job file: {args.job_file}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*80}\n")
    
    # Check job file exists
    if not os.path.exists(args.job_file):
        print(f"ERROR: Job file '{args.job_file}' not found")
        sys.exit(1)
    
    # Submit jobs for each percentage and each partition
    all_job_ids = {}
    for percentage in percentages:
        print(f"\nSubmitting jobs for percentage: {percentage}")
        print(f"-" * 80)
        
        job_ids = []
        for i in range(1, args.partitions + 1):
            job_id = submit_partition_job(
                partition_num=i,
                total_partitions=args.partitions,
                percentage=percentage,
                datasets=args.datasets,
                input_dir=args.input_dir,
                results_base=args.results_base,
                batch_size=args.batch_size,
                job_file=args.job_file,
                dry_run=args.dry_run
            )
            if job_id:
                job_ids.append(job_id)
        
        all_job_ids[percentage] = job_ids
    
        all_job_ids[percentage] = job_ids
    
    print(f"\n{'='*80}")
    if args.dry_run:
        print(f"DRY RUN COMPLETE")
        print(f"Would have submitted {sum(len(jobs) for jobs in all_job_ids.values())} jobs across {len(percentages)} percentages")
    else:
        print(f"SUBMISSION COMPLETE")
        total_jobs = sum(len(jobs) for jobs in all_job_ids.values())
        print(f"✓ Submitted {total_jobs} jobs across {len(percentages)} percentages")
        for percentage, job_ids in all_job_ids.items():
            print(f"  {percentage}: {len(job_ids)} jobs ({', '.join(job_ids)})")
        print(f"\nTo merge results after completion, run:")
        for percentage in percentages:
            print(f"  python llm_src/llm_evaluation_merger.py --percentage {percentage} --datasets {args.datasets} --partitions {args.partitions}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
