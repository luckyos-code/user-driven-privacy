#!/usr/bin/env python3
"""
LLM Evaluation Merger - Combine results from partitioned evaluation runs

This script merges partial results from multiple partitions into complete
evaluation results, including regenerating summary statistics and logs.

Usage:
    python llm_evaluation_merger.py --percentage 33-33-34 --datasets German-Employment --partitions 4
    python llm_evaluation_merger.py --percentage 33-33-34 --datasets Adult-Diabetes --partitions 2 --results-base llm_evaluation
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


def compute_classification_metrics(results_df, positive_label='1'):
    """Compute binary classification metrics."""
    if results_df is None or len(results_df) == 0:
        return None

    y_true = results_df['true_value'].astype(str).str.strip().fillna("")
    y_pred = results_df['predicted_value'].astype(str).str.strip().fillna("")

    def _norm_label(x):
        try:
            return str(int(float(x)))
        except Exception:
            return x

    y_true = y_true.map(_norm_label)
    y_pred = y_pred.map(_norm_label)

    pos = str(positive_label)
    tp = int(((y_pred == pos) & (y_true == pos)).sum())
    fp = int(((y_pred == pos) & (y_true != pos)).sum())
    fn = int(((y_pred != pos) & (y_true == pos)).sum())
    tn = int(((y_pred != pos) & (y_true != pos)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    support = int((y_true == pos).sum())
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'support': support, 'accuracy': accuracy
    }


def _is_binary_labels(series):
    """Return True if the series contains only binary labels (0/1)."""
    vals = set(series.dropna().astype(str).str.strip())
    if len(vals) == 0:
        return False
    normalized = set()
    for v in vals:
        try:
            nv = str(int(float(v)))
            normalized.add(nv)
        except Exception:
            normalized.add(v)
    return normalized <= {"0", "1"}


def print_evaluation_summary(results_df, task_name, log_file=None):
    """Print summary statistics for evaluation results."""
    def _log(msg):
        print(msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
    
    _log(f"\n{'='*80}")
    _log(f"{task_name} EVALUATION RESULTS (MERGED)")
    _log(f"{'='*80}")

    total = len(results_df)
    _log(f"Total predictions: {total}")
    
    if total == 0:
        _log("⚠️  No predictions to evaluate (empty results)")
        return

    if 'correct' in results_df.columns:
        correct_count = int(results_df['correct'].sum())
        accuracy = (correct_count / total) if total > 0 else None
        _log(f"Correct predictions: {correct_count}")
        if accuracy is not None:
            _log(f"Accuracy: {accuracy:.2%}")

    if all(c in results_df.columns for c in ('predicted_value', 'true_value')):
        if _is_binary_labels(results_df['true_value']) and _is_binary_labels(results_df['predicted_value']):
            metrics = compute_classification_metrics(results_df, positive_label='1')
            if metrics:
                _log(f"\nBinary classification metrics (positive label=1):")
                _log(f"  Precision: {metrics['precision']:.2%}")
                _log(f"  Recall:    {metrics['recall']:.2%}")
                _log(f"  F1 score:  {metrics['f1']:.2%}")
                _log(f"  Support:   {metrics['support']}")
                _log(f"  Confusion: TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")

    if 'value_type' in results_df.columns:
        _log(f"\nBy value type:")
        grouped = results_df.groupby('value_type')['correct'].agg(['mean', 'sum', 'count'])
        for vtype, row in grouped.iterrows():
            _log(f"  {vtype}: {row['mean']:.2%} ({int(row['sum'])}/{int(row['count'])})")

    if 'column' in results_df.columns:
        _log(f"\nBy column:")
        grouped = results_df.groupby('column')['correct'].agg(['mean', 'sum', 'count'])
        for col, row in grouped.iterrows():
            _log(f"  {col}: {row['mean']:.2%} ({int(row['sum'])}/{int(row['count'])})")


def merge_partitions(
    results_dir,
    dataset_name,
    part_name,
    task_type,
    total_partitions,
    log_file=None
):
    """Merge partition files for a specific task."""
    print(f"\nMerging {task_type} results for {dataset_name} ({part_name})...")
    
    # Find all partition files
    pattern = f"{results_dir}/{dataset_name}_{part_name}_{task_type}_results_part*of{total_partitions}.csv"
    partition_files = sorted(glob.glob(pattern))
    
    if not partition_files:
        print(f"  ⚠️  No partition files found matching: {pattern}")
        return None
    
    if len(partition_files) != total_partitions:
        print(f"  ⚠️  Warning: Expected {total_partitions} files, found {len(partition_files)}")
        for f in partition_files:
            print(f"    - {os.path.basename(f)}")
    
    # Load and concatenate
    dfs = []
    for file in partition_files:
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"  ✓ Loaded {os.path.basename(file)}: {len(df)} records")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"  ✓ Merged {len(merged_df)} total records from {len(dfs)} partitions")
    
    # Save merged results
    merged_filename = f"{results_dir}/{dataset_name}_{part_name}_{task_type}_results.csv"
    merged_df.to_csv(merged_filename, index=False)
    print(f"  ✓ Saved to {merged_filename}")
    
    # Log the save operation if log file provided
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"\n✓ Results saved to {merged_filename}\n")
    
    return merged_df


def merge_imputed_datasets(
    results_dir,
    dataset_name,
    part_name,
    total_partitions,
    log_file=None
):
    """Merge imputed dataset partitions."""
    print(f"\nMerging imputed datasets for {dataset_name} ({part_name})...")
    
    # Find all partition files
    pattern = f"{results_dir}/{dataset_name}_{part_name}_imputed_dataset_part*of{total_partitions}.csv"
    partition_files = sorted(glob.glob(pattern))
    
    if not partition_files:
        print(f"  ⚠️  No imputed dataset partitions found")
        return None
    
    # Load and concatenate
    dfs = []
    for file in partition_files:
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"  ✓ Loaded {os.path.basename(file)}: {len(df)} records")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"  ✓ Merged {len(merged_df)} total records")
    
    # Save merged dataset
    merged_filename = f"{results_dir}/{dataset_name}_{part_name}_imputed_dataset.csv"
    merged_df.to_csv(merged_filename, index=False)
    print(f"  ✓ Saved to {merged_filename}")
    
    # Log the save operation if log file provided
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"✓ Imputed dataset saved to {merged_filename}\n")
    
    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description='Merge partitioned LLM evaluation results',
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
        required=True,
        help='Hyphen-separated list of datasets'
    )
    parser.add_argument(
        '--partitions',
        type=int,
        required=True,
        help='Number of partitions to merge'
    )
    parser.add_argument(
        '--results-base',
        type=str,
        default='llm_evaluation',
        help='Base folder for results'
    )
    
    args = parser.parse_args()
    
    # Parse datasets and percentages (support comma-separated lists)
    datasets = [d.strip() for d in args.datasets.split('-')]
    percentages = [p.strip() for p in args.percentage.split(',')]
    
    print(f"{'='*80}")
    print(f"LLM EVALUATION MERGER")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Percentages: {', '.join(percentages)}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Partitions: {args.partitions}")
    print(f"  Results base: {args.results_base}")
    print(f"{'='*80}\n")
    
    # Process each percentage
    for percentage in percentages:
        results_dir = os.path.join(args.results_base, f'{percentage}_results')
        
        print(f"\n{'#'*80}")
        print(f"PROCESSING PERCENTAGE: {percentage}")
        print(f"{'#'*80}")
        print(f"Results directory: {results_dir}\n")
        
        if not os.path.exists(results_dir):
            print(f"ERROR: Results directory '{results_dir}' not found")
            print(f"Skipping percentage {percentage}\n")
            continue
        
        # Process each dataset for this percentage
        for dataset in datasets:
            dataset_name = dataset.lower()
            
            # Create merged log file (same format as non-partitioned)
            log_file = f"{results_dir}/{dataset_name}_evaluation.log"
            with open(log_file, 'w') as f:
                f.write(f"{'='*80}\n")
                f.write(f"{dataset.upper()} EVALUATION LOG\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Configuration:\n")
                f.write(f"  Percentage: {percentage}\n")
                f.write(f"  Merged from: {args.partitions} partitions\n")
                f.write(f"{'='*80}\n\n\n")
        
        # VALUE IMPUTATION SECTION
        with open(log_file, 'a') as f:
            f.write(f"{'='*80}\n")
            f.write(f"{dataset.upper()} VALUE IMPUTATION\n")
            f.write(f"{'='*80}\n\n")
        
        # Merge train imputation
        imputation_train_df = merge_partitions(
            results_dir, dataset_name, 'train', 'imputation', args.partitions, log_file
        )
        if imputation_train_df is not None:
            print_evaluation_summary(
                imputation_train_df,
                f"{dataset.upper()} IMPUTATION (train)",
                log_file=log_file
            )
        
        # Merge train imputed dataset
        merge_imputed_datasets(
            results_dir, dataset_name, 'train', args.partitions, log_file
        )
        
        # Merge test imputation
        imputation_test_df = merge_partitions(
            results_dir, dataset_name, 'test', 'imputation', args.partitions, log_file
        )
        if imputation_test_df is not None:
            print_evaluation_summary(
                imputation_test_df,
                f"{dataset.upper()} IMPUTATION (test)",
                log_file=log_file
            )
        
        # Merge test imputed dataset
        merge_imputed_datasets(
            results_dir, dataset_name, 'test', args.partitions, log_file
        )
        
        # TARGET PREDICTION SECTION
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"{dataset.upper()} TARGET PREDICTION\n")
            f.write(f"{'='*80}\n\n")
        
        # Merge train prediction
        prediction_train_df = merge_partitions(
            results_dir, dataset_name, 'train', 'prediction', args.partitions, log_file
        )
        if prediction_train_df is not None:
            print_evaluation_summary(
                prediction_train_df,
                f"{dataset.upper()} PREDICTION (train)",
                log_file=log_file
            )
        
        # Merge test prediction  
        prediction_test_df = merge_partitions(
            results_dir, dataset_name, 'test', 'prediction', args.partitions, log_file
        )
        if prediction_test_df is not None:
            print_evaluation_summary(
                prediction_test_df,
                f"{dataset.upper()} PREDICTION (test)",
                log_file=log_file
            )
        
        # Add final summary to log (matching non-partitioned format)
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"{dataset.upper()} COMPLETE\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"All results merged from {args.partitions} partitions\n")
            f.write(f"Log file: {log_file}\n")
        
            print(f"✓ Evaluation log saved to {log_file}")
    
    print(f"\n{'#'*80}")
    print(f"ALL MERGES COMPLETE")
    print(f"{'#'*80}")
    print(f"\n✓ Processed {len(percentages)} percentages: {', '.join(percentages)}")
    print(f"✓ Processed {len(datasets)} datasets per percentage: {', '.join(datasets)}")
    print(f"✓ Merged from {args.partitions} partitions each")
    print(f"\nGenerated files per percentage/dataset combination:")
    for percentage in percentages:
        print(f"\n  {percentage}:")
        for dataset in datasets:
            dataset_name = dataset.lower()
            print(f"    {dataset}:")
            print(f"      - {dataset_name}_train_imputation_results.csv")
            print(f"      - {dataset_name}_train_prediction_results.csv")
            print(f"      - {dataset_name}_test_imputation_results.csv")
            print(f"      - {dataset_name}_test_prediction_results.csv")
            print(f"      - {dataset_name}_train_imputed_dataset.csv")
            print(f"      - {dataset_name}_test_imputed_dataset.csv")
            print(f"      - {dataset_name}_evaluation.log")


if __name__ == '__main__':
    main()
