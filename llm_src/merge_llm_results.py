#!/usr/bin/env python3
"""
Merge partitioned LLM evaluation results.
Usage: python merge_llm_results.py --results-dir llm_evaluation/33-33-34_results
"""

import argparse
import os
import pandas as pd
import glob
import re

def merge_results(results_dir, delete_parts=False):
    print(f"Merging results in {results_dir}...")
    
    # Find all partition files
    # Pattern: {dataset}_{part}_{task}_results_part{i}of{n}.csv
    # Example: adult_train_imputation_results_part1of4.csv
    
    files = glob.glob(os.path.join(results_dir, "*_part*of*.csv"))
    
    if not files:
        print("No partitioned files found.")
        return

    # Group files by their base name (excluding the partition suffix)
    # Regex to capture the base name and the partition info
    # We want to group: adult_train_imputation_results_part1of4.csv -> adult_train_imputation_results
    
    groups = {}
    for f in files:
        filename = os.path.basename(f)
        match = re.match(r"(.+)_part\d+of\d+\.csv", filename)
        if match:
            base_name = match.group(1)
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(f)
    
    for base_name, file_list in groups.items():
        print(f"Processing {base_name} ({len(file_list)} parts)...")
        
        dfs = []
        for f in sorted(file_list):
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if not dfs:
            continue
            
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Sort if possible (e.g. by record_id)
        if 'record_id' in merged_df.columns:
            # Try to sort by numeric id if possible
            try:
                # Extract number from record_id if it's like "ID_123"
                # This is dataset specific, so maybe just string sort
                merged_df = merged_df.sort_values('record_id')
            except:
                pass
        
        output_path = os.path.join(results_dir, f"{base_name}.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"✓ Merged {len(merged_df)} records to {output_path}")
        
        if delete_parts:
            print("Deleting partition files...")
            for f in file_list:
                os.remove(f)

    # Also merge imputed datasets if they exist
    # Pattern: {dataset}_{part}_imputed_dataset_part{i}of{n}.csv
    dataset_files = glob.glob(os.path.join(results_dir, "*_imputed_dataset_part*of*.csv"))
    dataset_groups = {}
    for f in dataset_files:
        filename = os.path.basename(f)
        match = re.match(r"(.+)_part\d+of\d+\.csv", filename)
        if match:
            base_name = match.group(1)
            if base_name not in dataset_groups:
                dataset_groups[base_name] = []
            dataset_groups[base_name].append(f)
            
    for base_name, file_list in dataset_groups.items():
        print(f"Processing dataset {base_name} ({len(file_list)} parts)...")
        dfs = []
        for f in sorted(file_list):
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if not dfs:
            continue
            
        merged_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(results_dir, f"{base_name}.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"✓ Merged dataset to {output_path}")
        
        if delete_parts:
            for f in file_list:
                os.remove(f)

def main():
    parser = argparse.ArgumentParser(description='Merge partitioned LLM results')
    parser.add_argument('--results-dir', type=str, required=True, help='Directory containing results')
    parser.add_argument('--delete-parts', action='store_true', help='Delete partition files after merging')
    args = parser.parse_args()
    
    merge_results(args.results_dir, args.delete_parts)

if __name__ == '__main__':
    main()
