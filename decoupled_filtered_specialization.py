"""
Standalone script to create filtered specialization data using record-based approach.
May take significant space depending on parameters, take care if running locally.

Usage:
    python prepare_filtered_specialization.py <dataset> <percentages> <n_duplicates> <filtering_mode>
    
Examples:
    # Adult dataset, 33-33-34 split, 5 variants per record, random filtering
    python prepare_filtered_specialization.py adult 33-33-34 5 random
    
    # With imputation filtering (profile-based)
    python prepare_filtered_specialization.py adult 33-33-34 5 imputation
    
    # With KNN filtering (distance-based)
    python prepare_filtered_specialization.py adult 33-33-34 5 knn
    
    # No filtering (keep all variants)
    python prepare_filtered_specialization.py adult 33-33-34 0 none
    
Output:
    Saves result to: data/<dataset>/specialization_filtered/<percentages>/specialized_filtered.csv
"""

from src.Vorverarbeitung import prepare_specialization_filtered
import os
import sys
import time

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python prepare_filtered_specialization.py <dataset> <percentages> [n_duplicates] [filtering_mode]")
        print("\nExamples:")
        print("  python prepare_filtered_specialization.py adult 33-33-34 5 random")
        print("  python prepare_filtered_specialization.py adult 33-33-34 5 imputation")
        print("  python prepare_filtered_specialization.py adult 33-33-34 5 knn")
        print("  python prepare_filtered_specialization.py adult 33-33-34 0 none  # No filtering")
        sys.exit(1)
    
    dataset = sys.argv[1]
    percentages = sys.argv[2] if len(sys.argv) > 2 else '33-33-34'
    n_duplicates = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    filtering_mode = sys.argv[4] if len(sys.argv) > 4 else 'random'
    
    # Handle "none" or "0" as no filtering
    if filtering_mode.lower() == 'none' or n_duplicates == 0:
        n_duplicates = None
        filtering_mode = None
    
    print("="*80)
    print("RECORD-BASED SPECIALIZATION WITH FILTERING")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Percentages: {percentages}")
    print(f"n_duplicates: {n_duplicates or 'unlimited (no filtering)'}")
    print(f"filtering_mode: {filtering_mode or 'none'}")
    print(f"Realistic mode: True (uses observed values)")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Process with new record-based method
        result_df = prepare_specialization_filtered(
            dataset=dataset,
            data_dir='data',
            percentages=percentages,
            n_duplicates=n_duplicates,
            filtering_mode=filtering_mode,
            limit_to_observed_values=True,  # Realistic mode
            seed=42
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"Total rows: {len(result_df):,}")
        print(f"Unique record_ids: {result_df['record_id'].nunique() if 'record_id' in result_df.columns else 'N/A'}")
        if n_duplicates:
            avg_variants = len(result_df) / result_df['record_id'].nunique() if 'record_id' in result_df.columns else 0
            print(f"Avg variants per record: {avg_variants:.2f}")
        print(f"Processing time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
        
        # Save to output directory
        output_dir = os.path.join('data', dataset, 'specialization_filtered', percentages)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'specialized_filtered.csv')
        
        print(f"\nSaving to: {output_path}")
        result_df.to_csv(output_path, index=False)
        
        file_size_mb = os.path.getsize(output_path) / 1e6
        print(f"File size: {file_size_mb:.1f} MB")
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"\nFiltered specialization saved to:")
        print(f"  {output_path}")
        print(f"\nYou can now load this file directly instead of running specialization + filtering.")
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR!")
        print("="*80)
        print(f"Failed to process specialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
