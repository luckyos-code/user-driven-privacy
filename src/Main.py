import os
import numpy as np
import random
import time
import pandas as pd
import dask
import dask.dataframe as dd
import dask.array as da
import dask.dataframe.methods
import dask.dataframe.dispatch
import dask.dataframe.core
from datetime import timedelta
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

from src.DataLoader import DataLoader, DatasetPart
from src.ModelEvaluator import ModelEvaluator
from src.ResultsIOHandler import ResultsIOHandler
from src.PreparingMethod import PreparingMethod
from src.DatasetManager import DatasetManager
from src.FilteringHandler import FilteringHandler
from src.RecordBasedSpecialization import RecordBasedSpecialization
from src.Vorverarbeitung import extract_observed_values
from src.ImputationHandler import ImputationHandler
from src.dask_utils import count_dask_rows

# Set global seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================================
# FILTERING METHOD SELECTION
# ============================================================================
# Choose which filtering approach to use for specialization:
# - False (OLD): File-based method - specialization creates CSV files, 
#                then loads and merges them, then applies filtering
# - True (NEW):  Record-based method - processes each record sequentially,
#                generating variants and filtering immediately without files
# 
# Both methods produce same kind of result but with different
# performance characteristics:
# - OLD: creates intermediate files
# - NEW: faster, no intermediate files
USE_RECORD_BASED_FILTERING = True  # Set to True to use NEW method

# OBSERVED_VALUES_REF_SAMPLE: Controls which data is used to determine "observed" values for filtering.
# - False:              Do not limit to observed values (use all possible values from hierarchy).
# - 0.0:                Limit to values observed in the generalized data itself.
# - float (0.0 < x <= 1.0): Limit to values observed in x% sample of the ORIGINAL data.
OBSERVED_VALUES_REF_SAMPLE = 0.0 # Use 0.0 to use observed values from generalized data

# USE_ORIGINAL_SAMPLE_REF: Use a sample of the ORIGINAL (non-anonymized) data as reference
# for imputation/KNN instead of the anonymized data.
# - None: Use generalized data (standard behavior)
# - float (0.0-1.0): Use this fraction of original data (e.g. 0.15 = 15%)
USE_ORIGINAL_SAMPLE_REF = None # Set to None to use anonymized data

# IMPUTATION PREFILTER: For large datasets where imputation is too slow.
# Instead of scoring ALL possible variants, first sample randomly, then score only those.
# - None/0: Disabled - score all variants (current behavior)
# - int > 0: Generate this many random variants first, then score with imputation
# Example: Set to 100 to generate 100 random variants, score them, keep top N
IMPUTATION_PREFILTER_RANDOM_SIZE = None # Set to None to disable prefiltering
# ============================================================================

def ensure_consistent_categories(ddf, original_df, categorical_columns):
    """
    Ensure that categorical columns in the Dask DataFrame have the same categories
    as the original DataFrame. This is crucial for XGBoost to handle categories correctly
    across different partitions and datasets.
    
    This version handles the case where partitions have inconsistent category dtypes
    by first converting to string, then to categorical with a unified category set.
    """
    if ddf is None:
        return None
        
    for col in categorical_columns:
        if col in ddf.columns and col in original_df.columns:
            # Get unique categories from original data
            # Ensure original is categorical first to get categories
            if not isinstance(original_df[col].dtype, pd.CategoricalDtype):
                original_df[col] = original_df[col].astype('category')
                
            unique_categories = list(set(original_df[col].cat.categories))
            
            # FIX: Convert to string first to avoid "dtype of categories must be the same" error
            # This happens when partitions have categories stored with different index dtypes
            # (e.g., int64 vs object). Converting to string normalizes the dtype.
            ddf[col] = ddf[col].astype(str).astype('category').cat.set_categories(
                [str(c) for c in unique_categories]
            )
            
    return ddf

def run_evaluation(n_workers: int = 1, use_gpu: bool = False, save_dir: str = None, data_dir: str = None,
                dataset: str = None, train_method: PreparingMethod = None, test_method: PreparingMethod = None,
                group_duplicates: bool = False, filter_by_record_id: bool = False, random_seed: int = 42, percentages: str = None,
                use_caching: bool = False, cache_only: bool = False, redo_cache: bool = True, verify_row_counts: bool = False):
    """
    Main function to run model evaluation with specified configuration.
    
    Args:
        n_workers: Number of workers for distributed processing
        use_gpu: Whether to use GPU acceleration
        save_dir: Directory to save results
        data_dir: Base directory containing datasets
        dataset: Name of the dataset to load
        train_method: Method to apply to training data (None = keep original)
        test_method: Method to apply to testing data (None = keep original)
        group_duplicates: Whether to deduplicate records
        filter_by_record_id: Whether to filter records by ID
        random_seed: Seed for random number generators to ensure reproducibility
        percentages: Percentage string for subfolder (e.g., '33-33-34')
        use_caching: If True, use caching for datasets and filtering to speed up subsequent runs
        cache_only: If True, only prepare and cache datasets without running evaluations
        redo_cache: If True, force recomputation of cache even if it exists
        verify_row_counts: If True, verify calculated row counts against actual (slow but validates accuracy)
    """
    # Set global seeds
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if cache_only:
        print("Running in CACHE-ONLY mode: Will prepare and cache datasets without running evaluations.")
    
    # Start timing the full experiment
    experiment_start_time = time.time()

    # Create experiment configuration dictionary to track all settings and results
    train_method_str = train_method.name if train_method else "original"
    test_method_str = test_method.name if test_method else "original"
    
    # Validate percentages and methods compatibility
    if percentages == "1-0-0":
        # If 1-0-0 is given, only allow original/original methods
        if train_method_str != "original" or test_method_str != "original":
            raise ValueError("Percentages '1-0-0' can only be used with original train and original test methods. "
                           "Use a different percentage split for preprocessing methods.")
    
    # If both methods are original, force percentages to "1-0-0" since only original sets used
    if train_method_str == "original" and test_method_str == "original":
        percentages = "1-0-0"
    experiment_config = {
        # Basic configuration
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "dataset": dataset,
        "training_method": train_method_str,
        "testing_method": test_method_str,
        "group_duplicates": group_duplicates,
        "filter_by_record_id": filter_by_record_id,
        "random_seed": random_seed,
        "percentages": percentages,
        "use_caching": use_caching,
        "cache_only": cache_only,
        "redo_cache": redo_cache,
        
        # Hardware configuration
        "n_workers": int(n_workers),
        "use_gpu": use_gpu,
        "cores": 8,
        "memory": (
            # If dataset is the employment dataset AND we use specialization, request more memory
            '512GB' if (
                (isinstance(dataset, str) and dataset.lower() == "employment")
                and (
                    (train_method and "specialization" in getattr(train_method, "name", "").lower())
                    or (test_method and "specialization" in getattr(test_method, "name", "").lower())
                )
            )
            # Specialization on other datasets
            else '256GB' if (
                (train_method and "specialization" in getattr(train_method, "name", "").lower())
                or (test_method and "specialization" in getattr(test_method, "name", "").lower())
            )
            # Employment dataset without specialization
            else '128GB' if (isinstance(dataset, str) and dataset.lower() == "employment")
            else '64GB'
        ),
        "queue": 'paula',
        "slurm_job_id": os.environ.get('SLURM_JOB_ID'),
        
        # Paths
        "save_dir": save_dir,
        "data_dir": data_dir,
        
        # Results container (will be populated during processing)
        "anonymization_results": [],
        
        # Performance metrics
        "total_runtime": None,
    }

    # Initialize appropriate cluster
    cluster = create_cluster(n_workers=experiment_config["n_workers"],
                             use_gpu=experiment_config["use_gpu"],
                             cores=experiment_config["cores"],
                             memory=experiment_config["memory"],
                             queue=experiment_config["queue"],
                             random_seed=random_seed,
                             data_dir=data_dir)
    client = Client(cluster)
    print("Dask cluster dashboard:", client.dashboard_link)
    experiment_config["dask_dashboard_link"] = client.dashboard_link
    
    # Fix for Dask circular import issue: Preload dask.dataframe on all workers
    # This prevents ImportError when workers deserialize tasks
    def _preload_dask_modules():
        """Import dask modules on worker to prevent circular import issues."""
        import dask
        import dask.dataframe as dd
        import dask.array as da
        # Explicitly import problematic modules that cause circular imports
        import dask.dataframe.methods
        import dask.dataframe.dispatch
        import dask.dataframe.core
        return "Dask modules preloaded successfully"
    
    # Run the preload function on all workers
    preload_results = client.run(_preload_dask_modules)
    print(f"Preloaded dask modules on {len(preload_results)} workers")
    
    # Initialize results handler with experiment config
    results_io_handler = ResultsIOHandler(save_dir, dataset, train_method, test_method, group_duplicates, filter_by_record_id, percentages)
    results_io_handler.set_experiment_config(experiment_config)

    # Determine if we're using weighted methods
    weighted = any(method and "weighted" in method.name 
                for method in [train_method, test_method] if method)
    experiment_config["weighted_methods"] = weighted

    # Determine if highest_confidence should be used
    highest_confidence = any(method and "highest_confidence" in method.name
                        for method in [train_method, test_method] if method)
    experiment_config["highest_confidence"] = highest_confidence

    # Get dataset-specific anonymization enum
    Anonymization = DatasetManager.get_anonymization_class(dataset)
    spalten_dict, spalten_list = DatasetManager.get_spalten_classes(dataset)
    
    # Run evaluation for each anonymization level
    for anonymization in Anonymization:
        # If both train and test methods are "original" (i.e., None or named "original"),
        # only run the "no" anonymization level and skip all others for time saving.
        both_original = ((train_method is None) or (getattr(train_method, "name", "").lower() == "original")) \
                and ((test_method is None) or (getattr(test_method, "name", "").lower() == "original"))
        if both_original and anonymization.name != "no":
            continue
        
        # TODO this only allows 'all' anonymization: consider all attributes as sensitive, change this to allow other modes if needed, see DatasetManager
        if not both_original and anonymization.name not in ["all"]: continue
        
        # Create a specific config for this anonymization level
        anonymization_config = {
            "level": anonymization.name,
            "columns": [col.name for col in anonymization.value] if anonymization.value else [],
            "start_time": time.time(),
            "preprocessing_time": None,
            "filtering_time": None,
            "total_time": None,
            "metrics": {},
            "row_counts_original": {},
            "row_counts_used": {},
            "anonymization_ratios": {},
            "feature_importance": {},
            "percentages": percentages,
        }
        
        print(f"Processing data for ANONYMIZATION LEVEL: {anonymization.name}")
     
        # Initialize data loader with dataset
        preprocessing_start_time = time.time()
        data_loader = DataLoader(data_dir, dataset, percentages=percentages)

        # get ratios of missing vs generalized vs original data for anonymization level
        anonymization_ratios = data_loader.calculate_anonymization_ratios(anonymization, train_method, test_method)
        anonymization_config["anonymization_ratios"] = anonymization_ratios
     
        # Map anonymization.value (enum-like) to actual column classes
        columns = [spalten_dict[col.name] for col in anonymization.value]

        # ============================================================================
        # CACHING PHILOSOPHY: Focus on Specialization Data
        # ============================================================================
        # We only cache specialization-related data because: They generate many filtered variants that can reuse base data
        # ============================================================================

        # Determine which data parts need caching (only specialization methods)
        cache_train = train_method and "specialization" in train_method.name
        cache_test = test_method and "specialization" in test_method.name
        use_specialization_cache = (cache_train or cache_test) and use_caching
        
        # Determine if we should skip expensive preprocessing for record-based filtering
        # When using record-based filtering with specialization, we don't need the merged
        # DataFrame at all - RecordBasedSpecialization works directly from generalized reference
        train_is_specialization = train_method and "specialization" in train_method.name
        test_is_specialization = test_method and "specialization" in test_method.name
        skip_specialization_preprocessing = (
            USE_RECORD_BASED_FILTERING and 
            filter_by_record_id and 
            (train_is_specialization or test_is_specialization)
        )

        data_train, data_test, preprocessing_time = None, None, None

        if skip_specialization_preprocessing:
            # OPTIMIZED PATH: Skip expensive file-based preprocessing for record-based filtering
            # RecordBasedSpecialization will handle data generation directly from generalized reference
            print("Skipping file-based specialization preprocessing (using record-based filtering)")
            
            # Calculate upper bound stats without actually merging data
            data_loader.calculate_specialization_stats_only(columns, train_method, test_method)
            
            # For non-specialization parts, we still need to load the data normally
            # e.g., if train_method=original but test_method=specialization
            if not train_is_specialization:
                # Train is not specialization, load it normally
                data_loader.preprocess(columns, train_method, None)
                data_train, _ = data_loader.get_data()
            else:
                # Train uses specialization - will be handled by RecordBasedSpecialization
                # Use generalized reference as placeholder (won't be used for training)
                data_train = dd.from_pandas(data_loader.data_train_generalized_reference, 
                                           npartitions=data_loader.config["partition_size"])
            
            if not test_is_specialization:
                # Test is not specialization, load it normally
                data_loader.preprocess(columns, None, test_method)
                _, data_test = data_loader.get_data()
            else:
                # Test uses specialization - will be handled by RecordBasedSpecialization
                # Use generalized reference as placeholder (won't be used for testing)
                data_test = dd.from_pandas(data_loader.data_test_generalized_reference,
                                          npartitions=data_loader.config["partition_size"])
            
            preprocessing_time = time.time() - preprocessing_start_time
            print(f"Stats-only preprocessing finished in {timedelta(seconds=preprocessing_time)}")

        elif use_specialization_cache:
            # Use a single cache directory with train.parquet and test.parquet files
            base_cache_key = f"{dataset}_{anonymization.name}_{percentages or 'none'}_specialization"
            base_cache_dir = os.path.join(data_dir, "_cache", base_cache_key)
            base_train_path = os.path.join(base_cache_dir, "train.parquet")
            base_test_path = os.path.join(base_cache_dir, "test.parquet")

            # Check what cached data is available
            train_cached = cache_train and not redo_cache and os.path.exists(base_train_path)
            test_cached = cache_test and not redo_cache and os.path.exists(base_test_path)

            # Load available cached data
            if train_cached:
                data_train = dd.read_parquet(base_train_path)
            if test_cached:
                data_test = dd.read_parquet(base_test_path)

            # Process any missing data parts
            if not train_cached and cache_train:
                # Need to process train data
                if not test_cached:
                    # Process both if neither was cached
                    data_loader.preprocess(columns, train_method, test_method)
                    data_train, data_test = data_loader.get_data()
                else:
                    # Only process train, test is already loaded
                    data_loader.preprocess(columns, train_method, None)
                    data_train, _ = data_loader.get_data()
                    # Keep the cached test data

            elif not test_cached and cache_test:
                # Need to process test data
                if not train_cached:
                    # Process both if neither was cached
                    data_loader.preprocess(columns, train_method, test_method)
                    data_train, data_test = data_loader.get_data()
                else:
                    # Only process test, train is already loaded
                    data_loader.preprocess(columns, None, test_method)
                    _, data_test = data_loader.get_data()
                    # Keep the cached train data

            preprocessing_time = time.time() - preprocessing_start_time

            # Save newly processed data to cache
            os.makedirs(base_cache_dir, exist_ok=True)
            if cache_train and not train_cached and data_train is not None:
                data_train.to_parquet(base_train_path, compression='zstd')
            if cache_test and not test_cached and data_test is not None:
                data_test.to_parquet(base_test_path, compression='zstd')

            if train_cached or test_cached:
                print(f"Loaded cached specialization data (train: {train_cached}, test: {test_cached})")
            if (cache_train and not train_cached) or (cache_test and not test_cached):
                print(f"Saved specialization data to cache (train: {cache_train and not train_cached}, test: {cache_test and not test_cached})")

        else:
            # Non-specialization runs: process without caching
            data_loader.preprocess(columns, train_method, test_method)
            data_train, data_test = data_loader.get_data(verify_row_counts=verify_row_counts)
            preprocessing_time = time.time() - preprocessing_start_time
            print(f"Processed {train_method_str}/{test_method_str} data (without caching because not specialization)")

        # track original row counts right after loading
        # For specialization, use pre-calculated UPPER BOUND counts to avoid expensive computation
        # Note: train_is_specialization and test_is_specialization were defined earlier
        
        if hasattr(data_loader, 'row_count_tracker') and (train_is_specialization or test_is_specialization):
            # Use calculated UPPER BOUND counts from merge metadata (instant, worst-case estimate!)
            # This avoids triggering expensive .compute() on large specialized datasets
            if train_is_specialization:
                original_train_count = data_loader.row_count_tracker[DatasetPart.TRAIN]['current_upper_bound']
            else:
                original_train_count = count_dask_rows(data_train)
                
            if test_is_specialization:
                original_test_count = data_loader.row_count_tracker[DatasetPart.TEST]['current_upper_bound']
            else:
                original_test_count = count_dask_rows(data_test)
            print(f"Using pre-calculated UPPER BOUND row counts where applicable: {original_train_count:,} train, {original_test_count:,} test")
        else:
            # For non-specialization methods, count normally (fast enough)
            original_train_count = count_dask_rows(data_train)
            original_test_count = count_dask_rows(data_test)
        
        anonymization_config["row_counts_original"] = {
            "train": int(original_train_count),
            "test": int(original_test_count)
        }

        print("Row counts original: "+str(anonymization_config["row_counts_original"]))
        
        # Record preprocessing time
        anonymization_config["preprocessing_time"] = preprocessing_time
        if preprocessing_time > 0:
            print(f"Preprocessing finished in {timedelta(seconds=preprocessing_time)}")
        else:
            print("Data loaded from cache, skipping preprocessing" if use_caching else "Caching disabled, preprocessing completed")

        # Helper function to compute Marketer Risk
        def compute_and_log_privacy_metrics(data, method_name=""):
            """Compute privacy metrics and log them."""
            quasi_identifiers = [col.name for col in spalten_list 
                               if col.name not in [DatasetManager.get_label_column(dataset), 
                                                   DatasetManager.get_record_id_column(dataset)]]
            metrics = compute_privacy_metrics(data, quasi_identifiers)
            print(f"Privacy metrics for {method_name}: {metrics}")
            return metrics

        # Run baseline imputation experiments (similar to filtering for specialization)
        if train_method == PreparingMethod.baseline_imputation or test_method == PreparingMethod.baseline_imputation:
            # Get column lists from DatasetManager
            dataset_config = DatasetManager.get_config(dataset)
            cat_cols = dataset_config['categorical_columns']
            num_cols = dataset_config['numerical_columns']
            
            # Define all imputation configurations to try
            imputation_configs = [
                'simple',
                'mice',
                'constrained_mice'
            ]
            
            # Track imputation configurations in experiment config
            anonymization_config["baseline_imputation"] = {
                "enabled": True,
                "impute_train": train_method == PreparingMethod.baseline_imputation,
                "impute_test": test_method == PreparingMethod.baseline_imputation,
                "configs": [{"name": name, "method": name} for name in imputation_configs],
                "results": []
            }
            
            imputation_start_time = time.time()
            
            # Prepare data with NaN (convert generalized/missing once)
            data_train_with_nan = data_train
            data_test_with_nan = data_test
            
            if train_method == PreparingMethod.baseline_imputation:
                print("Converting training data generalized values to NaN...")
                data_train_with_nan = ImputationHandler.convert_to_missing(
                    data_train, dataset, cat_cols + num_cols, missing_indicator="?"
                )
            
            if test_method == PreparingMethod.baseline_imputation:
                print("Converting test data generalized values to NaN...")
                data_test_with_nan = ImputationHandler.convert_to_missing(
                    data_test, dataset, cat_cols + num_cols, missing_indicator="?"
                )
            
            # Run each imputation configuration
            for config_name in imputation_configs:
                print(f"\n{'='*60}")
                print(f"Running baseline imputation: {config_name}")
                print(f"{'='*60}")
                
                # Apply imputation to train data if needed
                if train_method == PreparingMethod.baseline_imputation:
                    print(f"Applying {config_name} to training data...")
                    
                    if config_name == 'simple':
                        imputed_train = ImputationHandler.apply_simple_imputation(
                            data_train_with_nan, cat_cols, num_cols
                        )
                    elif config_name == 'mice':
                        imputed_train = ImputationHandler.apply_mice_imputation(
                            data_train_with_nan, cat_cols, num_cols, random_seed=random_seed
                        )
                    elif config_name == 'constrained_mice':
                        imputed_train = ImputationHandler.apply_constrained_mice_imputation(
                            data_train_with_nan, dataset, cat_cols, num_cols, random_seed=random_seed
                        )
                    
                    train_count = count_dask_rows(imputed_train)
                    print(f"Training data imputation complete. Shape: {train_count}")
                else:
                    imputed_train = data_train
                
                # Apply imputation to test data if needed
                if test_method == PreparingMethod.baseline_imputation:
                    print(f"Applying {config_name} to test data...")
                    
                    if config_name == 'simple':
                        imputed_test = ImputationHandler.apply_simple_imputation(
                            data_test_with_nan, cat_cols, num_cols
                        )
                    elif config_name == 'mice':
                        imputed_test = ImputationHandler.apply_mice_imputation(
                            data_test_with_nan, cat_cols, num_cols, random_seed=random_seed
                        )
                    elif config_name == 'constrained_mice':
                        imputed_test = ImputationHandler.apply_constrained_mice_imputation(
                            data_test_with_nan, dataset, cat_cols, num_cols, random_seed=random_seed
                        )
                    
                    test_count = count_dask_rows(imputed_test)
                    print(f"Test data imputation complete. Shape: {test_count}")
                else:
                    imputed_test = data_test
                
                # Create a config for this specific imputation configuration
                imputation_result_config = {
                    "imputation_method": config_name,
                    "metrics": {},
                    "row_counts_used": {}
                }
                
                # Compute privacy metrics for imputed training data
                privacy_metrics = compute_and_log_privacy_metrics(imputed_train, f"{config_name} training data")
                imputation_result_config["privacy_metrics"] = privacy_metrics
                
                if not cache_only:
                    # FIX: Ensure consistent categories using original data
                    # This is crucial for XGBoost to handle categories correctly, especially after imputation (e.g. MICE)
                    if imputed_train is not None and imputed_test is not None:
                        # Combine train/test original for complete category set
                        original_combined = pd.concat([data_loader.data_train_original, data_loader.data_test_original], ignore_index=True)
                        
                        imputed_train = ensure_consistent_categories(imputed_train, original_combined, cat_cols)
                        imputed_test = ensure_consistent_categories(imputed_test, original_combined, cat_cols)

                    # Run model evaluation with this imputed dataset
                    evaluate_model(
                        dataset, imputed_train, imputed_test, client, weighted=weighted,
                        highest_confidence=highest_confidence, group_duplicates=group_duplicates,
                        use_gpu=use_gpu, config=imputation_result_config, random_seed=random_seed
                    )
                
                # Add imputation result to the anonymization config
                anonymization_config["baseline_imputation"]["results"].append(imputation_result_config)
            
            imputation_time = time.time() - imputation_start_time
            print(f"\nAll baseline imputations finished in {timedelta(seconds=imputation_time)}")

        # Apply filtering for specialized entries if wanted
        elif (any(method and "specialization" in method.name
            for method in [train_method, test_method] if method)  # Check both train and test methods
                and filter_by_record_id):
            record_id_column = DatasetManager.get_record_id_column(dataset)

            # Determine which data parts need filtering
            filter_train = train_method and "specialization" in train_method.name
            filter_test = test_method and "specialization" in test_method.name

            # Define all configurations you want to try
            if anonymization.name == 'no':
                filtering_configs = [(0, None)]
            else:
                filtering_configs = [ 
                    (0, None),       # Keep only unique records
                    
                    (1, 'random'),   # Keep 1 random duplicate
                    (2, 'random'),   # Keep 2 random duplicate
                    (3, 'random'),   # Keep 3 random duplicates
                    (5, 'random'),   # Keep 5 random duplicates
                    
                    (1, 'imputation'),  # Keep 1 duplicate by imputation
                    (2, 'imputation'),  # Keep 2 duplicate by imputation
                    (3, 'imputation'),  # Keep 3 duplicates by imputation
                    (5, 'imputation'),  # Keep 5 duplicates by imputation
                    
                    # knn performed slightly worse than imputation with longer runtimes in multiple tests
                    # (1, 'knn'),      # Keep 1 duplicate using KNN similarity
                    # (3, 'knn'),      # Keep 3 duplicates using KNN similarity
                    # (5, 'knn'),      # Keep 5 duplicates using KNN similarity
                ]
            
            # Track filtering configurations in experiment config
            anonymization_config["filtering"] = {
                "enabled": True,
                "filtered_train": filter_train,
                "filtered_test": filter_test,
                "method": "record_based" if USE_RECORD_BASED_FILTERING else "file_based",
                "configs": [{"n_duplicates": n, "mode": mode} for n, mode in filtering_configs],
                "results": []
            }
           
            # Set up filtering cache directory (reuse base cache dir for specialization runs)
            filtered_cache_dir = os.path.join(base_cache_dir, "filtered") if use_specialization_cache else None

            # Get filtered datasets for all configurations
            filtering_start_time = time.time()

            # ========================================================================
            # CHOOSE FILTERING METHOD: OLD (file-based) vs NEW (record-based)
            # ========================================================================
            if USE_RECORD_BASED_FILTERING:
                # NEW METHOD: Record-based specialization with integrated filtering
                # Process generalized data directly, no intermediate files needed
                print("Using NEW record-based filtering method (integrated specialization + filtering)")
                
                # Extract observed values from GENERALIZED data (not original!)
                # This matches OLD method behavior: extract from the data being specialized
                # The generalized data contains a mix of:
                #   - X% generalized values (high_school, ?, etc.)
                #   - Y% missing values
                #   - Z% original values (9th, 10th, Bachelors, etc.)
                # extract_observed_values only looks at the Z% original values
                # to determine which values we have evidence for
                # Determine observed values source based on OBSERVED_VALUES_REF_SAMPLE
                observed_values_ref_sample_param = False
                observed_values_train = {}
                observed_values_test = {}
                
                if OBSERVED_VALUES_REF_SAMPLE is not False:
                    observed_values_ref_sample_param = True
                    
                    # Determine source data for observed values
                    if isinstance(OBSERVED_VALUES_REF_SAMPLE, (int, float)) and OBSERVED_VALUES_REF_SAMPLE > 0:
                        # Use sample of ORIGINAL data
                        print(f"DEBUG: Using {OBSERVED_VALUES_REF_SAMPLE*100:.1f}% sample of ORIGINAL data for observed values extraction")
                        obs_source_train = data_loader.data_train_original.sample(frac=OBSERVED_VALUES_REF_SAMPLE, random_state=random_seed)
                        obs_source_test = data_loader.data_test_original.sample(frac=OBSERVED_VALUES_REF_SAMPLE, random_state=random_seed)
                    else:
                        # Use GENERALIZED data (default/0.0 behavior)
                        print("DEBUG: Using GENERALIZED data for observed values extraction")
                        obs_source_train = data_loader.data_train_generalized_reference
                        obs_source_test = data_loader.data_test_generalized_reference
                    
                    # Extract observed values
                    observed_values_train = extract_observed_values(
                        dataset, 
                        obs_source_train,
                        data_dir
                    ) if filter_train else {}
                    
                    observed_values_test = extract_observed_values(
                        dataset,
                        obs_source_test,
                        data_dir
                    ) if filter_test else {}
                
                # Get numerical columns
                numerical_columns = DatasetManager.get_numerical_columns(dataset)
                label_col = DatasetManager.get_label_column(dataset)
                
                # Process training data if needed
                if filter_train:
                    print("Processing training data with record-based method...")
                    filtered_train_datasets = {}
                    
                    # Determine reference data
                    if USE_ORIGINAL_SAMPLE_REF is not None:
                        print(f"DEBUG: Using {USE_ORIGINAL_SAMPLE_REF*100:.1f}% sample of ORIGINAL data as reference for training")
                        train_reference = data_loader.data_train_original.sample(frac=USE_ORIGINAL_SAMPLE_REF, random_state=random_seed)
                    else:
                        train_reference = data_loader.data_train_generalized_reference
                    
                    # Create record-based processor
                    rbs_train = RecordBasedSpecialization(
                        dataset_name=dataset,
                        spalten_list=spalten_list,
                        numerical_columns=numerical_columns,
                        record_id_col=record_id_column,
                        label_col=label_col,
                        observed_values_dict=observed_values_train,  # Only pass if extracted
                        limit_to_observed_values=observed_values_ref_sample_param,  # Enable if we have observed values
                        seed=random_seed,
                        imputation_prefilter_size=IMPUTATION_PREFILTER_RANDOM_SIZE
                    )
                    
                    # Process all filtering configurations in one batch (optimized)
                    filtered_train_datasets = rbs_train.process_data_batch(
                        data_loader.data_train_generalized_reference,
                        filtering_configs=filtering_configs,
                        original_reference_data=train_reference
                    )
                else:
                    filtered_train_datasets = {(n, mode): data_train for n, mode in filtering_configs}
                
                # Process test data if needed  
                if filter_test:
                    print("Processing test data with record-based method...")
                    filtered_test_datasets = {}
                    
                    # Determine reference data
                    if USE_ORIGINAL_SAMPLE_REF is not None:
                        print(f"DEBUG: Using {USE_ORIGINAL_SAMPLE_REF*100:.1f}% sample of ORIGINAL data as reference for testing")
                        test_reference = data_loader.data_test_original.sample(frac=USE_ORIGINAL_SAMPLE_REF, random_state=random_seed)
                    else:
                        test_reference = data_loader.data_test_generalized_reference
                    
                    # Create record-based processor
                    rbs_test = RecordBasedSpecialization(
                        dataset_name=dataset,
                        spalten_list=spalten_list,
                        numerical_columns=numerical_columns,
                        record_id_col=record_id_column,
                        label_col=label_col,
                        observed_values_dict=observed_values_test,  # Only pass if extracted
                        limit_to_observed_values=observed_values_ref_sample_param,  # Enable if we have observed values
                        seed=random_seed,
                        imputation_prefilter_size=IMPUTATION_PREFILTER_RANDOM_SIZE
                    )
                    
                    # Process all filtering configurations in one batch (optimized)
                    filtered_test_datasets = rbs_test.process_data_batch(
                        data_loader.data_test_generalized_reference,
                        filtering_configs=filtering_configs,
                        original_reference_data=test_reference
                    )
                else:
                    filtered_test_datasets = {(n, mode): data_test for n, mode in filtering_configs}
                    
            else:
                # OLD METHOD: File-based specialization, then load and filter
                # Creates intermediate CSV files, merges them, then applies filtering
                print("Using OLD file-based filtering method (specialization files + merge + filter)")
                
                # Filter training data if needed
                if filter_train:
                    print("Filtering training data...")
                    filtered_train_datasets = FilteringHandler.filter_specialized_data(
                        data_train, record_id_column, filtering_configs, random_seed,
                        cache_dir=os.path.join(filtered_cache_dir, "train") if filtered_cache_dir else None,
                        redo_cache=redo_cache,
                        original_reference_data=data_loader.data_train_generalized_reference,
                        per_record_counts=data_loader.row_count_tracker[DatasetPart.TRAIN]['per_record_counts'],
                        spalten_dict=spalten_dict,
                        dataset_name=dataset,
                        label_column=DatasetManager.get_label_column(dataset)
                    )
                else:
                    # No filtering needed for train, use original data for all configs
                    filtered_train_datasets = {(n, mode): data_train for n, mode in filtering_configs}

                # Filter test data if needed
                if filter_test:
                    print("Filtering test data...")
                    filtered_test_datasets = FilteringHandler.filter_specialized_data(
                        data_test, record_id_column, filtering_configs, random_seed,
                        cache_dir=os.path.join(filtered_cache_dir, "test") if filtered_cache_dir else None,
                        redo_cache=redo_cache,
                        original_reference_data=data_loader.data_test_generalized_reference,
                        per_record_counts=data_loader.row_count_tracker[DatasetPart.TEST]['per_record_counts'],
                        spalten_dict=spalten_dict,
                        dataset_name=dataset,
                        label_column=DatasetManager.get_label_column(dataset)
                    )
                else:
                    # No filtering needed for test, use original data for all configs
                    filtered_test_datasets = {(n, mode): data_test for n, mode in filtering_configs}

            filtering_time = time.time() - filtering_start_time
            anonymization_config["filtering_time"] = filtering_time
            print(f"Filtering finished in {timedelta(seconds=filtering_time)}")
            
            # Add variant statistics at anonymization level (dataset property, not filtering-specific)
            if USE_RECORD_BASED_FILTERING and filter_by_record_id and hasattr(data_loader, 'row_count_tracker'):
                variant_stats = {"counts": {}}
                
                # Add train statistics if train is being filtered
                if filter_train:
                    # Calculate full feature domain size for train (theoretical max if all categorical were missing)
                    full_domain_size_train = 1
                    for col in spalten_list:
                        # Skip record_id, label, and numerical columns
                        if col.name == record_id_column or col.name == label_col or col.name in numerical_columns:
                            continue
                        
                        # Get all possible leaf values by querying the highest generalization level (always "?")
                        all_possible_values = col.get_value("?") if hasattr(col, 'get_value') else []
                        if all_possible_values is None:
                            all_possible_values = []
                        
                        # Filter by observed values if enabled (using train observed values)
                        if OBSERVED_VALUES_REF_SAMPLE and observed_values_train and col.name in observed_values_train:
                            observed = observed_values_train[col.name]
                            observed_str = {str(v) for v in observed}
                            all_possible_values = [v for v in all_possible_values if str(v) in observed_str]
                        
                        num_values = len(all_possible_values) if all_possible_values else 1
                        full_domain_size_train *= num_values
                    
                    train_per_record = data_loader.row_count_tracker[DatasetPart.TRAIN]['per_record_counts']
                    if train_per_record:
                        train_counts = list(train_per_record.values())
                        variant_stats["counts"]["train"] = {
                            "full_domain_size": full_domain_size_train,
                            "worst_case_sum": sum(train_counts),
                            "worst_case_max": max(train_counts) if train_counts else 0,
                            "worst_case_avg": sum(train_counts) / len(train_counts) if train_counts else 0.0,
                            "num_records": len(train_counts)
                        }
                
                # Add test statistics if test is being filtered
                if filter_test:
                    # Calculate full feature domain size for test (theoretical max if all categorical were missing)
                    full_domain_size_test = 1
                    for col in spalten_list:
                        # Skip record_id, label, and numerical columns
                        if col.name == record_id_column or col.name == label_col or col.name in numerical_columns:
                            continue
                        
                        # Get all possible leaf values by querying the highest generalization level (always "?")
                        all_possible_values = col.get_value("?") if hasattr(col, 'get_value') else []
                        if all_possible_values is None:
                            all_possible_values = []
                        
                        # Filter by observed values if enabled (using test observed values)
                        if OBSERVED_VALUES_REF_SAMPLE and observed_values_test and col.name in observed_values_test:
                            observed = observed_values_test[col.name]
                            observed_str = {str(v) for v in observed}
                            all_possible_values = [v for v in all_possible_values if str(v) in observed_str]
                        
                        num_values = len(all_possible_values) if all_possible_values else 1
                        full_domain_size_test *= num_values
                    
                    test_per_record = data_loader.row_count_tracker[DatasetPart.TEST]['per_record_counts']
                    if test_per_record:
                        test_counts = list(test_per_record.values())
                        variant_stats["counts"]["test"] = {
                            "full_domain_size": full_domain_size_test,
                            "worst_case_sum": sum(test_counts),
                            "worst_case_max": max(test_counts) if test_counts else 0,
                            "worst_case_avg": sum(test_counts) / len(test_counts) if test_counts else 0.0,
                            "num_records": len(test_counts)
                        }
                
                anonymization_config["specialization_variants"] = variant_stats
           
            # Now you can use each filtered dataset as needed
            for (n_duplicates, mode), filtered_train in filtered_train_datasets.items():
                filtered_test = filtered_test_datasets[(n_duplicates, mode)]
                
                # FIX: Ensure consistent categories using original data
                # This matches DataLoader behavior and fixes the performance regression
                # Without this, Dask partitions might have different category codes, confusing XGBoost
                if filtered_train is not None and filtered_test is not None:
                    categorical_columns = DatasetManager.get_categorical_columns(dataset)
                    # Combine train/test original for complete category set (safest)
                    original_combined = pd.concat([data_loader.data_train_original, data_loader.data_test_original], ignore_index=True)
                    
                    filtered_train = ensure_consistent_categories(filtered_train, original_combined, categorical_columns)
                    filtered_test = ensure_consistent_categories(filtered_test, original_combined, categorical_columns)
                
                # Create a config for this specific filtering configuration
                filtering_result_config = {
                    "n_duplicates": n_duplicates,
                    "mode": mode or "unique",
                    "metrics": {},
                    "row_counts_used": {}
                }
                
                # Handle case where datasets are None (shouldn't happen but defensive check)
                if filtered_train is None or filtered_test is None:
                    print(f"Warning: Filtered datasets are None for config (n={n_duplicates}, mode={mode}). Creating dummy entry.")
                    filtering_result_config["error"] = "Filtered datasets are None"
                    filtering_result_config["metrics"] = {
                        "accuracy": None,
                        "f1_score_0": None,
                        "f1_score_1": None,
                        "f1_score_avg": None,
                        "precision": None,
                        "recall": None,
                        "training_time": None,
                        "inference_time": None,
                        "feature_importance": [],
                    }
                    filtering_result_config["row_counts_used"] = {
                        "train": None,
                        "test": None
                    }
                    filtering_result_config["privacy_metrics"] = {
                        "marketer_risk": None,
                        "avg_group_size": None,
                        "min_group_size": None,
                        "max_group_size": None,
                    }
                    anonymization_config["filtering"]["results"].append(filtering_result_config)
                    continue

                # Use efficient counting method
                train_count = count_dask_rows(filtered_train)
                test_count = count_dask_rows(filtered_test)
                
                print(f"Filtered run (n_duplicates, mode, train_filtered, test_filtered): {n_duplicates}, {mode}, {filter_train}, {filter_test}")
                print(f"Train data size: {train_count}, Test data size: {test_count}")
                
                # Compute privacy metrics for filtered training data
                privacy_metrics = compute_and_log_privacy_metrics(filtered_train, f"filtered training data (n={n_duplicates}, mode={mode})")
                filtering_result_config["privacy_metrics"] = privacy_metrics
                
                if not cache_only:
                    # Run your model evaluation with this filtered dataset
                    evaluate_model(
                        dataset, filtered_train, filtered_test, client, weighted=weighted,
                        highest_confidence=highest_confidence, group_duplicates=group_duplicates,
                        use_gpu=use_gpu, config=filtering_result_config, random_seed=random_seed
                    )

                # Add filtering result to the anonymization config
                anonymization_config["filtering"]["results"].append(filtering_result_config)
        
        # Regular evaluation (no filtering or baseline imputation)
        else:
            # Run evaluation and save results without filtering or baseline imputation
            anonymization_config["filtering"] = {"enabled": False}
            anonymization_config["baseline_imputation"] = {"enabled": False}

            # Compute privacy metrics for training data
            privacy_metrics = compute_and_log_privacy_metrics(data_train, "training data")
            anonymization_config["privacy_metrics"] = privacy_metrics
            
            if not cache_only:
                evaluate_model(
                    dataset, data_train, data_test, client, weighted=weighted, highest_confidence=highest_confidence,
                    group_duplicates=group_duplicates, use_gpu=use_gpu, config=anonymization_config, random_seed=random_seed
                )
        
        # Calculate total time for this anonymization level
        anonymization_config["total_time"] = time.time() - anonymization_config["start_time"]
        
        # Add anonymization result to the experiment config
        experiment_config["anonymization_results"].append(anonymization_config)

        # Save the intermediate experiment configuration ("total_runtime" will be None as indicator if not finished)
        results_io_handler.save_experiment_config(experiment_config)

    # Calculate total experiment runtime up to this point
    experiment_config["total_runtime"] = time.time() - experiment_start_time
    
    # Save the final experiment configuration with all results
    results_io_handler.save_experiment_config(experiment_config)
    
    # Clean up resources
    client.close()
    cluster.close()

def create_cluster(n_workers: int = 1, use_gpu: bool = False, cores: int = 8, memory: str = '64GB', queue: str = 'paul', random_seed: int = 42, data_dir: str = None):
    """
    Configure a SLURM cluster or return a local cluster if not in SLURM.

    Parameters:
        n_workers (int): Number of workers to scale the cluster to.
        use_gpu (bool): Whether to request GPUs for the cluster.
        cores (int): Number of cores per job.
        memory (str): Memory per job (e.g., '64GB').
        random_seed (int): Seed for random number generators

    Returns:
        cluster: A configured SLURMCluster or LocalCluster.
    """
    # Get current job ID from SLURM environment if present
    job_id = os.environ.get('SLURM_JOB_ID')
    
    if job_id:
        print(f"Running in SLURM environment with job ID: {job_id}")
     
        # Add GPU-specific directives if use_gpu is True
        job_extra = []
        if use_gpu:
            job_extra.append("--gres=gpu:a30:1")  # Request 1 GPU per job, change to correct type
            print("GPU support enabled: Requesting 1 GPU per job.")

        # Gave all worker jobs inherit the original job's ID as an environment variable
        job_extra.append(f"--export=ALL,PARENT_SLURM_JOB_ID={job_id}")
     
        # Ensure local directory exists and is accessible
        local_dir = os.path.join(data_dir, 'dask-worker-space', job_id)
        os.makedirs(local_dir, exist_ok=True)
        
        cluster = SLURMCluster(
            cores=cores,
            memory=memory,
            processes=1,
            walltime='48:00:00',  # Adjust based on your cluster's limits
            queue=queue,  # Change to the correct SLURM queue
            job_extra=job_extra,
            log_directory=f'slurm_logs/{job_id}/dask_logs',
            local_directory=local_dir,
            # Additional memory management settings
            death_timeout=600,  # Longer timeout for large operations
        )
        cluster.scale(jobs=n_workers)  # Scale number of jobs to workers
        print(f"Dask workers will use {memory} memory each with {cores} cores")
        print(f"Worker scratch space: {local_dir}")
        return cluster
    else:
        print("SLURM environment not detected. Falling back to LocalCluster.")
        return LocalCluster(n_workers=n_workers, threads_per_worker=cores, memory_limit=memory)


def evaluate_model(dataset, data_train, data_test, client, weighted=False, highest_confidence=False,
                   group_duplicates=False, use_gpu=False, config=None, random_seed=42):
    """
    Evaluate model performance and save results for a specific configuration.
    
    Args:
        dataset: Name of the dataset
        data_train: Training data Dask DataFrame
        data_test: Test data Dask DataFrame
        client: Dask client
        weighted: Whether to use weighted samples
        highest_confidence: Whether to use highest confidence predictions
        group_duplicates: Whether to deduplicate records
        use_gpu: Whether to use GPU acceleration
        config: Configuration dictionary to update with results
        random_seed: Seed for random number generators
    """
    # Set seeds for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Create a new config dict if none is provided
    if config is None:
        config = {}
    
    # Record start time
    time.time()
    
    # Get label column name from dataset manager
    label_column = DatasetManager.get_label_column(dataset)
    
    # Prepare features and target variables
    X_train = data_train.drop(columns=[label_column])
    Y_train = data_train[label_column]
    X_test = data_test.drop(columns=[label_column])
    
    # Ensure column order matches between train and test (critical for XGBoost)
    # Use train column order as reference
    train_cols = X_train.columns.tolist()
    X_test = X_test[train_cols]
    
    # Get record_id column for the dataset
    record_id_col = DatasetManager.get_record_id_column(dataset)
    y_test_with_record_ids = data_test[[record_id_col, label_column]].drop_duplicates()

    # Persist the feature and target dataframes for better performance during model training
    # CHANGE: not persisting during training to save memory
    # X_train = X_train.persist()
    # Y_train = Y_train.persist()
    # X_test = X_test.persist()
    # y_test_with_record_ids = y_test_with_record_ids.persist()

    # Initialize and run model evaluation
    model_evaluator = ModelEvaluator(X_train, X_test, Y_train, y_test_with_record_ids, client, dataset)
    
    # Try to train model, handle case where training data is empty or invalid
    try:
        _X_train, _X_test, accuracy, f1_score_0, f1_score_1, precision, recall, training_time, inference_time, feature_importance_df = model_evaluator.train_model(
            weighted, highest_confidence, group_duplicates, use_gpu
        )
        
        # Get prediction results
        true_labels, pred_proba = model_evaluator.get_true_values_and_pred_proba()
        
        # Store metrics
        metrics = {
            "accuracy": float(accuracy),
            "f1_score_0": float(f1_score_0),
            "f1_score_1": float(f1_score_1),
            "f1_score_avg": float((f1_score_0 + f1_score_1) / 2),
            "precision": float(precision),
            "recall": float(recall),
            "training_time": float(training_time),
            "inference_time": float(inference_time),
            "feature_importance": feature_importance_df.to_dict(orient='records') if feature_importance_df is not None else [],
        }
        
        # track used row counts after successful training
        used_train_count = count_dask_rows(_X_train)
        used_test_count = count_dask_rows(_X_test)
        
    except Exception as e:
        print(f"WARNING: Model training failed: {str(e)}")
        print("This typically happens when training data is empty (e.g., n_duplicates=0 with all generalized/missing values)")
        
        # Return null metrics to indicate training failure (null is valid JSON, not NaN)
        metrics = {
            "accuracy": None,
            "f1_score_0": None,
            "f1_score_1": None,
            "f1_score_avg": None,
            "precision": None,
            "recall": None,
            "training_time": 0.0,
            "inference_time": 0.0,
            "feature_importance": [],
            "training_failed": True,
            "failure_reason": str(e)
        }
        
        # Use original counts since training didn't happen
        used_train_count = count_dask_rows(X_train)
        used_test_count = count_dask_rows(X_test)
    
    # Update config
    config["metrics"] = metrics
    
    config["row_counts_used"] = {
        "train": int(used_train_count),
        "test": int(used_test_count)
    }

    print("Row counts used: "+str(config["row_counts_used"]))
    
    # Return metrics for potential further use
    return metrics

def compute_privacy_metrics(data, quasi_identifiers):
    """
    Compute privacy metrics for the dataset, including Marketer Risk (Rm)
    and equivalence class size statistics.
    
    Args:
        data: Dask DataFrame
        quasi_identifiers: list of column names to group by
    
    Returns:
        dict: A dictionary with Rm, and stats about equivalence class sizes.
    """
    # Use efficient counting method instead of .shape[0].compute()
    if not quasi_identifiers:
        n = count_dask_rows(data)
        return {
            "marketer_risk": 1.0 if n > 0 else 0.0,
            "avg_group_size": 1.0 if n > 0 else 0.0,
            "min_group_size": 1 if n > 0 else 0,
            "max_group_size": 1 if n > 0 else 0,
        }
    
    # To avoid pandas creating a MultiIndex (and exploding memory) for
    # multi-column groupby, compute a per-row hash of the quasi-identifier
    # tuple and then compute counts of those hashes. This avoids the
    # MultiIndex product problem and keeps memory usage distributed.
    # Note: hashing can have collisions (very unlikely with 64-bit hashes),
    # which would merge distinct equivalence classes -> slight underestimation
    # of marketer risk. For most practical purposes this is acceptable.

    # Create a Dask Series of 64-bit hashes for quasi-identifier tuples
    def _hash_partition(df):
        # df is a pandas DataFrame for this partition
        # hash_pandas_object returns a Series of integer hashes
        return pd.util.hash_pandas_object(df, index=False).astype('uint64')

    # Use only the quasi-identifier columns for hashing
    qid_df = data[quasi_identifiers]
    hash_series = qid_df.map_partitions(_hash_partition, meta=pd.Series(dtype='uint64', name='_qhash'))

    # Compute value counts of the hashes (counts per equivalence class)
    # Use higher split_out to parallelize better and reduce memory pressure
    counts_dd = hash_series.value_counts(split_out=50)

    # Total number of records using efficient counting
    n = count_dask_rows(data)

    # Compute all statistics in one pass to avoid multiple compute() calls
    if n > 0:
        num_classes, avg_size, min_size, max_size = dask.compute(
            counts_dd.count(),
            counts_dd.mean(),
            counts_dd.min(),
            counts_dd.max()
        )
        num_equivalence_classes = int(num_classes)
        avg_group_size = float(avg_size)
        min_group_size = int(min_size)
        max_group_size = int(max_size)
    else:
        num_equivalence_classes = 0
        avg_group_size = 0.0
        min_group_size = 0
        max_group_size = 0

    # Marketer Risk Rm = number_of_equivalence_classes / n
    Rm = num_equivalence_classes / n if n > 0 else 0.0
    
    return {
        "marketer_risk": Rm,
        "avg_group_size": float(avg_group_size),
        "min_group_size": int(min_group_size),
        "max_group_size": int(max_group_size),
    }