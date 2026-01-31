"""
Record-based specialization with integrated filtering.

This module processes data per record_id instead of per column, avoiding the creation
of full Cartesian products. It produces mathematically identical results to the 
column-based approach but with better performance and memory usage.

Key principle: For each record_id, generate all combinations in memory, apply 
filtering immediately, then move to the next record_id.
"""

import pandas as pd
import numpy as np
import random
import dask
import dask.dataframe as dd
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import os


class RecordBasedSpecialization:
    """
    Specialization processor that works per record_id with integrated filtering.
    
    Produces identical results to column-based + separate filtering approach,
    but never materializes the full Cartesian product.
    """
    
    def __init__(self, dataset_name, spalten_list, numerical_columns, record_id_col, 
                 label_col=None, observed_values_dict=None, limit_to_observed_values=True, seed=42,
                 imputation_prefilter_size=None):
        """
        Initialize record-based processor.
        
        Args:
            dataset_name: Name of the dataset
            spalten_list: List of column class instances
            numerical_columns: List of numerical column names
            record_id_col: Name of the record ID column
            label_col: Name of the label column (to preserve)
            observed_values_dict: Dict of column_name -> set of observed values (for realistic mode)
            limit_to_observed_values: If True, only create variants with observed values (requires observed_values_dict)
            seed: Random seed for reproducibility
            imputation_prefilter_size: If set, use random prefilter before imputation scoring (optimization for large datasets)
        """
        self.dataset_name = dataset_name
        self.spalten_list = spalten_list
        self.numerical_columns = numerical_columns or []
        self.record_id_col = record_id_col
        self.label_col = label_col
        self.observed_values_dict = observed_values_dict or {}
        self.limit_to_observed_values = limit_to_observed_values
        print("Limiting filterting to observed values: "+str(limit_to_observed_values))
        self.seed = seed
        self.imputation_prefilter_size = imputation_prefilter_size
        if imputation_prefilter_size:
            print(f"Imputation prefilter enabled: Will generate {imputation_prefilter_size} random variants before scoring")
    
    def process_data_batch(self, df, filtering_configs: List[Tuple[Optional[int], Optional[str]]],
                          original_reference_data = None):
        """
        Process all records with multiple filtering configurations efficiently.
        Optimizes by grouping same-mode configs and generating/scoring once per mode.
        
        Args:
            df: DataFrame with generalized data (Pandas or Dask, one row per record_id)
            filtering_configs: List of (n_duplicates, mode) tuples, e.g.:
                               [(0, None), (5, 'random'), (10, 'random'), (5, 'imputation'), (10, 'imputation')]
            original_reference_data: Reference data for building profiles/models if needed
            
        Returns:
            Dict mapping (n_duplicates, mode) -> Dask DataFrame with filtered results
        """
        print(f"\n{'='*80}")
        print(f"Record-Based Specialization - Batch Processing (Dask)")
        print(f"{'='*80}")
        print(f"Configurations to process: {len(filtering_configs)}")
        for n, mode in filtering_configs:
            print(f"  - n_duplicates={n}, mode={mode}")
        
        # Group configs by mode to minimize redundant work
        mode_groups = defaultdict(list)
        for n_duplicates, mode in filtering_configs:
            mode_groups[(mode,)].append(n_duplicates)
        
        print(f"\nOptimized grouping ({len(mode_groups)} unique modes):")
        for (mode,), n_list in mode_groups.items():
            max_n = max(n_list) if n_list else None
            print(f"  - mode={mode}: n_duplicates={sorted(n_list)}, will generate for max={max_n}")
        
        # Convert to Dask DataFrame if not already
        if not isinstance(df, dd.DataFrame):
            ddf = dd.from_pandas(df, npartitions=max(1, len(df) // 1000))
        else:
            ddf = df
        
        input_records = len(ddf)
        print(f"\nInput records: {input_records}")
        
        # Process each mode group
        results = {}
        
        for (mode,), n_list in mode_groups.items():
            max_n = max(n_list) if n_list else None
            
            # Special handling for n_duplicates=0 (no mode needed)
            if max_n == 0:
                print(f"\nProcessing n_duplicates=0 (unique records only)...")
                result_ddf = self._process_single_config(ddf, 0, None, None, None, None)
                results[(0, None)] = result_ddf
                continue
            
            # Build resources for this mode
            column_profiles = None
            knn_model = None
            knn_preprocessor = None
            
            if mode == 'imputation':
                print(f"\nBuilding column profiles for imputation mode...")
                reference_data = original_reference_data if original_reference_data is not None else df
                column_profiles = self._build_column_profiles(reference_data)
            elif mode == 'knn':
                print(f"\nBuilding KNN model for knn mode...")
                reference_data = original_reference_data if original_reference_data is not None else df
                knn_model, knn_preprocessor = self._build_knn_model(reference_data)
            
            # Process with max_n to generate enough variants for all configs in this group
            print(f"\nProcessing mode={mode} with max_n={max_n}...")
            result_ddf = self._process_single_config(
                ddf, max_n, mode, column_profiles, knn_model, knn_preprocessor
            )
            
            # For each n in this group, filter the results
            for n in sorted(n_list, reverse=True):  # Process largest first
                if n == max_n:
                    # Already have the right size
                    results[(n, mode)] = result_ddf
                else:
                    # Filter to keep only top-n per record_id
                    print(f"  Filtering to n_duplicates={n}...")
                    filtered_ddf = self._filter_to_n_duplicates(result_ddf, n, mode)
                    results[(n, mode)] = filtered_ddf
        
        print(f"\n{'='*80}")
        print(f"Batch Processing Complete - {len(results)} configurations ready")
        print(f"{'='*80}\n")
        
        return results
    
    def process_data(self, df, n_duplicates: Optional[int] = None, 
                     filtering_mode: Optional[str] = None,
                     column_profiles: Optional[Dict] = None,
                     knn_model = None,
                     knn_preprocessor = None,
                     original_reference_data = None):
        """
        Process all records with specialization and optional filtering using Dask.
        Single-config version - for batch processing use process_data_batch() instead.
        
        Args:
            df: DataFrame with generalized data (Pandas or Dask, one row per record_id)
            n_duplicates: Number of variants to keep per record_id (None = keep all)
            filtering_mode: 'random', 'imputation', 'knn', or None
            column_profiles: Column profiles for imputation mode
            knn_model: Trained KNN model for knn mode
            knn_preprocessor: Preprocessor for KNN features
            original_reference_data: Reference data for building profiles/models if needed
            
        Returns:
            Dask DataFrame with specialized (and optionally filtered) data
        """
        # Delegate to batch processor with single config
        results = self.process_data_batch(
            df, 
            filtering_configs=[(n_duplicates, filtering_mode)],
            original_reference_data=original_reference_data
        )
        return results[(n_duplicates, filtering_mode)]
    
    def _process_single_config(self, ddf, n_duplicates: Optional[int], 
                              filtering_mode: Optional[str],
                              column_profiles: Optional[Dict],
                              knn_model, knn_preprocessor):
        """
        Internal method to process a single configuration.
        
        Returns:
            Dask DataFrame with specialized (and optionally filtered) data
        """
        print(f"Processing via Dask map_partitions (distributed across {ddf.npartitions} partitions)...")
        
        # Create proper meta DataFrame with correct dtypes
        # This ensures Dask preserves the dtypes after _fix_dtypes() is applied
        meta_dict = {}
        for col in ddf.columns:
            if col == self.record_id_col or col == self.label_col:
                # Keep record_id and label as-is
                meta_dict[col] = ddf[col].dtype
            elif col in self.numerical_columns:
                # Numerical columns should be float (after pd.to_numeric conversion)
                meta_dict[col] = 'float64'
            else:
                # Categorical columns should be category dtype
                meta_dict[col] = 'category'
        
        meta_df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in meta_dict.items()})
        
        # Process each partition using map_partitions
        result_ddf = ddf.map_partitions(
            self._process_partition,
            n_duplicates=n_duplicates,
            filtering_mode=filtering_mode,
            column_profiles=column_profiles,
            knn_model=knn_model,
            knn_preprocessor=knn_preprocessor,
            meta=meta_df
        )
        
        # Persist result to distributed memory and compute statistics in one go
        # This triggers computation ONCE, caches result, and gives us stats
        print("\nPersisting result to distributed memory and computing statistics...")
        result_ddf = result_ddf.persist()
        
        # Now get statistics efficiently (data is already computed and cached)
        result_size, unique_records = dask.compute(
            result_ddf.shape[0],
            result_ddf[self.record_id_col].nunique()
        )
        
        print(f"Variants generated and kept: {result_size:,}")
        print(f"Unique record_ids in output: {unique_records}")
        print(f"Avg variants per record: {result_size/max(unique_records, 1):.2f}")
        print(f"Result cached in distributed memory for fast reuse")
        
        return result_ddf
    
    def _filter_to_n_duplicates(self, ddf, n_duplicates: int, mode: Optional[str]):
        """
        Filter a Dask DataFrame to keep only top-n variants per record_id.
        Assumes data is already scored/ordered appropriately.
        
        For random mode: keeps first n (already randomly ordered)
        For imputation/knn: keeps top n by score (assumes sorted by score desc)
        
        Args:
            ddf: Dask DataFrame with variants
            n_duplicates: Number to keep per record_id
            mode: Filtering mode (for logging)
            
        Returns:
            Filtered Dask DataFrame
        """
        def keep_top_n(group_df):
            """Keep first n rows per group (assumes already sorted if needed)"""
            return group_df.head(n_duplicates)
        
        # Group by record_id and keep top n
        filtered_ddf = ddf.groupby(self.record_id_col, group_keys=False).apply(
            keep_top_n,
            meta=ddf._meta
        )
        
        return filtered_ddf
    
    def _process_partition(self, partition_df: pd.DataFrame, n_duplicates: Optional[int],
                          filtering_mode: Optional[str], column_profiles: Optional[Dict],
                          knn_model, knn_preprocessor) -> pd.DataFrame:
        """
        Process a single Dask partition.
        Called by map_partitions for distributed processing.
        
        Args:
            partition_df: Pandas DataFrame partition
            n_duplicates: Number of variants to keep per record_id
            filtering_mode: Filtering mode
            column_profiles: Column profiles for imputation
            knn_model: KNN model for knn mode
            knn_preprocessor: Preprocessor for knn mode
            
        Returns:
            Pandas DataFrame with processed variants
        """
        if len(partition_df) == 0:
            return partition_df
        
        all_results_dfs = []
        partition_variants_generated = 0
        partition_variants_kept = 0
        
        # Process each record_id in this partition
        for record_id, record_group in partition_df.groupby(self.record_id_col):
            # Get single record
            record = record_group.iloc[0]
            
            # For n_duplicates=0, check if record has ANY generalized values
            # (including numerical columns - they use mean imputation but still count as specialized)
            # If yes, skip immediately (can't have exactly 1 variant)
            if n_duplicates == 0:
                has_generalized = self._has_any_generalized_value(record)
                if has_generalized:
                    # Skip - this record will generate multiple variants
                    continue
                else:
                    # No generalized values - keep this single record
                    partition_variants_generated += 1
                    partition_variants_kept += 1
                    all_results_dfs.append(record.to_frame().T)
                    continue
            
            # Generate combinations using unified function with column-wise optimization
            # The function intelligently handles each mode:
            # - random: generates EXACTLY n_duplicates complete variants (no filtering needed)
            # - imputation: reduces to top-N values per column, then Cartesian product, then filters
            # - knn: generates ALL combinations (row-based scoring, can't pre-select)
            variants_df = self._generate_combinations_as_dataframe(
                record, 
                n_duplicates=n_duplicates,
                filtering_mode=filtering_mode,
                column_profiles=column_profiles
            )
            partition_variants_generated += len(variants_df)
            
            # For imputation/KNN: filter after generation to select best n_duplicates
            # Random mode already generates exactly n_duplicates, no filtering needed
            if filtering_mode in ['imputation', 'knn'] and n_duplicates and len(variants_df) > n_duplicates:
                variants_df = self._apply_filtering_to_dataframe(
                    variants_df, record_id, n_duplicates, filtering_mode,
                    column_profiles, knn_model, knn_preprocessor
                )
            
            partition_variants_kept += len(variants_df)
            all_results_dfs.append(variants_df)
        
        # Log partition completion (helps track progress across workers)
        if partition_variants_generated > 0:
            reduction = (1 - partition_variants_kept/max(partition_variants_generated, 1))*100
            print(f"  Partition: {len(partition_df)} records → {partition_variants_generated:,} variants → {partition_variants_kept:,} kept ({reduction:.1f}% reduction)")
        
        # Concatenate all DataFrames once (much faster than converting N times)
        if all_results_dfs:
            result_df = pd.concat(all_results_dfs, ignore_index=True)
        else:
            # Empty result with same schema
            result_df = pd.DataFrame(columns=partition_df.columns)
        
        # Fix dtypes: convert numerical columns from object to numeric
        # This is crucial for XGBoost which requires proper dtypes
        result_df = self._fix_dtypes(result_df)
        
        return result_df
    
    def _fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to proper dtypes after specialization.
        - Numerical columns: object → numeric (int/float), coercing errors to NaN
        - Categorical columns: object → category dtype
        This is crucial for XGBoost which requires proper dtypes.
        
        Args:
            df: DataFrame with potentially incorrect dtypes
            
        Returns:
            DataFrame with corrected dtypes
        """
        if df.empty:
            return df
        
        # 1. Handle Numerical Columns Explicitly
        for col_name in self.numerical_columns:
            if col_name in df.columns:
                # Coerce errors to NaN (handles '?', missing, etc.)
                # This ensures the column is numeric, not object
                # Matches DataLoader behavior: errors='coerce'
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        # 2. Handle Categorical Columns
        for column in self.spalten_list:
            # Skip record_id, label, and numerical columns
            if column.name == self.record_id_col or column.name == self.label_col or column.name in self.numerical_columns:
                continue
            
            if column.name not in df.columns:
                continue
            
            # Convert to category dtype
            # FIX: Convert to string FIRST to ensure consistent category dtype across partitions
            # This prevents "dtype of categories must be the same" error when Dask
            # concatenates partitions (each partition may have categories with different
            # underlying dtypes like int64 vs object)
            if not isinstance(df[column.name].dtype, pd.CategoricalDtype):
                df[column.name] = df[column.name].astype(str).astype('category')
            else:
                # Already categorical but may have inconsistent underlying dtype
                df[column.name] = df[column.name].astype(str).astype('category')
        
        return df
    
    def _has_any_generalized_value(self, record: pd.Series) -> bool:
        """
        Quick check if record has ANY generalized values (categorical OR numerical).
        Used for n_duplicates=0 optimization.
        
        Returns:
            True if record has any generalized column (categorical or numerical)
        """
        for column in self.spalten_list:
            # Skip record_id and label columns
            if column.name == self.record_id_col or column.name == self.label_col:
                continue
            
            value = record.get(column.name)
            
            # Check if this column (categorical OR numerical) is generalized
            if value is not None and column.is_generalized(value):
                return True
        
        return False
    
    def _select_column_values(self, column, value, n_values: Optional[int], 
                             selection_mode: str, column_profile: Optional[Dict] = None) -> List:
        """
        Select values for a generalized column using different strategies.
        CRITICAL: Filtering by observed values happens here before selection.
        
        Args:
            column: Column object from spalten_list
            value: Current (potentially generalized) value
            n_values: Max number of values to select (None = all)
            selection_mode: 'all', 'random', or 'imputation'
            column_profile: Column profile for imputation scoring (optional)
            
        Returns:
            List of selected values (filtered and scored/sampled as needed)
        """
        # Get all possible values for this generalized value
        possible_values = column.get_value(value)
        
        # STEP 1: Filter by observed values FIRST (before any selection)
        if self.limit_to_observed_values and self.observed_values_dict and column.name in self.observed_values_dict:
            observed = self.observed_values_dict[column.name]
            # Type-safe filtering: convert both to strings for comparison
            # This handles cases where hierarchy has ints [0, 1] but observed has strings ['0', '1']
            observed_str = {str(v) for v in observed}
            possible_values = [v for v in possible_values if str(v) in observed_str]
        
        # Fallback if no values available after filtering
        if not possible_values:
            return [value]
        
        # STEP 2: Select values based on strategy
        if n_values is None or len(possible_values) <= n_values:
            # Return all values (no selection needed)
            return possible_values
        
        # Need to select n_values from possible_values
        if selection_mode == 'random':
            # Random sampling (deterministic per record)
            rng = random.Random(self.seed + hash(column.name) % 10000)
            return rng.sample(possible_values, n_values)
        
        elif selection_mode == 'imputation' and column_profile:
            # Score-based selection using column profile
            if column_profile.get('type') == 'categorical' and 'frequencies' in column_profile:
                # Sort by frequency (most common first)
                freqs = column_profile['frequencies']
                scored = [(v, freqs.get(v, 0.0)) for v in possible_values]
                scored.sort(key=lambda x: x[1], reverse=True)
                return [v for v, _ in scored[:n_values]]
            else:
                # No scoring available, fall back to random
                rng = random.Random(self.seed + hash(column.name) % 10000)
                return rng.sample(possible_values, n_values)
        
        else:
            # Default: return all or first n_values
            return possible_values[:n_values]
    
    def _generate_combinations_as_dataframe(self, record: pd.Series, n_duplicates: Optional[int] = None,
                              filtering_mode: Optional[str] = None,
                              column_profiles: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate combinations as DataFrame (optimized wrapper).
        
        Uses List[Dict] internally for cleaner Cartesian product logic, then converts once.
        
        Args:
            record: Single record (pd.Series)
            n_duplicates: Target number of variants (None = generate all)
            filtering_mode: 'random', 'imputation', 'knn', or None
            column_profiles: Feature profiles for imputation scoring
            
        Returns:
            DataFrame with variants
        """
        # Generate as List[Dict] using existing logic (Cartesian product)
        variants_list = self._generate_combinations(
            record, n_duplicates, filtering_mode, column_profiles
        )
        
        # Convert to DataFrame once (acceptable overhead, happens once per record)
        return pd.DataFrame(variants_list)
    
    def _generate_combinations(self, record: pd.Series, n_duplicates: Optional[int] = None,
                              filtering_mode: Optional[str] = None,
                              column_profiles: Optional[Dict] = None) -> List[Dict]:
        """
        Unified variant generation with intelligent strategies per filtering mode.
        
        Strategy:
        1. Random mode: Generate exactly n_duplicates variants by sampling one value per column
        2. Imputation mode: Select top-N values per column with column profiles, then create Cartesian product
        3. KNN mode: Generate ALL combinations (row-based scoring, can't pre-select)
        
        Args:
            record: Single record (pd.Series)
            n_duplicates: Target number of variants (None = generate all)
            filtering_mode: 'random', 'imputation', 'knn', or None
            column_profiles: Feature profiles for imputation scoring
            
        Returns:
            List of variant dictionaries
        """
        # STRATEGY 1: Random mode - generate exactly n_duplicates complete variants
        if filtering_mode == 'random' and n_duplicates:
            return self._generate_random_variants(record, n_duplicates)
        
        # STRATEGY 1.5: Imputation with prefilter - generate random variants first, then score
        # This is an optimization for large datasets where scoring all variants is expensive
        if (filtering_mode == 'imputation' and self.imputation_prefilter_size and 
            self.imputation_prefilter_size > 0):
            # Generate random variants first (larger sample than needed)
            random_variants = self._generate_random_variants(record, self.imputation_prefilter_size)
            # Then filter those with imputation scoring to get top N
            # Note: This happens in the caller via _filter_variants_after_generation
            return random_variants
        
        # STRATEGY 2 & 3: Imputation/KNN - use Cartesian product with optional column reduction
        # For imputation: select top-N values per column to reduce combinatorial explosion
        # For KNN: use all values (row-based scoring requires full combinations)
        values_per_column = n_duplicates if (filtering_mode == 'imputation' and n_duplicates) else None
        selection_mode = 'imputation' if (filtering_mode == 'imputation' and column_profiles) else 'all'
        
        # Start with base record
        variants = [record.to_dict()]
        
        # Process each column
        for column in self.spalten_list:
            # Skip record_id and label columns
            if column.name == self.record_id_col or column.name == self.label_col:
                continue
            
            new_variants = []
            
            for variant in variants:
                value = variant.get(column.name)
                
                # Handle numerical columns (mean imputation, no variants)
                if column.name in self.numerical_columns:
                    if value is not None and column.is_generalized(value):
                        intervall = column.get_value(value)
                        if intervall:
                            variant[column.name] = intervall[len(intervall) // 2]
                    new_variants.append(variant)
                    continue
                
                # Handle categorical columns
                if value is None or not column.is_generalized(value):
                    # Original value or missing, no variants needed
                    new_variants.append(variant)
                    continue
                
                # Get column profile for this column (if available)
                col_profile = column_profiles.get(column.name) if column_profiles else None
                
                # Select values using intelligent strategy
                selected_values = self._select_column_values(
                    column, value, values_per_column, selection_mode, col_profile
                )
                
                # Create variant for each selected value
                for val in selected_values:
                    new_variant = variant.copy()
                    new_variant[column.name] = val
                    new_variants.append(new_variant)
            
            variants = new_variants
        
        return variants
    
    def _generate_random_variants(self, record: pd.Series, n_duplicates: int) -> List[Dict]:
        """
        Generate UP TO n_duplicates UNIQUE variants by randomly sampling one value per column.
        If fewer unique combinations exist, returns only the unique ones (no duplicate rows).
        
        Args:
            record: Single record (pd.Series)
            n_duplicates: Maximum number of unique variants to generate
            
        Returns:
            List of up to n_duplicates UNIQUE variant dictionaries
        """
        # Use record-specific seed for deterministic results
        record_seed = self.seed + hash(str(record.get(self.record_id_col, 0))) % 10000
        rng = random.Random(record_seed)
        
        variants = []
        seen_combinations = set()  # Track unique row combinations
        max_attempts = n_duplicates * 42  # Limit attempts to avoid infinite loops
        attempts = 0
        
        # Generate variants until we have n_duplicates unique ones or can't create more
        while len(variants) < n_duplicates and attempts < max_attempts:
            attempts += 1
            
            # Start with full record (includes record_id, label, and all columns)
            variant = record.to_dict()
            
            # Process each column that can be specialized
            for column in self.spalten_list:
                # Skip record_id and label columns (already in variant)
                if column.name == self.record_id_col or column.name == self.label_col:
                    continue
                
                value = record.get(column.name)
                
                # Handle numerical columns (mean imputation)
                if column.name in self.numerical_columns:
                    if value is not None and column.is_generalized(value):
                        intervall = column.get_value(value)
                        if intervall:
                            variant[column.name] = intervall[len(intervall) // 2]
                    # else: keep original value (already in variant)
                    continue
                
                # Handle categorical columns
                if value is None or not column.is_generalized(value):
                    # Keep original value (already in variant)
                    continue
                
                # Get possible values (with observed filtering)
                possible_values = column.get_value(value)
                if self.limit_to_observed_values and self.observed_values_dict and column.name in self.observed_values_dict:
                    observed = self.observed_values_dict[column.name]
                    # Type-safe filtering: convert both to strings for comparison
                    # This handles cases where hierarchy has ints [0, 1] but observed has strings ['0', '1']
                    observed_str = {str(v) for v in observed}
                    possible_values = [v for v in possible_values if str(v) in observed_str]
                
                # Sample one value
                if possible_values:
                    variant[column.name] = rng.choice(possible_values)
                # else: keep original value (already in variant)
            
            # Create hashable representation (excluding record_id and label)
            variant_key = tuple(sorted((k, v) for k, v in variant.items() 
                                      if k != self.record_id_col and k != self.label_col))
            
            # Only add if this combination is new
            if variant_key not in seen_combinations:
                seen_combinations.add(variant_key)
                variants.append(variant)
        
        return variants
    
    def _apply_filtering_to_dataframe(self, variants_df: pd.DataFrame, record_id, n_duplicates: int,
                                      mode: str, column_profiles: Optional[Dict],
                                      knn_model, knn_preprocessor) -> pd.DataFrame:
        """
        Apply filtering to DataFrame.
        
        Args:
            variants_df: DataFrame with variants
            record_id: Current record ID
            n_duplicates: Number of variants to keep
            mode: Filtering mode ('random', 'imputation', 'knn')
            column_profiles: For imputation mode
            knn_model: For KNN mode
            knn_preprocessor: For KNN mode
            
        Returns:
            Filtered DataFrame
        """
        if len(variants_df) <= n_duplicates:
            return variants_df
        
        if mode == 'random' or mode is None:
            return self._filter_random_df(variants_df, n_duplicates, record_id)
        elif mode == 'imputation':
            return self._filter_imputation_df(variants_df, n_duplicates, column_profiles)
        elif mode == 'knn':
            return self._filter_knn_df(variants_df, n_duplicates, knn_model, knn_preprocessor)
        else:
            raise ValueError(f"Unknown filtering mode: {mode}")
    
    def _apply_filtering(self, variants: List[Dict], record_id, n_duplicates: int, 
                        mode: str, column_profiles: Optional[Dict],
                        knn_model, knn_preprocessor) -> List[Dict]:
        """
        Apply filtering to variants using specified mode.
        Uses same logic as FilteringHandler.py for equivalence.
        
        DEPRECATED: Use _apply_filtering_to_dataframe for better performance.
        
        Args:
            variants: List of variant dictionaries
            record_id: Current record ID
            n_duplicates: Number of variants to keep
            mode: Filtering mode ('random', 'imputation', 'knn')
            column_profiles: For imputation mode
            knn_model: For KNN mode
            knn_preprocessor: For KNN mode
            
        Returns:
            Filtered list of variants
        """
        if len(variants) <= n_duplicates:
            return variants
        
        if mode == 'random' or mode is None:
            return self._filter_random(variants, n_duplicates, record_id)
        elif mode == 'imputation':
            return self._filter_imputation(variants, n_duplicates, column_profiles)
        elif mode == 'knn':
            return self._filter_knn(variants, n_duplicates, knn_model, knn_preprocessor)
        else:
            raise ValueError(f"Unknown filtering mode: {mode}")
    
    def _filter_random_df(self, variants_df: pd.DataFrame, n_duplicates: int, record_id) -> pd.DataFrame:
        """
        Random sampling on DataFrame directly (optimized version).
        Avoids List[Dict] conversions for better performance.
        """
        group_seed = self.seed + hash(str(record_id)) % 10000
        return variants_df.sample(n=min(n_duplicates, len(variants_df)), random_state=group_seed)
    
    def _filter_random(self, variants: List[Dict], n_duplicates: int, record_id) -> List[Dict]:
        """
        Random sampling using FilteringHandler logic.
        
        DEPRECATED: Use _filter_random_df for better performance (avoids List[Dict] conversions).
        """
        # Use same seed derivation as FilteringHandler for exact equivalence
        group_seed = self.seed + hash(str(record_id)) % 10000
        
        # Convert to DataFrame and sample (matches FilteringHandler behavior)
        variants_df = pd.DataFrame(variants)
        sampled_df = variants_df.sample(n=min(n_duplicates, len(variants)), random_state=group_seed)
        return sampled_df.to_dict('records')
    
    def _filter_imputation_df(self, variants_df: pd.DataFrame, n_duplicates: int,
                             column_profiles: Optional[Dict]) -> pd.DataFrame:
        """
        Profile-based filtering on DataFrame directly (optimized version).
        Avoids List[Dict] conversions and uses vectorized hashing for better performance.
        """
        if not column_profiles:
            return self._filter_random_df(variants_df, n_duplicates, 0)
        
        from src.FilteringHandler import FilteringHandler
        
        # Convert numeric columns to proper dtype based on profiles
        # Do this BEFORE copying to avoid unnecessary copies
        for col, profile in column_profiles.items():
            if col in variants_df.columns and profile.get('type') == 'numeric':
                variants_df[col] = pd.to_numeric(variants_df[col], errors='coerce')
        
        # Use FilteringHandler method to calculate similarities (vectorized)
        similarity_scores = FilteringHandler._calculate_similarity_score_vectorized(
            variants_df, column_profiles, self.record_id_col
        )
        
        # Add scores (modify in place, then we only copy once at the end)
        variants_df['_score'] = similarity_scores
        
        # Hash-based tie-breaking (optimized: use tuple of all values)
        exclude_cols = {self.record_id_col}
        if self.label_col:
            exclude_cols.add(self.label_col)
        hash_cols = sorted([c for c in variants_df.columns if c not in exclude_cols and c != '_score'])
        
        # Vectorized hashing: convert rows to tuples, then hash
        # Much faster than .apply() row-by-row
        variants_df['_tie_break'] = pd.util.hash_pandas_object(
            variants_df[hash_cols], index=False
        )
        
        # Sort and select
        sorted_df = variants_df.sort_values(
            by=['_score', '_tie_break'],
            ascending=[False, True]
        )
        selected = sorted_df.head(n_duplicates).drop(['_score', '_tie_break'], axis=1)
        
        return selected
    
    def _filter_imputation(self, variants: List[Dict], n_duplicates: int, 
                          column_profiles: Optional[Dict]) -> List[Dict]:
        """
        Profile-based filtering using FilteringHandler._calculate_profile_similarity.
        
        DEPRECATED: Use _filter_imputation_df for better performance (avoids List[Dict] conversions).
        """
        if not column_profiles:
            return self._filter_random(variants, n_duplicates, 0)
        
        from src.FilteringHandler import FilteringHandler
        
        # Convert to DataFrame for FilteringHandler
        variants_df = pd.DataFrame(variants)
        
        # Convert numeric columns to proper dtype based on profiles
        for col, profile in column_profiles.items():
            if col in variants_df.columns and profile.get('type') == 'numeric':
                variants_df[col] = pd.to_numeric(variants_df[col], errors='coerce')
        
        # Use FilteringHandler method to calculate similarities (vectorized)
        # Note: column_profiles already excludes record_id and label (built in _build_column_profiles)
        # The similarity calculation only uses columns present in column_profiles
        similarity_scores = FilteringHandler._calculate_similarity_score_vectorized(
            variants_df, column_profiles, self.record_id_col
        )
        
        # Use same filtering logic as FilteringHandler for exact equivalence
        group_with_scores = variants_df.copy()
        group_with_scores['_score'] = similarity_scores
        
        # Hash-based tie-breaking: deterministic regardless of column order or DataFrame construction
        # Exclude record_id and label from hash to focus on actual variant values  
        # Sort columns alphabetically to ensure same order in both OLD and NEW methods
        exclude_cols = {self.record_id_col}
        if self.label_col:
            exclude_cols.add(self.label_col)
        hash_cols = sorted([c for c in variants_df.columns if c not in exclude_cols])
        group_with_scores['_tie_break'] = variants_df[hash_cols].apply(
            lambda row: hash(tuple(zip(hash_cols, row.values))), axis=1
        )
        
        sorted_group = group_with_scores.sort_values(
            by=['_score', '_tie_break'], 
            ascending=[False, True]
        )
        selected = sorted_group.head(n_duplicates).drop(['_score', '_tie_break'], axis=1)
        
        return selected.to_dict('records')
    
    def _filter_knn_df(self, variants_df: pd.DataFrame, n_duplicates: int,
                      knn_model, knn_preprocessor) -> pd.DataFrame:
        """
        KNN-based filtering on DataFrame directly (optimized version).
        Avoids List[Dict] conversions for better performance.
        """
        if not knn_model or not knn_preprocessor:
            return self._filter_random_df(variants_df, n_duplicates, 0)
        
        try:
            # Get feature columns (exclude record_id and label)
            exclude_cols = {self.record_id_col}
            if self.label_col:
                exclude_cols.add(self.label_col)
            feature_cols = [c for c in variants_df.columns if c not in exclude_cols]
            
            X = variants_df[feature_cols]
            
            # Preprocess
            X_processed = knn_preprocessor.transform(X)
            
            # Get KNN distances
            distances, _ = knn_model.kneighbors(X_processed)
            mean_distances = distances.mean(axis=1)
            
            # Add distance column and select top n (in place modification)
            variants_df['_knn_dist'] = mean_distances
            
            # Select top n by smallest distance
            selected = variants_df.nsmallest(n_duplicates, '_knn_dist').drop('_knn_dist', axis=1)
            return selected
            
        except Exception as e:
            print(f"Warning: KNN filtering failed ({e}), falling back to random")
            return self._filter_random_df(variants_df, n_duplicates, 0)
    
    def _filter_knn(self, variants: List[Dict], n_duplicates: int,
                   knn_model, knn_preprocessor) -> List[Dict]:
        """
        KNN-based filtering using FilteringHandler approach.
        
        DEPRECATED: Use _filter_knn_df for better performance (avoids List[Dict] conversions).
        """
        if not knn_model or not knn_preprocessor:
            return self._filter_random(variants, n_duplicates, 0)
        
        try:
            # Convert variants to DataFrame
            variants_df = pd.DataFrame(variants)
            
            # Get feature columns (exclude record_id and label)
            exclude_cols = {self.record_id_col}
            if self.label_col:
                exclude_cols.add(self.label_col)
            feature_cols = [c for c in variants_df.columns if c not in exclude_cols]
            
            X = variants_df[feature_cols]
            
            # Preprocess
            X_processed = knn_preprocessor.transform(X)
            
            # Get KNN distances
            distances, _ = knn_model.kneighbors(X_processed)
            mean_distances = distances.mean(axis=1)
            
            # Take variants with smallest distances
            indices = np.argsort(mean_distances)[:n_duplicates]
            return [variants[i] for i in sorted(indices)]
            
        except Exception as e:
            print(f"Warning: KNN filtering failed ({e}), falling back to random")
            return self._filter_random(variants, n_duplicates, 0)
    
    def _build_column_profiles(self, df: pd.DataFrame) -> Dict:
        """Build column profiles from reference data using FilteringHandler."""
        from src.FilteringHandler import FilteringHandler
        
        # Extract original rows (non-generalized values)
        feature_cols = [c for c in df.columns if c != self.record_id_col and c != self.label_col]
        
        # Build spalten_dict from spalten_list
        spalten_dict = {col.name: col for col in self.spalten_list}
        
        # Use FilteringHandler's methods for consistency
        try:
            # _extract_original_rows takes 4 args: (df, columns, record_id_column, spalten_dict)
            original_rows = FilteringHandler._extract_original_rows(
                df, feature_cols, self.record_id_col, spalten_dict
            )
            
            # _build_profiles_from_dataframe takes 3 args: (df, columns, record_id_column)
            # This ensures we only profile the specified feature columns, not label or other columns
            profiles = FilteringHandler._build_profiles_from_dataframe(
                original_rows, feature_cols, self.record_id_col
            )
            
            return profiles
        except Exception as e:
            print(f"Warning: Could not build profiles ({e})")
            return {}
    
    def _build_knn_model(self, df: pd.DataFrame) -> Tuple:
        """Build KNN model from reference data using FilteringHandler."""
        from src.FilteringHandler import FilteringHandler
        
        feature_cols = [c for c in df.columns if c != self.record_id_col and c != self.label_col]
        
        # Build spalten_dict from spalten_list
        spalten_dict = {col.name: col for col in self.spalten_list}
        
        try:
            # _extract_original_rows takes 4 args: (df, columns, record_id_column, spalten_dict)
            original_rows = FilteringHandler._extract_original_rows(
                df, feature_cols, self.record_id_col, spalten_dict
            )
            
            # _build_knn_model takes 3 args: (original_rows, columns, record_id_column)
            # It returns (knn_model, preprocessor) tuple
            knn_model, preprocessor = FilteringHandler._build_knn_model(
                original_rows, feature_cols, self.record_id_col
            )
            
            return knn_model, preprocessor
        except Exception as e:
            print(f"Warning: Could not build KNN model ({e})")
            return None, None
