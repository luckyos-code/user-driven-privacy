import numpy as np
import random
import pandas as pd
import dask
import os
import dask.dataframe as dd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.dask_utils import count_dask_rows

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class FilteringHandler:
    """
    Handles filtering of specialized data entries for data analysis.
    
    This class provides functionality to process datasets with duplicates,
    offering various strategies to select which duplicates to keep.
    """
    
    @staticmethod
    def filter_specialized_data(data_train, record_id_column, filtering_configs, random_seed=42, cache_dir=None, redo_cache=False, original_reference_data=None, per_record_counts=None, spalten_dict=None, dataset_name=None, label_column=None):
        """
        Process data with different filtering configurations in one pass.
        Always uses chunked processing (small datasets = one chunk) to avoid code duplication.
        
        Args:
            data_train: Dask DataFrame with data
            record_id_column: Column containing record IDs
            filtering_configs: List of (n_duplicates, mode) tuples to process
            random_seed: Seed for random number generators
            cache_dir: Directory to cache filtered datasets (optional)
            redo_cache: If True, force recomputation even if cache exists
            original_reference_data: Pandas DataFrame with generalized reference data (pre-specialization) for profile building
            per_record_counts: Dict mapping record_id -> variant count for chunked processing (REQUIRED)
            spalten_dict: Dict mapping column names to spalten classes (for detecting generalized values)
            dataset_name: Name of the dataset (for spalten lookup if spalten_dict not provided)
            label_column: Name of the label column to exclude from feature processing (optional)
            
        Returns:
            Dictionary mapping (n_duplicates, mode) to filtered DataFrames
        """
        # Check cache first
        if cache_dir and not redo_cache:
            cached_results = {}
            all_cached = True
            for n_duplicates, mode in filtering_configs:
                cache_path = os.path.join(cache_dir, f"{n_duplicates}_{mode or 'none'}.parquet")
                if os.path.exists(cache_path):
                    cached_results[(n_duplicates, mode)] = dd.read_parquet(cache_path)
                else:
                    all_cached = False
                    break
            if all_cached:
                print("Loaded all filtered datasets from cache")
                return cached_results
        
        # Set seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Repartition to reduce task graph complexity for large datasets
        # Reduce to ~100-200 larger partitions before any operations
        current_npartitions = data_train.npartitions
        if current_npartitions > 200:
            print(f"Repartitioning from {current_npartitions} to 200 partitions to reduce scheduler overhead...")
            data_train = data_train.repartition(npartitions=200)

        # Analyze record distribution if dataset is not too large
        #if len(data_train) < 128298198:
        #    FilteringHandler.analyze_record_id_distribution(data_train, record_id_column)
        
        # Always use chunked processing - small datasets will just have one chunk
        if per_record_counts is not None:
            total_upper_bound = sum(per_record_counts.values())
            print(f"Dataset size: {total_upper_bound:,} upper bound rows. Using chunked processing.")
        else:
            raise ValueError("per_record_counts is required for filter_specialized_data")

        # Find all unique modes needed across configurations
        modes_needed = set(mode for _, mode in filtering_configs if mode is not None)
        
        # Use upper bound as initial count to avoid expensive computation on massive datasets
        initial_count = total_upper_bound
        print(f"Using upper bound {initial_count:,} as initial count (avoids computing exact count on large dataset)")
        
        # Prepare result dictionary
        results = {}
        
        # Build resources from original reference data if needed
        column_profiles = None
        knn_model = None
        knn_preprocessor = None
        
        if 'imputation' in modes_needed or 'knn' in modes_needed:
            if original_reference_data is None:
                raise ValueError("original_reference_data is required for imputation/knn modes")
            
            # Extract feature columns, excluding record_id AND label column
            exclude_columns = {record_id_column}
            if label_column:
                exclude_columns.add(label_column)
            feature_columns = [col for col in data_train.columns if col not in exclude_columns]
            
            # Replace generalized and missing values with NaN (keep rows with partial original data)
            print("Processing reference data: replacing generalized/missing values with NaN...")
            original_rows = FilteringHandler._extract_original_rows(
                original_reference_data, feature_columns, record_id_column, spalten_dict
            )
            
            print(f"Prepared {len(original_rows):,} reference rows (original values + NaN markers)")
            
            # Build profiles for imputation mode
            if 'imputation' in modes_needed:
                print("Building column profiles from original rows...")
                column_profiles = FilteringHandler._build_profiles_from_dataframe(
                    original_rows, feature_columns, record_id_column
                )
                if not column_profiles:
                    raise ValueError("Empty profiles generated")
            
            # Build KNN model for knn mode
            if 'knn' in modes_needed:
                print("Building KNN model from original rows...")
                knn_model, knn_preprocessor = FilteringHandler._build_knn_model(
                    original_rows, feature_columns, record_id_column
                )
                if knn_model is None:
                    raise ValueError("Failed to build KNN model")
        
        # Now create the specific filtered datasets for each configuration
        for n_duplicates, mode in filtering_configs:
            print(f"n_duplicates: {n_duplicates}, mode: {mode}")
            
            # Always use chunked processing (small datasets = one chunk)
            filtered_result = FilteringHandler._process_data_chunked(
                data_train, record_id_column, per_record_counts, n_duplicates, 
                mode, column_profiles, knn_model, knn_preprocessor, random_seed
            )
            
            # Repartition to reduce partition count after filtering removes most data
            # Target ~100MB per partition for good balance
            # Only repartition if it's a Dask DataFrame (chunked processing always returns Dask)
            if isinstance(filtered_result, dd.DataFrame):
                filtered_result = filtered_result.repartition(partition_size="100MB")
            
            results[(n_duplicates, mode)] = filtered_result.persist() if isinstance(filtered_result, dd.DataFrame) else filtered_result
            
            # Use efficient counting method
            final_count = count_dask_rows(results[(n_duplicates, mode)])
            removed_count = initial_count - final_count
            
            # Calculate percentage change
            if initial_count > 0:
                percentage_reduction = (removed_count / initial_count) * 100
            else:
                percentage_reduction = 0
            
            print(f"Final number of rows: {final_count}")
            print(f"Removed {removed_count} rows ({percentage_reduction:.2f}%)")
            
        # Save results to cache if cache_dir provided
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            for key, value in results.items():
                cache_path = os.path.join(cache_dir, f"{key[0]}_{key[1] or 'none'}.parquet")
                if not os.path.exists(cache_path):
                    value.to_parquet(cache_path, compression='zstd')
                    print(f"Saved filtered dataset to cache: {cache_path}")
        
        return results

    @staticmethod
    def analyze_record_id_distribution(ddf, record_id_column='record_id'):
        """
        Analyze the distribution of record_id groups in a Dask DataFrame.
        
        Parameters:
        -----------
        ddf : dask.dataframe.DataFrame
            Input Dask DataFrame
        record_id_column : str, default='record_id'
            Column to analyze distribution for
        
        Returns:
        --------
        pandas.Series
            Distribution of record_id group sizes
        """
        # Count occurrences of each record_id
        id_counts = ddf.groupby(record_id_column).size().compute()
        
        # Get distribution of group sizes
        distribution = id_counts.value_counts().sort_index()
        
        print("Record ID Distribution:")
        print("----------------------")
        print("Group Size | Count | Percentage")
        print("----------------------------------")
        
        total_ids = len(id_counts)
        for size, count in distribution.items():
            percentage = (count / total_ids) * 100
            print(f"{size:^10} | {count:^5} | {percentage:.2f}%")
        
        print("----------------------------------")
        print(f"Total unique record_ids: {total_ids}")
        
        return distribution

    @staticmethod
    def _split_by_duplicate_status(ddf, record_id_column):
        """Split dataframe into records with and without duplicates."""
        counts = ddf.groupby(record_id_column).size().reset_index().persist()
        counts.columns = [record_id_column, 'count']
        
        # Get record_ids that appear exactly once (non-duplicates)
        single_records = counts[counts['count'] == 1]
        multi_records = counts[counts['count'] > 1]
        
        # Filter original dataframe
        non_duplicates = ddf.merge(single_records[[record_id_column]], on=record_id_column)
        duplicates = ddf.merge(multi_records[[record_id_column]], on=record_id_column)
        
        print(f"Non-duplicate entries: {non_duplicates.shape[0].compute()}")
        print(f"Entries with duplicates: {duplicates.shape[0].compute()}")
        
        return non_duplicates, duplicates

    @staticmethod
    def _process_duplicates_with_metadata(df, record_id_column, seed=42):
        """
        Process all duplicates at once, keeping them grouped by record_id.
        This allows for extracting different subsets later without reprocessing.
        """
        # Group data by record ID and retain the full structure
        grouped_data = df.groupby(record_id_column)
        
        # Create a list to store processed groups
        processed_groups = []
        
        # Process each group and add metadata
        for record_id, group in grouped_data:
            # Create a seed that's unique for each record_id but reproducible
            group_seed = seed + hash(str(record_id)) % 10000
            group_rng = np.random.RandomState(group_seed)
            
            # Add the record_id as metadata to make reassembly easier
            group_with_meta = group.copy()
            
            # Add a random ordering for random selection, using the seeded RNG
            group_with_meta['_random_order'] = group_rng.permutation(len(group_with_meta))
            
            processed_groups.append(group_with_meta)
        
        # Combine all processed groups
        if processed_groups:
            result = pd.concat(processed_groups)
            return result
        else:
            return df.copy()  # Return empty dataframe with same schema

    @staticmethod
    def _extract_random_duplicates(df_with_metadata, record_id_column, n_duplicates, seed=42):
        """
        Extract n random duplicates per record_id from pre-processed data.
        """
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Group by record_id
        result = df_with_metadata.groupby(record_id_column).apply(
            lambda group: group.nsmallest(min(n_duplicates, len(group)), '_random_order')
        ).reset_index(drop=True)
        
        # Remove metadata columns
        result = result.drop('_random_order', axis=1)
        
        return result

    @staticmethod
    def _score_and_extract_duplicates(df_with_metadata, record_id_column, profiles, n_duplicates, seed=42):
        """
        Score duplicates by similarity to profiles and extract top n using vectorized operations.
        """
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Process each group using the vectorized scoring function
        result_groups = []
        
        for record_id, group in df_with_metadata.groupby(record_id_column):
            # Calculate similarity scores for the entire group at once
            similarity_scores = FilteringHandler._calculate_similarity_score_vectorized(group, profiles, record_id_column)
            
            # Add scores to group
            group_copy = group.copy()
            group_copy['_similarity_score'] = similarity_scores
            
            # Get top n by score
            top_n = group_copy.nlargest(min(n_duplicates, len(group)), '_similarity_score')
            result_groups.append(top_n)
        
        # Combine all processed groups
        if result_groups:
            result = pd.concat(result_groups)
            # Remove metadata columns
            result = result.drop(['_random_order', '_similarity_score'], axis=1, errors='ignore')
            return result
        else:
            # Return empty dataframe with same schema minus metadata columns
            return df_with_metadata.drop(['_random_order'], axis=1, errors='ignore').head(0)

    @staticmethod
    def _calculate_similarity_score_vectorized(df, feature_profiles, record_id_column):
        """
        Vectorized implementation of similarity score calculation across all rows at once.
        Much faster than row-by-row processing for large datasets.
        """
        # Initialize scores array
        scores = np.zeros(len(df))
        feature_count = len(feature_profiles)
        
        # Process each feature column
        for col, profile in feature_profiles.items():
            if col == record_id_column:
                feature_count -= 1
                continue
                
            # Skip if profile structure is invalid
            if not isinstance(profile, dict) or 'type' not in profile:
                feature_count -= 1
                continue
                
            if profile['type'] == 'numeric':
                # For numeric features: vectorized z-score calculation
                if 'std' in profile and 'mean' in profile and profile['std'] > 0:
                    # Calculate z-scores for entire column at once
                    z_scores = np.abs((df[col].values - profile['mean']) / profile['std'])
                    # Convert z-scores to similarities (closer to reference = higher score)
                    similarities = np.maximum(0, 1 - np.minimum(z_scores / 3, 1))
                    # Add to total scores, handling NaN values
                    mask = ~np.isnan(similarities)
                    scores[mask] += similarities[mask]
            else:
                # For categorical features: use frequency of each value
                if 'frequencies' in profile:
                    # Map each value to its frequency
                    value_freqs = df[col].map(profile['frequencies'])
                    
                    # Check if value_freqs is a Categorical Series and convert to float if needed
                    if pd.api.types.is_categorical_dtype(value_freqs):
                        value_freqs = value_freqs.astype(float)
                    
                    # Replace NaN with small non-zero value
                    value_freqs = value_freqs.fillna(0.1)
                    scores += value_freqs.values
        
        # Normalize scores by number of features
        return scores / feature_count if feature_count > 0 else np.zeros(len(df))

    @staticmethod
    def _build_feature_profiles(non_dup_df, record_id_column):
        """Build feature profiles from non-duplicate entries."""
        profiles = {}
        
        for col in non_dup_df.columns:
            if col == record_id_column:
                continue
                
            col_data = non_dup_df[col]
            
            # Check column type
            if pd.api.types.is_numeric_dtype(col_data):
                # For numeric columns, store mean and std
                profiles[col] = {
                    'type': 'numeric',
                    'mean': col_data.mean(),
                    'std': col_data.std() if col_data.std() > 0 else 1.0  # Avoid division by zero
                }
            else:
                # For categorical/text, store value frequencies
                value_counts = col_data.value_counts(normalize=True)
                profiles[col] = {
                    'type': 'categorical',
                    'frequencies': value_counts.to_dict()
                }
                
        return profiles
    
    @staticmethod
    def _extract_original_rows(original_df, columns, record_id_column, spalten_dict=None):
        """
        Replace generalized and missing ('?') values with NaN to keep rows with partial original data.
        
        This preserves rows that have a mix of original, missing, and generalized values.
        Only original values remain; everything else becomes NaN.
        
        Args:
            original_df: Pandas DataFrame with generalized reference data (pre-specialization)
            columns: List of column names to process
            record_id_column: Column name for record ID (excluded from processing)
            spalten_dict: Dict mapping column names to spalten classes (for detecting generalized values)
            
        Returns:
            Pandas DataFrame where generalized and missing values are replaced with NaN
        """
        df = original_df.copy()
        
        # Replace generalized and missing values with NaN in each column
        for col in columns:
            if col == record_id_column or col not in df.columns:
                continue
            
            def convert_value(val):
                # Check if missing
                if pd.isna(val) or val == '?' or val == '':
                    return np.nan
                
                # Check if generalized using spalten classes
                if spalten_dict and col in spalten_dict:
                    spalten_class = spalten_dict[col]
                    if hasattr(spalten_class, 'is_generalized') and spalten_class.is_generalized(str(val)):
                        return np.nan
                
                # Otherwise keep original value
                return val
            
            df[col] = df[col].apply(convert_value)
        
        # Ensure proper dtypes: convert numeric columns that became 'object' back to numeric
        # This is crucial for sklearn preprocessing (StandardScaler requires numeric dtype)
        for col in columns:
            if col == record_id_column or col not in df.columns:
                continue
            # Try to convert to numeric, errors='ignore' keeps non-numeric as-is
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        # Return DataFrame with generalized/missing values replaced by NaN
        # Original values stay as is
        return df
    
    @staticmethod
    def _build_profiles_from_dataframe(df, columns, record_id_column):
        """
        Build column profiles from a DataFrame, excluding NaN values (column-wise filtering).
        
        Uses only original values for computing statistics, ignoring missing values.
        
        Args:
            df: Pandas DataFrame with NaN for missing/generalized values
            columns: List of column names to build profiles for
            record_id_column: Column name for record ID (excluded from profiles)
            
        Returns:
            dict: Column profiles with statistics for each column (computed from non-missing values only)
        """
        profiles = {}
        
        for col in columns:
            if col == record_id_column or col not in df.columns:
                continue
            
            # Filter out NaN values for this column (column-wise filtering)
            col_data = df[col]
            original_values = col_data.dropna()
            
            if len(original_values) == 0:
                # No original values in this column - skip
                continue
            
            # Try to detect if numeric
            try:
                numeric_values = pd.to_numeric(original_values, errors='coerce')
                # If >50% are valid numbers, treat as numeric
                if numeric_values.notna().sum() / len(original_values) > 0.5:
                    numeric_values = numeric_values.dropna()
                    if len(numeric_values) > 0:
                        std_val = float(numeric_values.std())
                        profiles[col] = {
                            'type': 'numeric',
                            'mean': float(numeric_values.mean()),
                            'std': std_val if std_val > 0 else 1.0
                        }
                else:
                    # Categorical - NaN already excluded by dropna()
                    value_counts = original_values.value_counts(normalize=True)
                    profiles[col] = {
                        'type': 'categorical',
                        'frequencies': value_counts.to_dict()
                    }
            except:
                # Fallback to categorical
                value_counts = original_values.value_counts(normalize=True)
                profiles[col] = {
                    'type': 'categorical',
                    'frequencies': value_counts.to_dict()
                }
        
        return profiles
    
    @staticmethod
    def _build_knn_model(original_rows, columns, record_id_column):
        """
        Build sklearn KNN model from rows with original values.
        
        Optionally filters out all-NaN rows (row-wise filtering) since they provide
        no useful information for distance calculation.
        
        Args:
            original_rows: Pandas DataFrame with NaN for missing/generalized values
            columns: List of feature column names
            record_id_column: Column name for record ID (excluded from features)
            
        Returns:
            tuple: (knn_model, preprocessor) or (None, None) if failed
        """
        feature_cols = [col for col in columns if col != record_id_column]
        
        # Optional: Remove rows that are ALL NaN (row-wise filtering)
        # These rows have no useful information for distance calculation
        # Keep rows with at least one non-NaN value
        filtered_rows = original_rows.dropna(subset=feature_cols, how='all')
        
        if len(filtered_rows) == 0:
            print("Warning: No rows with at least one original value found for KNN")
            return None, None
        
        print(f"Filtered out {len(original_rows) - len(filtered_rows):,} all-NaN rows for KNN")
        
        reference_features = filtered_rows[feature_cols]
        
        # Create preprocessor (will handle NaN appropriately for numeric/categorical)
        preprocessor, _, _ = FilteringHandler._create_preprocessor(reference_features)
        
        # Transform reference features
        transformed_features = preprocessor.transform(reference_features)
        
        # Build KNN model
        knn = NearestNeighbors(algorithm='ball_tree', n_neighbors=min(10, len(filtered_rows)))
        knn.fit(transformed_features)
        
        print(f"KNN model built with {len(filtered_rows):,} reference points")
        
        return knn, preprocessor

    @staticmethod
    def _create_preprocessor(reference_features):
        """
        Create a consistent preprocessor for feature transformation.
        Includes imputation to handle NaN values that may exist in the reference data.
        
        Args:
            reference_features: DataFrame with reference features (may contain NaN)
            
        Returns:
            tuple: (preprocessor, numeric_cols, cat_cols)
        """
        # Identify numeric and categorical columns
        numeric_cols = reference_features.select_dtypes(include=['number']).columns.tolist()
        cat_cols = reference_features.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipeline with imputation
        # For numeric: impute with mean, then scale
        # For categorical: impute with most frequent, then one-hot encode
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, cat_cols)
            ], remainder='drop')
        
        # Fit the preprocessor on reference features
        preprocessor.fit(reference_features)
        
        return preprocessor, numeric_cols, cat_cols
    
    @staticmethod
    def _process_with_knn(duplicate_data, record_id_column, reference_features, feature_columns, n_duplicates, random_seed=42):
        """
        Process duplicates using K-Nearest Neighbors for similarity calculation.
        
        Args:
            duplicate_data: Dask DataFrame containing duplicate records with metadata
            record_id_column: Column name for record ID
            reference_features: DataFrame with reference features from non-duplicate records
            feature_columns: List of feature column names
            n_duplicates: Number of duplicates to keep per record_id
            random_seed: Seed for random number generator
            
        Returns:
            Dask DataFrame with selected duplicates
        """
        # Set seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        print(f"Processing with KNN mode (n_duplicates={n_duplicates})")
    
        # Create preprocessor
        preprocessor, _, _ = FilteringHandler._create_preprocessor(reference_features)
        
        # Transform reference features
        print("Preprocessing reference data")
        transformed_reference_features = preprocessor.transform(reference_features)
        
        # Build KNN model from transformed reference data
        print("Building KNN model from reference data")
        knn = NearestNeighbors(algorithm='ball_tree', n_neighbors=min(10, len(reference_features)))
        knn.fit(transformed_reference_features)
        
        # Define a function to process each partition
        def process_partition(df_part):
            if len(df_part) == 0:
                return df_part
                
            # Group by record_id
            groups = []
            for record_id, group in df_part.groupby(record_id_column):
                if len(group) <= n_duplicates:
                    # If we have fewer rows than needed, keep all
                    groups.append(group)
                    continue
                    
                # Extract features
                group_features = group[feature_columns]
                
                # Transform group features using the same preprocessor
                try:
                    transformed_group_features = preprocessor.transform(group_features)
                    
                    # Find distances to reference data
                    distances, _ = knn.kneighbors(transformed_group_features)
                    
                    # Average distance to k nearest neighbors
                    avg_distances = np.mean(distances, axis=1)
                    
                    # Create a copy with distances
                    group_with_dist = group.copy()
                    group_with_dist['_knn_distance'] = avg_distances
                    
                    # Keep instances with smallest average distance (closest to reference distribution)
                    top_matches = group_with_dist.nsmallest(n_duplicates, '_knn_distance')
                    groups.append(top_matches.drop('_knn_distance', axis=1))
                except Exception as e:
                    # Fall back to random selection if transformation fails
                    print(f"Error in KNN processing for record {record_id}: {e}")
                    sampled = group.sample(n=min(n_duplicates, len(group)), random_state=random_seed)
                    groups.append(sampled)
            
            if groups:
                return pd.concat(groups)
            else:
                return df_part.head(0)  # Empty frame with same schema
        
        # Process each partition and collect results
        result = duplicate_data.map_partitions(process_partition).persist()
        
        # Remove metadata column
        result = result.drop('_random_order', axis=1, errors='ignore')
        
        return result
    
    @staticmethod
    def _process_data_chunked(data, record_id_column, per_record_counts, n_duplicates, mode, column_profiles, knn_model, knn_preprocessor, random_seed):
        """
        Process data in chunks to avoid scheduler explosion.
        Handles all cases: n_duplicates=0 (non-duplicates) and n_duplicates>0 (filtered duplicates).
        For small datasets, creates just one chunk.
        
        Args:
            data: Dask DataFrame with all data
            record_id_column: Column name for record ID
            per_record_counts: Dict mapping record_id -> row count
            n_duplicates: Number of duplicates to keep (0 = keep only non-duplicates)
            mode: Filtering mode ('random', 'imputation', 'knn', or None)
            column_profiles: Column profiles for imputation mode (statistical similarity)
            knn_model: Sklearn KNN model for knn mode (true distance-based KNN)
            knn_preprocessor: Sklearn preprocessor for knn mode (feature transformation)
            random_seed: Random seed for reproducibility
            
        Returns:
            Dask DataFrame with filtered data
        """
        if per_record_counts is None:
            raise ValueError("per_record_counts is required for chunked processing")
        
        # Determine which records to process based on n_duplicates
        if n_duplicates == 0:
            # Only keep records with count = 1 (no variants, truly non-duplicate)
            records_to_process = {rid: count for rid, count in per_record_counts.items() if count == 1}
            print(f"Extracting {len(records_to_process):,} non-duplicate records...")
            
            # For n_duplicates=0, we can directly filter without processing
            # These records have count=1, so we just need to extract them (no groupby/selection needed)
            # This avoids iterating through rows
            if records_to_process:
                # Use efficient merge-based filtering
                ids_df = dd.from_pandas(
                    pd.DataFrame({record_id_column: list(records_to_process.keys())}),
                    npartitions=max(1, len(records_to_process) // 10000)  # 10K ids per partition
                )
                result = data.merge(ids_df, on=record_id_column, how='inner')
                print(f"Extracted {len(records_to_process):,} non-duplicate records using direct merge (fast path)")
                return result
            else:
                print(f"Warning: No non-duplicate records found.")
                return data.head(0)  # Empty dataframe with same schema
        else:
            # Process all records (will filter each group to n_duplicates)
            records_to_process = per_record_counts
        
        if not records_to_process:
            print(f"Warning: No records match criteria (n_duplicates={n_duplicates}).")
            return data.head(0)  # Empty dataframe with same schema
        
        # For large datasets, use much smaller chunks
        # to avoid memory overflow and enable incremental processing
        # Target: Process ~1000-5000 record_ids at a time
        total_rows_estimate = sum(records_to_process.values())
        if total_rows_estimate > 1_000_000_000:  # > 1 billion rows
            # Very aggressive chunking for massive datasets
            target_rows_per_chunk = 50_000_000  # 50M rows per chunk
            print(f"MASSIVE dataset detected ({total_rows_estimate:,} rows). Using aggressive chunking (50M rows/chunk)")
        elif total_rows_estimate > 100_000_000:  # > 100M rows
            target_rows_per_chunk = 200_000_000  # 200M rows per chunk
            print(f"Large dataset detected ({total_rows_estimate:,} rows). Using smaller chunks (200M rows/chunk)")
        else:
            target_rows_per_chunk = 500_000_000  # 500M rows per chunk (original)
        
        # Create balanced chunks using bin-packing
        chunks = FilteringHandler._create_balanced_chunks(records_to_process, target_rows_per_chunk=target_rows_per_chunk)
        
        print(f"Processing {len(chunks)} chunk(s)...")
        chunk_results = []
        
        for i, record_ids_in_chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"  Processing chunk {i+1}/{len(chunks)} ({len(record_ids_in_chunk):,} record_ids)...")
            
            # For large datasets, create a Dask Series for efficient joining
            # This avoids memory issues and task graph explosion from isin() or map()
            if len(record_ids_in_chunk) > 100000:
                # Create a Dask DataFrame with the record_ids to filter
                chunk_ids_df = dd.from_pandas(
                    pd.DataFrame({record_id_column: list(record_ids_in_chunk)}),
                    npartitions=max(1, len(record_ids_in_chunk) // 1000000)  # 1M ids per partition
                )
                # Use merge/join which is optimized for large-scale distributed operations
                chunk_data = data.merge(chunk_ids_df, on=record_id_column, how='inner')
            else:
                # For smaller chunks, isin() is fine and faster
                chunk_data = data[data[record_id_column].isin(record_ids_in_chunk)]
            
            # Process chunk with same logic for all modes
            chunk_result = chunk_data.map_partitions(
                lambda df: FilteringHandler._process_partition_with_filtering(
                    df, record_id_column, n_duplicates, mode, column_profiles, knn_model, knn_preprocessor, random_seed
                )
            )
            
            # For massive datasets with many chunks, write intermediate results to avoid memory overflow
            if len(chunks) > 20:
                # Persist and compute chunk to free up scheduler
                chunk_result = chunk_result.persist()
                # Let scheduler process this chunk
                _ = len(chunk_result)  # Force computation
            else:
                # For smaller number of chunks, just persist (normal behavior)
                chunk_result = chunk_result.persist()
            
            chunk_results.append(chunk_result)
        
        # Combine all chunk results
        if len(chunk_results) == 1:
            return chunk_results[0]
        else:
            print("Combining chunk results...")
            return dd.concat(chunk_results)
    
    @staticmethod
    def _process_partition_with_filtering(df, record_id_column, n_duplicates, mode, column_profiles, knn_model, knn_preprocessor, random_seed):
        """
        Process a single partition with filtering logic.
        This is the core filtering logic used by chunked processing.
        
        Args:
            df: Pandas DataFrame partition
            record_id_column: Column name for record ID
            n_duplicates: Number of duplicates to keep (0 = keep only non-duplicates)
            mode: Filtering mode ('random', 'imputation', 'knn', or None)
            column_profiles: Column profiles for imputation mode (dict with feature columns only, excludes label/record_id)
            knn_model: Sklearn KNN model for knn mode
            knn_preprocessor: Sklearn preprocessor for knn mode
            random_seed: Random seed for reproducibility
            
        Returns:
            Pandas DataFrame with filtered data
        """
        if len(df) == 0:
            return df
        
        # Group by record_id
        result_groups = []
        
        for record_id, group in df.groupby(record_id_column):
            group_size = len(group)
            
            # Handle based on n_duplicates
            if n_duplicates == 0:
                # For n_duplicates=0, we already filtered by per_record_counts[rid]==1
                # So just keep all groups (they should all be size 1)
                result_groups.append(group)
            else:
                # Keep up to n_duplicates
                if group_size <= n_duplicates:
                    # Keep all
                    result_groups.append(group)
                else:
                    # Filter to n_duplicates based on mode
                    if mode == 'random':
                        # Random selection
                        group_seed = random_seed + hash(str(record_id)) % 10000
                        selected = group.sample(n=n_duplicates, random_state=group_seed)
                        result_groups.append(selected)
                    elif mode == 'imputation':
                        # Profile-based similarity selection
                        scores = FilteringHandler._calculate_similarity_score_vectorized(
                            group, column_profiles, record_id_column
                        )
                        group_with_scores = group.copy()
                        group_with_scores['_score'] = scores
                        
                        # DEBUG: Print scores for first record
                        print(f"NEW DEBUG CHECK: record_id={record_id}, type={type(record_id)}", flush=True)
                        if int(record_id) == 11186:
                            print(f"\n=== DEBUG NEW METHOD: Record {record_id} ===", flush=True)
                            print(f"Group size: {len(group)}", flush=True)
                            print(f"Scores: min={scores.min():.4f}, max={scores.max():.4f}, unique={len(set(scores))}", flush=True)
                            group_display = group.copy()
                            group_display['_score'] = scores
                            print(group_display[[c for c in group.columns if c != record_id_column] + ['_score']].sort_values('_score', ascending=False).head(10), flush=True)
                        
                        # Sort by score desc, then by all columns for deterministic tie-breaking
                        sorted_group = group_with_scores.sort_values(
                            by=['_score'] + list(group.columns), 
                            ascending=[False] + [True] * len(group.columns)
                        )
                        selected = sorted_group.head(n_duplicates).drop('_score', axis=1)
                        result_groups.append(selected)
                    elif mode == 'knn':
                        # KNN using sklearn model
                        # CRITICAL: Use same feature columns as were used to build the preprocessor
                        # (excludes record_id AND label column to prevent data leakage)
                        if column_profiles is not None:
                            # Get feature columns from profiles (which were built excluding label)
                            feature_cols = [col for col in column_profiles.keys() if col in group.columns]
                        else:
                            # Fallback: exclude only record_id (shouldn't happen in practice)
                            feature_cols = [col for col in group.columns if col != record_id_column]
                        group_features = group[feature_cols]
                        
                        try:
                            # Transform features using the same preprocessor
                            transformed_features = knn_preprocessor.transform(group_features)
                            
                            # Find distances to reference data
                            distances, _ = knn_model.kneighbors(transformed_features)
                            
                            # Average distance to k nearest neighbors
                            avg_distances = np.mean(distances, axis=1)
                            
                            # Add distances to group
                            group_with_dist = group.copy()
                            group_with_dist['_knn_distance'] = avg_distances
                            
                            # Keep instances with smallest average distance (closest to reference)
                            selected = group_with_dist.nsmallest(n_duplicates, '_knn_distance').drop('_knn_distance', axis=1)
                            result_groups.append(selected)
                        except Exception as e:
                            # Fallback to random selection
                            print(f"KNN error for record {record_id}: {e}, using random selection")
                            group_seed = random_seed + hash(str(record_id)) % 10000
                            selected = group.sample(n=n_duplicates, random_state=group_seed)
                            result_groups.append(selected)
                    else:
                        # No mode or unknown mode - keep first n
                        result_groups.append(group.head(n_duplicates))
        
        # Combine all selected groups
        if result_groups:
            return pd.concat(result_groups)
        else:
            return df.head(0)  # Empty dataframe with same schema
    
    @staticmethod
    def _create_balanced_chunks(per_record_counts, target_rows_per_chunk=500_000_000):
        """
        Create balanced chunks using bin-packing algorithm.
        
        Groups record_ids into chunks such that each chunk has approximately
        target_rows_per_chunk rows (upper bound).
        
        Args:
            per_record_counts: Dict mapping record_id -> row count
            target_rows_per_chunk: Target number of rows per chunk
            
        Returns:
            List of lists, where each inner list contains record_ids for one chunk
        """
        # Sort record_ids by count (descending) for better bin-packing
        sorted_records = sorted(per_record_counts.items(), key=lambda x: x[1], reverse=True)
        
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for record_id, count in sorted_records:
            if current_chunk_size + count > target_rows_per_chunk and current_chunk:
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = [record_id]
                current_chunk_size = count
            else:
                # Add to current chunk
                current_chunk.append(record_id)
                current_chunk_size += count
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Print statistics
        if len(chunks) > 1:
            chunk_sizes = [sum(per_record_counts[rid] for rid in chunk) for chunk in chunks]
            print(f"Created {len(chunks)} chunks:")
            print(f"  Avg size: {sum(chunk_sizes) / len(chunk_sizes):,.0f} rows")
            print(f"  Min size: {min(chunk_sizes):,} rows")
            print(f"  Max size: {max(chunk_sizes):,} rows")
        
        return chunks

