import numpy as np
import random
import pandas as pd
import dask
import os
import dask.dataframe as dd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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
    def filter_specialized_data(data_train, record_id_column, filtering_configs, random_seed=42, cache_dir=None, redo_cache=False):
        """
        Process data with different filtering configurations in one pass.
        
        Args:
            data_train: Dask DataFrame with training data
            record_id_column: Column containing record IDs
            filtering_configs: List of (n_duplicates, mode) tuples to process
            random_seed: Seed for random number generators
            cache_dir: Directory to cache filtered datasets (optional)
            redo_cache: If True, force recomputation even if cache exists
            
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

        # Analyze record distribution if dataset is not too large
        #if len(data_train) < 128298198:
        #    FilteringHandler.analyze_record_id_distribution(data_train, record_id_column)
        
        # Split data once - this is an optimization to avoid repeated computation
        non_duplicates, duplicates = FilteringHandler._split_by_duplicate_status(data_train, record_id_column)
        
        # Find all unique modes needed across configurations
        modes_needed = set(mode for _, mode in filtering_configs if mode is not None)
        
        # Test, if non_duplicates empty (no original values left)
        has_non_duplicates = non_duplicates.shape[0].compute() > 0
        
        if not has_non_duplicates:
            print("Warning: No non-duplicate entries found. Using fallback strategy.")
            # Fallback: random subset of duplicats as "reference"
            if 'knn' in modes_needed or 'autoencoder' in modes_needed or 'imputation' in modes_needed:
                # Wsample 10% or 42000 rows of duplicates as pseudo reference
                sample_size = min(int(duplicates.shape[0].compute() * 0.1), 42000)
                reference_subset = duplicates.sample(n=sample_size, random_state=random_seed)
                non_duplicates = reference_subset  # take these as assumed non-duplicates
                # remove them to avoid overlapping between the sets
                duplicates = duplicates.loc[~duplicates.index.isin(reference_subset.index)]
        
        # Process all duplicates with metadata in a single pass
        processed_duplicates_metadata = duplicates.map_partitions(
            lambda df: FilteringHandler._process_duplicates_with_metadata(df, record_id_column, random_seed)
        ).persist()  # Persist to avoid recomputation
        
        # Prepare result dictionary
        results = {}
        
        # Pre-process profiles for imputation mode if needed - only once
        profiles = None
        if 'imputation' in modes_needed:
            print("Building feature profiles for imputation mode")
            # First create profiles on each partition
            partitioned_profiles = non_duplicates.map_partitions(
                lambda df: FilteringHandler._build_feature_profiles(df, record_id_column)
            ).compute()
            
            # Combine all partition profiles into a single dictionary
            profiles = {}
            for part_profile in partitioned_profiles:
                for col, prof in part_profile.items():
                    if col not in profiles:
                        profiles[col] = prof
            
            if not profiles:
                raise ValueError("Empty profiles generated")
        
        # Pre-process for knn mode if needed - only once
        reference_features = None
        feature_columns = None
        if 'knn' in modes_needed or 'autoencoder' in modes_needed:
            print("Preparing data for KNN/Autoencoder models")
            # Get feature columns (all except record_id)
            feature_columns = [col for col in non_duplicates.columns if col != record_id_column]
            
            # Create reference features from non-duplicates for models
            reference_features = non_duplicates[feature_columns].compute()
        
        # Now create the specific filtered datasets for each configuration
        for n_duplicates, mode in filtering_configs:
            if n_duplicates == 0:
                # Just use the non-duplicates directly
                results[(n_duplicates, mode)] = non_duplicates
                print(f"n_duplicates: {n_duplicates}, mode: {mode}")
                print(f"using original data of size: {non_duplicates.shape[0].compute()}")
                continue
                
            # Extract appropriate subset based on configuration
            if mode == 'random':
                processed_duplicates = processed_duplicates_metadata.map_partitions(
                    lambda df: FilteringHandler._extract_random_duplicates(df, record_id_column, n_duplicates, random_seed)
                )
            elif mode == 'imputation':
                processed_duplicates = processed_duplicates_metadata.map_partitions(
                    lambda df: FilteringHandler._score_and_extract_duplicates(df, record_id_column, profiles, n_duplicates, random_seed)
                )
                # After processing, explicitly update metadata by dropping columns at the Dask DataFrame level
                processed_duplicates = processed_duplicates.drop(['_random_order', '_similarity_score'], axis=1, errors='ignore')
            elif mode == 'knn':
                processed_duplicates = FilteringHandler._process_with_knn(
                    processed_duplicates_metadata, 
                    record_id_column, 
                    reference_features,
                    feature_columns,
                    n_duplicates,
                    random_seed
                )
            elif mode == 'autoencoder':
                processed_duplicates = FilteringHandler._process_with_autoencoder(
                    processed_duplicates_metadata,
                    record_id_column,
                    reference_features,
                    feature_columns,
                    n_duplicates,
                    random_seed
                )
            else:
                raise ValueError(f"Unsupported mode: {mode}")
                
            # Combine with non-duplicates
            filtered_result = dask.dataframe.concat([non_duplicates, processed_duplicates])
            results[(n_duplicates, mode)] = filtered_result.persist()
            
            # Log results
            print(f"n_duplicates: {n_duplicates}, mode: {mode}")
            
            initial_count = data_train.shape[0].compute()
            final_count = results[(n_duplicates, mode)].shape[0].compute()
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
    def _create_preprocessor(reference_features):
        """
        Create a consistent preprocessor for feature transformation.
        
        Args:
            reference_features: DataFrame with reference features
            
        Returns:
            tuple: (preprocessor, numeric_cols, cat_cols)
        """
        # Identify numeric and categorical columns
        numeric_cols = reference_features.select_dtypes(include=['number']).columns
        cat_cols = reference_features.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ], remainder='drop')
        
        # Fit the preprocessor on reference features
        preprocessor.fit(reference_features)
        
        return preprocessor, numeric_cols, cat_cols
    
    @staticmethod
    def _process_with_knn(duplicate_data, record_id_column, reference_features, feature_columns, n_duplicates, random_seed=42):
        """
        Process duplicates using K-Nearest Neighbors for efficient similarity calculation.
        This is much faster than pairwise comparison for large datasets.
        
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
    def _process_with_autoencoder(duplicate_data, record_id_column, reference_features, feature_columns, n_duplicates, random_seed=42):
        """
        Process duplicates using an autoencoder for feature reconstruction and similarity calculation.
        
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
        
        # Set seeds for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        
        print(f"Processing with Autoencoder mode (n_duplicates={n_duplicates})")
        
        # Define the autoencoder architecture
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim=64):
                super(Autoencoder, self).__init__()
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(True),
                    nn.Linear(128, encoding_dim),
                    nn.ReLU(True)
                )
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 128),
                    nn.ReLU(True),
                    nn.Linear(128, input_dim)
                )
                
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
                
            def encode(self, x):
                return self.encoder(x)
        
        # Prepare data for training
        print("Preparing data for autoencoder training")
        
        # Create preprocessor
        preprocessor, _, _ = FilteringHandler._create_preprocessor(reference_features)
        
        # Fit and transform the reference data
        print("Preprocessing reference data")
        preprocessed_features = preprocessor.transform(reference_features)
        
        # Convert to PyTorch tensor
        X_train = torch.tensor(preprocessed_features.toarray() 
                              if hasattr(preprocessed_features, "toarray") 
                              else preprocessed_features, 
                              dtype=torch.float32)
        
        # Create DataLoader for batch training
        batch_size = 256
        train_dataset = TensorDataset(X_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize the autoencoder
        input_dim = X_train.shape[1]
        encoding_dim = min(64, input_dim // 2)  # Bottleneck dimension
        autoencoder = Autoencoder(input_dim, encoding_dim)
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        autoencoder = autoencoder.to(device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        
        # Train the autoencoder
        print("Training autoencoder...")
        epochs = 10
        for epoch in range(epochs):
            running_loss = 0.0
            for data in train_loader:
                inputs = data[0].to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.6f}')
        
        print("Autoencoder training completed")
        
        # Switch to evaluation mode
        autoencoder.eval()
        
        # Define function to process each partition
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
                
                try:
                    # Transform features using the same preprocessor
                    processed_features = preprocessor.transform(group_features)
                    
                    # Convert to tensor
                    tensor_features = torch.tensor(processed_features.toarray() 
                                                 if hasattr(processed_features, "toarray") 
                                                 else processed_features, 
                                                 dtype=torch.float32).to(device)
                    
                    # Calculate reconstruction error
                    with torch.no_grad():
                        reconstructed = autoencoder(tensor_features)
                        errors = torch.mean((reconstructed - tensor_features)**2, dim=1)
                    
                    # Create a copy with errors
                    group_with_error = group.copy()
                    group_with_error['_reconstruction_error'] = errors.cpu().numpy()
                    
                    # Keep instances with smallest reconstruction error
                    top_matches = group_with_error.nsmallest(n_duplicates, '_reconstruction_error')
                    groups.append(top_matches.drop('_reconstruction_error', axis=1))
                    
                except Exception as e:
                    # Fall back to random selection if transformation fails
                    print(f"Error in autoencoder processing for record {record_id}: {e}")
                    sampled = group.sample(n=min(n_duplicates, len(group)), random_state=random_seed)
                    groups.append(sampled)
            
            if groups:
                return pd.concat(groups)
            else:
                return df_part.head(0)  # Empty frame with same schema
        
        # Process each partition and collect results
        print("Processing duplicates with autoencoder...")
        result = duplicate_data.map_partitions(process_partition).persist()
        
        # Remove metadata column
        result = result.drop('_random_order', axis=1, errors='ignore')
        
        return result
