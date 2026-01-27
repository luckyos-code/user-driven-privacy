import pandas as pd
import dask.dataframe as dd
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import List, Dict, Tuple, Any
import importlib

class ImputationHandler:
    """
    Handles MICE-based imputation and simple statistical imputation for baseline comparison.
    """
    
    @staticmethod
    def convert_to_missing(
        data: dd.DataFrame,
        dataset_name: str,
        columns_with_ranges: List[str] = None,
        missing_indicator: str = "?"
    ) -> dd.DataFrame:
        """
        Convert generalized ranges and missing indicators to NaN.
        
        Args:
            data: Input dataframe (original values, generalized ranges, missing)
            dataset_name: Name of the dataset (e.g., 'adult', 'diabetes')
            columns_with_ranges: Columns that may contain generalized ranges (if None, check all)
            missing_indicator: String used to indicate missing values (default "?")
        
        Returns:
            Dataframe with generalized/missing replaced by NaN
            
        Example:
            age: "[30-39]" → NaN
            age: 35 → 35
            education: "?" → NaN
        """
        result = data.copy()
        
        # Get spalten classes for this dataset
        from src.DatasetManager import DatasetManager
        spalten_dict, spalten_list = DatasetManager.get_spalten_classes(dataset_name)
        
        # If no specific columns provided, check all columns
        if columns_with_ranges is None:
            columns_with_ranges = list(spalten_dict.keys())
        
        def replace_generalized_with_nan(series, column_name):
            """Replace generalized/missing values with NaN for a single column"""
            if column_name not in spalten_dict:
                return series
            
            spalten_class = spalten_dict[column_name]
            
            def convert_value(val):
                # Check if missing
                if pd.isna(val) or val == missing_indicator or val == '':
                    return np.nan
                
                # Check if generalized (exists in any privacy level dict)
                if hasattr(spalten_class, 'is_generalized') and spalten_class.is_generalized(val):
                    return np.nan
                
                # Otherwise keep original value
                return val
            
            return series.map(convert_value, meta=(column_name, 'object'))
        
        # Process each column
        for col in columns_with_ranges:
            if col in result.columns:
                result[col] = replace_generalized_with_nan(result[col], col)
        
        return result
    
    @staticmethod
    def extract_range_metadata(
        data: dd.DataFrame,
        dataset_name: str,
        columns: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract range constraints from generalized values using spalten definitions.
        
        Args:
            data: Original dataframe before converting to NaN
            dataset_name: Name of the dataset
            columns: Columns to check for ranges (if None, check all)
        
        Returns:
            {column: {'min': lower_bound, 'max': upper_bound, 'values': [list of valid values]}}
            
        Example:
            age has "[30-39]" → {'age': {'min': 30, 'max': 39}}
            education → {'education': {'values': ['HS-grad', 'Bachelors', ...]}}
        """
        from src.DatasetManager import DatasetManager
        spalten_dict, spalten_list = DatasetManager.get_spalten_classes(dataset_name)
        
        if columns is None:
            columns = list(spalten_dict.keys())
        
        range_metadata = {}
        
        for col in columns:
            if col not in spalten_dict:
                continue
            
            spalten_class = spalten_dict[col]
            
            # Check if column has dict_all attribute
            if not hasattr(spalten_class, 'dict_all'):
                continue
            
            # Extract all possible values across all privacy levels
            all_values = set()
            min_val = None
            max_val = None
            is_numeric = False
            
            for privacy_level, value_dict in spalten_class.dict_all.items():
                for key, values in value_dict.items():
                    if isinstance(values, range):
                        # Numeric column with ranges
                        is_numeric = True
                        all_values.update(values)
                        if min_val is None or min(values) < min_val:
                            min_val = min(values)
                        if max_val is None or max(values) > max_val:
                            max_val = max(values)
                    elif isinstance(values, list):
                        # Categorical column
                        all_values.update(values)
            
            # Store metadata
            if is_numeric:
                range_metadata[col] = {
                    'type': 'numeric',
                    'min': min_val,
                    'max': max_val,
                    'all_values': sorted(list(all_values))
                }
            else:
                range_metadata[col] = {
                    'type': 'categorical',
                    'values': sorted(list(all_values))
                }
        
        return range_metadata
    
    @staticmethod
    def apply_mice_imputation(
        data: dd.DataFrame,
        categorical_columns: List[str],
        numerical_columns: List[str],
        random_seed: int = 42,
        max_iter: int = 10,
        n_nearest_features: int = None
    ) -> dd.DataFrame:
        """
        Apply MICE imputation using sklearn.impute.IterativeImputer.
        
        Args:
            data: Dataframe with NaN values to impute
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            random_seed: Random seed for reproducibility
            max_iter: Maximum iterations for MICE (default 10)
            n_nearest_features: Number of features to use (None = all)
        
        Returns:
            Dataframe with imputed values
            
        Process:
            1. Encode categorical columns (LabelEncoder)
            2. Apply IterativeImputer to all columns
            3. Decode categorical columns back to original labels
            4. Round categorical predictions to nearest valid class
        """
        # Convert dask to pandas for sklearn (MICE needs full data in memory)
        df = data.compute() if isinstance(data, dd.DataFrame) else data.copy()
        
        # Store original dtypes and index
        original_index = df.index
        original_columns = df.columns.tolist()
        
        # Identify columns that should NOT be imputed (record_id, label, etc.)
        # These are columns not in the specified categorical or numerical lists
        columns_to_impute = set(categorical_columns + numerical_columns)
        columns_to_preserve = [col for col in df.columns if col not in columns_to_impute]
        
        # Store these columns separately
        preserved_data = df[columns_to_preserve].copy() if columns_to_preserve else None
        
        # Separate columns by type
        cat_cols_in_data = [col for col in categorical_columns if col in df.columns]
        num_cols_in_data = [col for col in numerical_columns if col in df.columns]
        
        # Encode categorical columns
        label_encoders = {}
        encoded_df = df.copy()
        
        for col in cat_cols_in_data:
            # Convert to string dtype first to avoid Categorical issues
            encoded_df[col] = encoded_df[col].astype(str)
            
            le = LabelEncoder()
            # Get non-null values for fitting
            non_null_mask = (encoded_df[col].notna()) & (encoded_df[col] != 'nan')
            if non_null_mask.sum() > 0:
                le.fit(encoded_df.loc[non_null_mask, col])
                # Transform non-null values - create new series to avoid setitem issues
                encoded_values = le.transform(encoded_df.loc[non_null_mask, col])
                # Create numeric column with NaN for null values
                encoded_df[col] = pd.Series(np.nan, index=encoded_df.index, dtype=float)
                encoded_df.loc[non_null_mask, col] = encoded_values
                # Store encoder
                label_encoders[col] = le
        
        # Convert all columns to numeric for imputation
        for col in encoded_df.columns:
            encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce')
        
        # Get columns that will be used in imputation (have at least some non-null values)
        # Only include columns we actually want to impute (exclude record_id, label, etc.)
        cols_for_imputation = [col for col in encoded_df.columns 
                               if col in columns_to_impute and encoded_df[col].notna().sum() > 0]
        
        if len(cols_for_imputation) == 0:
            # No data to impute, return original
            return dd.from_pandas(df, npartitions=data.npartitions if isinstance(data, dd.DataFrame) else 1)
        
        # Apply MICE imputation only to columns with data
        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=random_seed,
            n_nearest_features=n_nearest_features,
            verbose=0
        )
        
        imputed_array = imputer.fit_transform(encoded_df[cols_for_imputation])
        
        # Create DataFrame with imputed values
        imputed_df = df.copy()
        for i, col in enumerate(cols_for_imputation):
            imputed_df[col] = imputed_array[:, i]
        
        # Decode categorical columns back to original values
        for col in cat_cols_in_data:
            if col in label_encoders:
                le = label_encoders[col]
                # Round to nearest integer (class index)
                imputed_df[col] = imputed_df[col].round().astype(int)
                # Clip to valid range
                imputed_df[col] = imputed_df[col].clip(0, len(le.classes_) - 1)
                # Inverse transform
                imputed_df[col] = le.inverse_transform(imputed_df[col])
                # Convert to category dtype for XGBoost compatibility
                imputed_df[col] = imputed_df[col].astype('category')
        
        # Ensure numeric columns have appropriate types
        for col in num_cols_in_data:
            imputed_df[col] = pd.to_numeric(imputed_df[col], errors='coerce')
        
        # Restore preserved columns (record_id, label, etc.) that should not have been imputed
        if preserved_data is not None:
            for col in columns_to_preserve:
                imputed_df[col] = preserved_data[col]
        
        # Convert back to Dask DataFrame
        return dd.from_pandas(imputed_df, npartitions=data.npartitions if isinstance(data, dd.DataFrame) else 1)
    
    @staticmethod
    def apply_constrained_mice_imputation(
        data: dd.DataFrame,
        dataset_name: str,
        categorical_columns: List[str],
        numerical_columns: List[str],
        random_seed: int = 42,
        max_iter: int = 10
    ) -> dd.DataFrame:
        """
        Apply MICE then clip values to respect range constraints from spalten definitions.
        
        Args:
            data: Dataframe with NaN values
            dataset_name: Name of the dataset
            categorical_columns: Categorical column names
            numerical_columns: Numerical column names
            random_seed: Random seed
            max_iter: Maximum iterations for MICE
        
        Returns:
            Dataframe with imputed and clipped values
        """
        # Extract range metadata from spalten definitions
        range_metadata = ImputationHandler.extract_range_metadata(
            data, dataset_name, categorical_columns + numerical_columns
        )
        
        # First apply standard MICE
        imputed = ImputationHandler.apply_mice_imputation(
            data, categorical_columns, numerical_columns, random_seed, max_iter
        )
        
        # Convert to pandas for easier manipulation
        df = imputed.compute() if isinstance(imputed, dd.DataFrame) else imputed.copy()
        
        # Clip numerical values to ranges
        for col, constraints in range_metadata.items():
            if col in df.columns and constraints['type'] == 'numeric':
                df[col] = df[col].clip(
                    lower=constraints.get('min'),
                    upper=constraints.get('max')
                )
        
        # For categorical columns, ensure values are within valid set
        for col, constraints in range_metadata.items():
            if col in df.columns and constraints['type'] == 'categorical':
                valid_values = set(constraints['values'])
                # Replace invalid values with most frequent valid value
                invalid_mask = ~df[col].isin(valid_values)
                if invalid_mask.sum() > 0:
                    # Get mode of valid values
                    valid_mode = df[col][~invalid_mask].mode()
                    if len(valid_mode) > 0:
                        df.loc[invalid_mask, col] = valid_mode[0]
                # Ensure category dtype is preserved
                if df[col].dtype.name != 'category':
                    df[col] = df[col].astype('category')
        
        # Convert back to Dask DataFrame
        return dd.from_pandas(df, npartitions=imputed.npartitions if isinstance(imputed, dd.DataFrame) else 1)
    
    @staticmethod
    def apply_simple_imputation(
        data: dd.DataFrame,
        categorical_columns: List[str],
        numerical_columns: List[str]
    ) -> dd.DataFrame:
        """
        Apply simple statistical imputation (mode for categorical, mean for numeric).
        
        Args:
            data: Dataframe with NaN values to impute
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
        
        Returns:
            Dataframe with imputed values
        """
        result = data.copy()
        
        # For categorical: fill with mode (most frequent value)
        for col in categorical_columns:
            if col in result.columns:
                # Compute mode
                mode_result = result[col].mode().compute()
                if len(mode_result) > 0:
                    mode_value = mode_result.iloc[0]
                    result[col] = result[col].fillna(mode_value)
                # Convert to category dtype for XGBoost compatibility
                result[col] = result[col].astype('category')
        
        # For numerical: fill with mean
        for col in numerical_columns:
            if col in result.columns:
                # Convert to numeric to handle object series
                numeric_col = dd.to_numeric(result[col], errors='coerce')
                mean_value = numeric_col.mean().compute()
                result[col] = numeric_col.fillna(mean_value)
        
        return result
