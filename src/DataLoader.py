import os
import pandas as pd
import dask.dataframe as dd
import time
from src.PreparingMethod import PreparingMethod
from src.DatasetManager import DatasetManager
from src.dask_utils import count_dask_rows
from enum import Enum, auto
from typing import Tuple


class DatasetPart(Enum):
    """Enum for specifying which part of the dataset to process."""
    TRAIN = auto()
    TEST = auto()
    BOTH = auto()


class DataLoader:
    """
    Loads and preprocesses data for ML experiments with different anonymization strategies.
    Handles various preprocessing methods for train and test sets independently.
    """
    
    def __init__(self, data_dir: str, dataset: str, partition_size: int = 32, percentages: str = None):
        """
        Initialize data loader with dataset configuration.
        
        Args:
            data_dir: Base directory containing datasets
            dataset: Name of the dataset to load
            partition_size: Number of partitions for initial loading
            percentages: Percentage string for subfolder (e.g., '33-33-34')
        """
        # Store configuration parameters
        self.dataset = dataset
        self.config = DatasetManager.get_config(dataset)
        self.dataset_path = os.path.join(data_dir, dataset)
        self.percentages = percentages
        
        # Load original data (true original without any generalization)
        train_path = os.path.join(self.dataset_path, f"{dataset}_train.csv")
        test_path = os.path.join(self.dataset_path, f"{dataset}_test.csv")
        self.data_train_original = pd.read_csv(train_path)
        self.data_test_original = pd.read_csv(test_path)
        
        # Load generalized reference data (used for profile building in filtering)
        # Do not fall back to original data if generalized files are missing.
        # If percentages explicitly indicate only ORIGINAL data is being used
        # (Main.py sets percentages = "1-0-0" when both train and test are original),
        # don't try to load generalized reference files that won't exist — use the
        # original datasets as the generalized reference instead.
        if self.percentages is None:
            raise ValueError("percentages must be specified to load generalized reference data (no fallback allowed)")

        if str(self.percentages) == "1-0-0":
            # Use original data as generalized reference when only original sets are used
            self.data_train_generalized_reference = self.data_train_original.copy()
            self.data_test_generalized_reference = self.data_test_original.copy()
        else:
            gen_train_path = os.path.join(self.dataset_path, 'generalization', self.percentages, f"{self.dataset}_train.csv")
            gen_test_path = os.path.join(self.dataset_path, 'generalization', self.percentages, f"{self.dataset}_test.csv")

            if not os.path.exists(gen_train_path):
                raise FileNotFoundError(f"Generalized training reference not found: {gen_train_path}")
            if not os.path.exists(gen_test_path):
                raise FileNotFoundError(f"Generalized test reference not found: {gen_test_path}")

            self.data_train_generalized_reference = pd.read_csv(gen_train_path)
            self.data_test_generalized_reference = pd.read_csv(gen_test_path)
        
        # Convert to Dask DataFrames for parallel processing
        self.config["partition_size"] = partition_size
        self.data_train = dd.from_pandas(self.data_train_original, npartitions=partition_size)
        self.data_test = dd.from_pandas(self.data_test_original, npartitions=partition_size)
        
        # Track which methods have been applied
        self.applied_methods = {
            DatasetPart.TRAIN: None,
            DatasetPart.TEST: None
        }
        
        # Track row counts during merges to avoid expensive counting operations
        # We track per-record_id counts for worst-case upper bound (assumes all combinations exist)
        self.row_count_tracker = {
            DatasetPart.TRAIN: {
                'base_rows': len(self.data_train_original),
                'current_upper_bound': len(self.data_train_original),
                'merge_multipliers': [],
                'per_record_counts': None  # Will store worst-case count per record_id
            },
            DatasetPart.TEST: {
                'base_rows': len(self.data_test_original),
                'current_upper_bound': len(self.data_test_original),
                'merge_multipliers': [],
                'per_record_counts': None  # Will store worst-case count per record_id
            }
        }

        # Remove old Spalten enum logic, use new get_spalten_classes
        self.spalten_dict, self.spalten_list = DatasetManager.get_spalten_classes(dataset)

    def preprocess(self, columns, train_method: PreparingMethod = None, 
                   test_method: PreparingMethod = None):
        """
        Apply preprocessing methods to the selected columns, with different methods
        for train and test sets.
        
        Args:
            columns: List of column enums to process
            train_method: Method to apply to training data (None = keep original)
            test_method: Method to apply to testing data (None = keep original)
        """
        # If columns is None, use all columns
        if columns is None:
            columns = self.spalten_list
        # Check if we should add the new "original" method
        if train_method and train_method.name == "original":
            # For "original" method, we handle as if doing no_preprocessing and no anonymization (same as doing nothing)
            # self._preprocess_part([], PreparingMethod.no_preprocessing, DatasetPart.TRAIN)
            self.applied_methods[DatasetPart.TRAIN] = train_method
        elif train_method:
            self._preprocess_part(columns, train_method, DatasetPart.TRAIN)
            self.applied_methods[DatasetPart.TRAIN] = train_method
            
        if test_method and test_method.name == "original":
            # For "original" method, we don't need to do anything as we already have the original data (same as doing nothing)
            # self._preprocess_part([], PreparingMethod.no_preprocessing, DatasetPart.TEST)
            self.applied_methods[DatasetPart.TEST] = test_method
        elif test_method:
            self._preprocess_part(columns, test_method, DatasetPart.TEST)
            self.applied_methods[DatasetPart.TEST] = test_method

    def _preprocess_part(self, columns, method: PreparingMethod, part: DatasetPart):
        """
        Apply the specified preprocessing method to the selected part of the dataset.
        
        Args:
            columns: List of column enums to process
            method: Preprocessing method to apply
            part: Which part of the dataset to process (TRAIN or TEST)
        """
        if "llm_imputation" in method.name:
            # Load LLM-imputed data from files
            base_path = 'llm_evaluation'
            if self.percentages:
                train_path = os.path.join(base_path, self.percentages, f"{self.dataset}_train_imputed_dataset.csv")
                test_path = os.path.join(base_path, self.percentages, f"{self.dataset}_test_imputed_dataset.csv")
            else:
                raise ValueError("llm_imputation method requires percentages to be specified")
            
            # Check if files exist before proceeding
            if part in [DatasetPart.TRAIN, DatasetPart.BOTH]:
                if not os.path.exists(train_path):
                    raise FileNotFoundError(f"LLM imputed training data not found: {train_path}")
            if part in [DatasetPart.TEST, DatasetPart.BOTH]:
                if not os.path.exists(test_path):
                    raise FileNotFoundError(f"LLM imputed test data not found: {test_path}")
            
            self._apply_llm_imputation(train_path, test_path, columns, part)
        
        elif any(keyword in method.name for keyword in ("no_preprocessing", "baseline_imputation")):
            base_path = 'generalization'
            if self.percentages:
                train_path = os.path.join(self.dataset_path, base_path, self.percentages, f"{self.dataset}_train.csv")
                test_path = os.path.join(self.dataset_path, base_path, self.percentages, f"{self.dataset}_test.csv")
            else:
                train_path = os.path.join(self.dataset_path, base_path, f"{self.dataset}_train.csv")
                test_path = os.path.join(self.dataset_path, base_path, f"{self.dataset}_test.csv")
            self._apply_no_preprocessing(train_path, test_path, columns, part)
        
        elif "forced_generalization" in method.name:
            if self.percentages:
                path = os.path.join(self.dataset_path, method.name, self.percentages, 'vorverarbeitet.csv')
            else:
                path = os.path.join(self.dataset_path, method.name, 'vorverarbeitet.csv')
            self._apply_forced_generalization(path, columns, part)
        
        else:  # Handle specialization methods
            base_path = method.name if method in [
                PreparingMethod.extended_weighted_specialization
            ] else 'specialization'
            if self.percentages:
                path = os.path.join(self.dataset_path, base_path, self.percentages)
            else:
                path = os.path.join(self.dataset_path, base_path)
            self._apply_specialization(path, columns, part)

    def _apply_no_preprocessing(self, train_path: str, test_path: str, columns, part: DatasetPart):
        """
        Apply no-preprocessing method using generalized data files to specified part.
        
        Args:
            train_path: Path to generalized training data
            test_path: Path to generalized testing data
            columns: List of column enums to process
            part: Which part of the dataset to process
        """
        # Load appropriate data based on which part we're processing
        if part == DatasetPart.TRAIN:
            generalized = pd.read_csv(train_path)
        elif part == DatasetPart.TEST:
            generalized = pd.read_csv(test_path)
        else:  # BOTH
            generalized = pd.concat(
                [pd.read_csv(train_path), pd.read_csv(test_path)],
                ignore_index=True
            )
        
        # Map enum values to column names and filter columns
        column_names = [col.name for col in columns]
        generalized = generalized[
            [self.config["record_id_column"]] + [col for col in generalized.columns if col in column_names]
        ]
        
        # Process the appropriate part
        self._merge_preprocessed_data(generalized, part)

    def _apply_llm_imputation(self, train_path: str, test_path: str, columns, part: DatasetPart):
        """
        Apply LLM imputation method by loading pre-imputed data files.
        Similar to no_preprocessing but loads from llm_evaluation folder.
        
        Args:
            train_path: Path to LLM-imputed training data
            test_path: Path to LLM-imputed testing data
            columns: List of column enums to process
            part: Which part of the dataset to process
        """
        # Load appropriate data based on which part we're processing
        if part == DatasetPart.TRAIN:
            imputed = pd.read_csv(train_path)
        elif part == DatasetPart.TEST:
            imputed = pd.read_csv(test_path)
        else:  # BOTH
            imputed = pd.concat(
                [pd.read_csv(train_path), pd.read_csv(test_path)],
                ignore_index=True
            )
        
        # Map enum values to column names and filter columns
        column_names = [col.name for col in columns]
        imputed = imputed[
            [self.config["record_id_column"]] + [col for col in imputed.columns if col in column_names]
        ]
        
        # Process the appropriate part
        self._merge_preprocessed_data(imputed, part)

    def _apply_forced_generalization(self, prepared_path: str, columns, part: DatasetPart):
        """
        Apply forced generalization preprocessing from a single preprocessed CSV file.
        
        Args:
            prepared_path: Path to the preprocessed data
            columns: List of column enums to process
            part: Which part of the dataset to process
        """
        # Load preprocessed data
        preprocessed = pd.read_csv(prepared_path)
        
        # Map enum values to column names
        column_names = [col.name for col in columns]
        
        # Keep only relevant columns and record_id
        preprocessed = preprocessed[
            [self.config["record_id_column"]] + [col for col in preprocessed.columns if col in column_names]
        ]
        
        # Merge with the appropriate part
        self._merge_preprocessed_data(preprocessed, part)

    def _apply_specialization(self, base_path: str, columns, part: DatasetPart):
        """
        Apply specialization preprocessing by loading individual column files.
        
        Args:
            base_path: Base path to specialized column files
            columns: List of column enums to process
            part: Which part of the dataset to process
        """
        for i, column in enumerate(columns):
            # Load preprocessed data for this column
            file_path = os.path.join(base_path, f"{column.name}_vorverarbeitet.csv")
            preprocessed = pd.read_csv(file_path)
            
            # Merge with the appropriate part
            self._merge_preprocessed_data(preprocessed, part)
            print(f"Processed column {i+1}/{len(columns)}: {column.name} for {part.name}")

    def calculate_specialization_stats_only(self, columns, train_method: PreparingMethod = None, 
                                            test_method: PreparingMethod = None):
        """
        Calculate upper bound row counts for specialization WITHOUT actually merging the data.
        This is useful when using record-based filtering, which doesn't need the merged data
        but still wants to report upper bound estimates.
        
        This method only reads the specialization CSV files to compute per-record counts
        and upper bounds, without creating the expensive merged Dask DataFrames.
        
        Args:
            columns: List of column enums to process
            train_method: Method for training data (if specialization, compute stats)
            test_method: Method for test data (if specialization, compute stats)
        """
        train_is_spec = train_method and "specialization" in train_method.name
        test_is_spec = test_method and "specialization" in test_method.name
        
        if not train_is_spec and not test_is_spec:
            return  # Nothing to compute
        
        # Determine base path for specialization files
        base_path = 'specialization'
        if self.percentages:
            path = os.path.join(self.dataset_path, base_path, self.percentages)
        else:
            path = os.path.join(self.dataset_path, base_path)
        
        print("Calculating specialization upper bounds (stats only, no data merge)...")
        
        for i, column in enumerate(columns):
            file_path = os.path.join(path, f"{column.name}_vorverarbeitet.csv")
            preprocessed = pd.read_csv(file_path)
            
            record_id_col = self.config["record_id_column"]
            counts_per_record = preprocessed.groupby(record_id_col).size()
            
            avg_multiplier = counts_per_record.mean()
            min_multiplier = counts_per_record.min()
            max_multiplier = counts_per_record.max()
            
            print(f"  Merge multiplier: avg={avg_multiplier:.2f}, range=[{min_multiplier}-{max_multiplier}]")
            
            # Update stats for train if it uses specialization
            if train_is_spec:
                tracker = self.row_count_tracker[DatasetPart.TRAIN]
                tracker['merge_multipliers'].append(avg_multiplier)
                
                if tracker['per_record_counts'] is None:
                    tracker['per_record_counts'] = counts_per_record.to_dict()
                else:
                    new_per_record = {}
                    for record_id, current_count in tracker['per_record_counts'].items():
                        new_count = counts_per_record.get(record_id, 1)
                        new_per_record[record_id] = current_count * new_count
                    tracker['per_record_counts'] = new_per_record
                
                tracker['current_upper_bound'] = sum(tracker['per_record_counts'].values())
            
            # Update stats for test if it uses specialization
            if test_is_spec:
                tracker = self.row_count_tracker[DatasetPart.TEST]
                tracker['merge_multipliers'].append(avg_multiplier)
                
                if tracker['per_record_counts'] is None:
                    tracker['per_record_counts'] = counts_per_record.to_dict()
                else:
                    new_per_record = {}
                    for record_id, current_count in tracker['per_record_counts'].items():
                        new_count = counts_per_record.get(record_id, 1)
                        new_per_record[record_id] = current_count * new_count
                    tracker['per_record_counts'] = new_per_record
                
                tracker['current_upper_bound'] = sum(tracker['per_record_counts'].values())
            
            print(f"Stats computed for column {i+1}/{len(columns)}: {column.name}")
        
        # Print summary
        if train_is_spec:
            train_rows = self.row_count_tracker[DatasetPart.TRAIN]['current_upper_bound']
            print(f"Train upper bound: {train_rows:,} rows")
        if test_is_spec:
            test_rows = self.row_count_tracker[DatasetPart.TEST]['current_upper_bound']
            print(f"Test upper bound: {test_rows:,} rows")

    def _merge_preprocessed_data(self, preprocessed_df: pd.DataFrame, part: DatasetPart):
        """
        Helper method to merge preprocessed data with specified dataset part.
        
        Args:
            preprocessed_df: Preprocessed DataFrame to merge
            part: Which part of the dataset to process
        """
        # Get columns to merge (excluding record_id)
        merge_columns = [col for col in preprocessed_df.columns if col != self.config["record_id_column"]]
        
        # We track per-record_id counts
        record_id_col = self.config["record_id_column"]
        counts_per_record = preprocessed_df.groupby(record_id_col).size()
        
        # Statistics for logging
        avg_multiplier = counts_per_record.mean()
        min_multiplier = counts_per_record.min()
        max_multiplier = counts_per_record.max()
        
        print(f"  Merge multiplier: avg={avg_multiplier:.2f}, range=[{min_multiplier}-{max_multiplier}]")
        
        # Update row count tracker for affected parts with EXACT per-record counts
        if part in [DatasetPart.TRAIN, DatasetPart.BOTH]:
            tracker = self.row_count_tracker[DatasetPart.TRAIN]
            tracker['merge_multipliers'].append(avg_multiplier)
            
            # Calculate UPPER BOUND using per-record multiplication (assumes all combinations exist)
            if tracker['per_record_counts'] is None:
                # First merge: initialize with counts from this column
                tracker['per_record_counts'] = counts_per_record.to_dict()
            else:
                # Subsequent merges: multiply existing counts by new counts
                new_per_record = {}
                for record_id, current_count in tracker['per_record_counts'].items():
                    new_count = counts_per_record.get(record_id, 1)  # Default to 1 if not in this column
                    new_per_record[record_id] = current_count * new_count
                tracker['per_record_counts'] = new_per_record
            
            # Sum all per-record counts for worst-case upper bound
            tracker['current_upper_bound'] = sum(tracker['per_record_counts'].values())
        
        if part in [DatasetPart.TEST, DatasetPart.BOTH]:
            tracker = self.row_count_tracker[DatasetPart.TEST]
            tracker['merge_multipliers'].append(avg_multiplier)
            
            # Calculate UPPER BOUND using per-record multiplication (assumes all combinations exist)
            if tracker['per_record_counts'] is None:
                # First merge: initialize with counts from this column
                tracker['per_record_counts'] = counts_per_record.to_dict()
            else:
                # Subsequent merges: multiply existing counts by new counts
                new_per_record = {}
                for record_id, current_count in tracker['per_record_counts'].items():
                    new_count = counts_per_record.get(record_id, 1)  # Default to 1 if not in this column
                    new_per_record[record_id] = current_count * new_count
                tracker['per_record_counts'] = new_per_record
            
            # Sum all per-record counts for worst-case upper bound
            tracker['current_upper_bound'] = sum(tracker['per_record_counts'].values())
        
        # Convert to Dask DataFrame with proper partitioning
        preprocessed_ddf = dd.from_pandas(preprocessed_df, npartitions=self.config["partition_size"])
        
        # Process train data if requested
        if part in [DatasetPart.TRAIN, DatasetPart.BOTH]:
            # Drop columns from original data that will be replaced
            for column in merge_columns:
                self.data_train = self.data_train.drop(column, axis=1, errors='ignore')
            
            # Merge preprocessed data
            self.data_train = dd.merge(self.data_train, preprocessed_ddf, on=self.config["record_id_column"])
            # Don't repartition here - it forces partitions even when data grows
            # Final repartitioning happens after all merges based on actual row count
            
        # Process test data if requested
        if part in [DatasetPart.TEST, DatasetPart.BOTH]:
            # Drop columns from original data that will be replaced
            for column in merge_columns:
                self.data_test = self.data_test.drop(column, axis=1, errors='ignore')
            
            # Merge preprocessed data
            self.data_test = dd.merge(self.data_test, preprocessed_ddf, on=self.config["record_id_column"])
            # Don't repartition here - let merge determine natural partition count
        
    # TODO refactor and better handling of non-gernalizable attributes?
    def calculate_anonymization_ratios(self, anonymization, train_method: PreparingMethod = None, test_method: PreparingMethod = None):
        # Handle original method for train data
        if train_method and train_method.name == "original":
            # For original method, use the original data (no anonymization)
            df_train = self.data_train_original.copy()
        elif train_method and train_method.name == "llm_imputation":
            # For LLM imputation method, load from llm_evaluation folder
            if self.percentages:
                train_path = os.path.join('llm_evaluation', self.percentages, f"{self.dataset}_train_imputed_dataset.csv")
            else:
                raise ValueError("llm_imputation method requires percentages to be specified")
            df_train = pd.read_csv(train_path)
        else:
            anon_train_method = 'generalization'
            if self.percentages and anon_train_method in ['generalization']:
                train_path = os.path.join(self.dataset_path, anon_train_method, self.percentages, f"{self.dataset}_train.csv")
            else:
                train_path = os.path.join(self.dataset_path, anon_train_method, f"{self.dataset}_train.csv")
            df_train = pd.read_csv(train_path)
        
        # Handle original method for test data
        if test_method and test_method.name == "original":
            # For original method, use the original data (no anonymization)
            df_test = self.data_test_original.copy()
        elif test_method and test_method.name == "llm_imputation":
            # For LLM imputation method, load from llm_evaluation folder
            if self.percentages:
                test_path = os.path.join('llm_evaluation', self.percentages, f"{self.dataset}_test_imputed_dataset.csv")
            else:
                raise ValueError("llm_imputation method requires percentages to be specified")
            df_test = pd.read_csv(test_path)
        else:
            anon_test_method = 'generalization'
            if self.percentages and anon_test_method in ['generalization']:
                test_path = os.path.join(self.dataset_path, anon_test_method, self.percentages, f"{self.dataset}_test.csv")
            else:
                test_path = os.path.join(self.dataset_path, anon_test_method, f"{self.dataset}_test.csv")
            df_test = pd.read_csv(test_path)
        
        df_train.drop(columns=[self.config["record_id_column"], self.config["label_column"]], inplace=True)
        df_test.drop(columns=[self.config["record_id_column"], self.config["label_column"]], inplace=True)
        
        # Count actual values (not ratios) across all anonymized columns
        total_values_train = 0
        total_original_train = 0
        total_generalized_train = 0
        total_missing_train = 0
        
        total_values_test = 0
        total_original_test = 0
        total_generalized_test = 0
        total_missing_test = 0
        
        for column in anonymization.value:
            original_train, generalized_train, missing_train, count_train = self.calculate_data_analysis_by_column(column, df_train)
            total_values_train += count_train
            total_original_train += original_train
            total_generalized_train += generalized_train
            total_missing_train += missing_train
            
            original_test, generalized_test, missing_test, count_test = self.calculate_data_analysis_by_column(column, df_test)
            total_values_test += count_test
            total_original_test += original_test
            total_generalized_test += generalized_test
            total_missing_test += missing_test
    
        # Calculate ratios from total counts
        generalized_ratio_train = total_generalized_train / total_values_train if total_values_train > 0 else 0
        missing_ratio_train = total_missing_train / total_values_train if total_values_train > 0 else 0
        original_ratio_train = total_original_train / total_values_train if total_values_train > 0 else 0
        
        generalized_ratio_test = total_generalized_test / total_values_test if total_values_test > 0 else 0
        missing_ratio_test = total_missing_test / total_values_test if total_values_test > 0 else 0
        original_ratio_test = total_original_test / total_values_test if total_values_test > 0 else 0
        
        # Calculate overall ratios
        total_values_all = total_values_train + total_values_test
        generalized_ratio = (total_generalized_train + total_generalized_test) / total_values_all if total_values_all > 0 else 0
        missing_ratio = (total_missing_train + total_missing_test) / total_values_all if total_values_all > 0 else 0
        original_ratio = (total_original_train + total_original_test) / total_values_all if total_values_all > 0 else 0

        # dictionary storing ratios
        ratios = {
            "Train": {
                "Original": original_ratio_train,
                "Generalized": generalized_ratio_train,
                "Missing": missing_ratio_train,
            },
            "Test": {
                "Original": original_ratio_test,
                "Generalized": generalized_ratio_test,
                "Missing": missing_ratio_test,
            },
            "Total": {
                "Original": original_ratio,
                "Generalized": generalized_ratio,
                "Missing": missing_ratio,
            },
        }

        print("Ratios")
        for key, value in ratios.items():
            print(f"{key}: Original: {value['Original']}, Generalized: {value['Generalized']}, Missing: {value['Missing']}")
    
        return ratios

    # TODO refactor and better handling of non-gernalizable attributes?
    def calculate_data_analysis_by_column(self, column, df):
        total_values = df[column.name].count()
        original = 0
        generalized = 0
        missing = 0
        
        #Gehe jeden Wert der Spalte durch
        for index, row in df.iterrows():
            value = row[column.name]
            if value == "?":
                missing += 1
            elif column.is_generalized(value):
                generalized += 1
            else:
                original += 1

        print(f"Spalte {column.name} analysiert")
        print(f"Anzahl Einträge: {total_values}")
        print(f"Original: {original/total_values}, Generalized: {generalized/total_values}, Missing: {missing/total_values}")
        
        # Return actual counts, not ratios
        return original, generalized, missing, total_values
    

    def get_data(self, verify_row_counts: bool = False) -> Tuple[dd.DataFrame, dd.DataFrame]:
        """
        Finalize data preparation and return train/test datasets.
        
        Args:
            verify_row_counts: If True, verify calculated row counts against actual (slow but validates accuracy)
        
        Returns:
            tuple: Processed training and testing Dask DataFrames
        """
        # Set appropriate data types
        self._set_types()
        
        # Reset indices to ensure consistency
        self.data_train = self.data_train.reset_index(drop=True)
        self.data_test = self.data_test.reset_index(drop=True)
        
        # Ensure consistent column ordering between train and test
        all_columns = set(list(self.data_train.columns) + list(self.data_test.columns))
        
        # Make sure both dataframes have all columns
        for col in all_columns:
            if col not in self.data_train.columns:
                self.data_train[col] = None
            if col not in self.data_test.columns:
                self.data_test[col] = None
                
        # Use the same column order
        column_order = sorted(list(all_columns))
        self.data_train = self.data_train[column_order].reset_index(drop=True)
        self.data_test = self.data_test[column_order].reset_index(drop=True)
        
        # Repartition for specialization methods based on actual data size
        # This MUST happen after preprocessing when we know the real size
        train_method = self.applied_methods.get(DatasetPart.TRAIN)
        test_method = self.applied_methods.get(DatasetPart.TEST)
        is_specialization = any(method and "specialization" in method.name 
                               for method in [train_method, test_method] if method)
        
        if is_specialization:
            # Use upper bound row counts instead of expensive counting
            # This assumes all combinations exist (worst-case) and avoids traversing the entire Dask task graph
            print("Using worst-case upper bound row counts from per-record merge metadata...")
            train_rows = self.row_count_tracker[DatasetPart.TRAIN]['current_upper_bound']
            test_rows = self.row_count_tracker[DatasetPart.TEST]['current_upper_bound']
            
            train_multipliers = self.row_count_tracker[DatasetPart.TRAIN]['merge_multipliers']
            test_multipliers = self.row_count_tracker[DatasetPart.TEST]['merge_multipliers']
            
            print(f"Calculated: {train_rows:,} train rows (base={self.row_count_tracker[DatasetPart.TRAIN]['base_rows']:,}, multipliers={train_multipliers})")
            print(f"Calculated: {test_rows:,} test rows (base={self.row_count_tracker[DatasetPart.TEST]['base_rows']:,}, multipliers={test_multipliers})")
            
            # OPTIONAL VERIFICATION: Compare upper bound vs actual counts (expensive)
            if verify_row_counts:
                print("\n⚠️  VERIFICATION MODE: Comparing upper bound vs actual counts...")
                print("   This will take a long time but shows the overestimate!\n")
                
                actual_train_start = time.time()
                actual_train_count = int(count_dask_rows(self.data_train))
                actual_train_time = time.time() - actual_train_start
                
                actual_test_start = time.time()
                actual_test_count = int(count_dask_rows(self.data_test))
                actual_test_time = time.time() - actual_test_start
                
                print(f"\n📊 VERIFICATION RESULTS:")
                print(f"   Training data:")
                print(f"     Upper bound: {train_rows:,} rows (instant)")
                print(f"     Actual:      {actual_train_count:,} rows ({actual_train_time:.1f}s)")
                print(f"     Status:      {'✅ EXACT' if train_rows == actual_train_count else f'⚠️  OVERESTIMATE: +{train_rows - actual_train_count:,} ({100*(train_rows-actual_train_count)/actual_train_count:.1f}%)'}")
                
                print(f"   Test data:")
                print(f"     Upper bound: {test_rows:,} rows (instant)")
                print(f"     Actual:      {actual_test_count:,} rows ({actual_test_time:.1f}s)")
                print(f"     Status:      {'✅ EXACT' if test_rows == actual_test_count else f'⚠️  OVERESTIMATE: +{test_rows - actual_test_count:,} ({100*(test_rows-actual_test_count)/actual_test_count:.1f}%)'}")
                
                print(f"\n⏱️  Time saved: {actual_train_time + actual_test_time:.1f}s (using upper bound counts)\n")
                
                # Assert for testing purposes
                if train_rows != actual_train_count or test_rows != actual_test_count:
                    raise ValueError(
                        f"Row count mismatch! "
                        f"Train: calculated={train_rows:,} vs actual={actual_train_count:,}, "
                        f"Test: calculated={test_rows:,} vs actual={actual_test_count:,}"
                    )
            
            # Calculate optimal partition count for large datasets
            # For huge datasets (>100M rows), use more partitions with smaller size
            # Target: 2M rows per partition for memory safety
            if train_rows > 100_000_000:  # > 100M rows
                # Use smaller partitions for huge datasets to prevent memory issues
                # Cap at 4096 partitions to avoid too much coordination overhead
                optimal_train_partitions = max(256, min(4096, train_rows // 2_000_000))
                print(f"Repartitioning HUGE training data: {train_rows:,} rows -> {optimal_train_partitions} partitions (~{train_rows//optimal_train_partitions:,} rows/partition)")
                self.data_train = self.data_train.repartition(npartitions=optimal_train_partitions)
            elif train_rows > 1_000_000:
                # Standard large dataset: ~500K rows per partition
                optimal_train_partitions = max(4, min(256, train_rows // 500_000))
                print(f"Repartitioning training data: {train_rows:,} rows -> {optimal_train_partitions} partitions")
                self.data_train = self.data_train.repartition(npartitions=optimal_train_partitions)
            
            if test_rows > 100_000_000:  # > 100M rows
                optimal_test_partitions = max(256, min(4096, test_rows // 2_000_000))
                print(f"Repartitioning HUGE test data: {test_rows:,} rows -> {optimal_test_partitions} partitions (~{test_rows//optimal_test_partitions:,} rows/partition)")
                self.data_test = self.data_test.repartition(npartitions=optimal_test_partitions)
            elif test_rows > 1_000_000:
                optimal_test_partitions = max(4, min(256, test_rows // 500_000))
                print(f"Repartitioning test data: {test_rows:,} rows -> {optimal_test_partitions} partitions")
                self.data_test = self.data_test.repartition(npartitions=optimal_test_partitions)
        
        return self.data_train, self.data_test

    def _set_types(self):
        """
        Set appropriate data types for columns based on dataset configuration.
        """
        # Reference data for categories
        data_original = pd.concat([self.data_train_original, self.data_test_original], ignore_index=True)
        categorical_columns = self.config.get("categorical_columns", [])
        numerical_columns = self.config.get("numerical_columns", [])
        
        # Check if we need special handling for categories
        # Only no_preprocessing and forced_generalization need this because they may have
        # generalized/missing values that aren't in the original data
        # llm_imputation, specialization, original: Complete real values → use original data categories
        needs_computed_categories = any(method and ("no_preprocessing" in method.name or "forced_generalization" in method.name)
                                      for method in self.applied_methods.values() if method)
        
        all_data = None
        if needs_computed_categories:
            all_data = dd.concat([self.data_train, self.data_test], ignore_index=True).compute()
        
        # Get columns to skip
        skip_columns = {self.config["record_id_column"], self.config["label_column"]}
        
        # Get all columns to process
        all_columns = set(list(self.data_train.columns) + list(self.data_test.columns))
        
        # Process each column
        for column in all_columns:
            if column in skip_columns:
                continue
                
            if column not in self.data_train.columns or column not in self.data_test.columns:
                continue
                
            if needs_computed_categories and column in all_data.columns:
                # For these methods, ensure consistent category values across train/test
                all_data[column] = all_data[column].astype('category')
                
                # Ensure unique categories
                unique_categories = list(set(all_data[column].cat.categories))

                # Set categories in train and test to match all_data
                self.data_train[column] = self.data_train[column].astype('category').cat.set_categories(unique_categories)
                self.data_test[column] = self.data_test[column].astype('category').cat.set_categories(unique_categories)
                
            elif (categorical_columns and column in categorical_columns) or (numerical_columns and column not in numerical_columns):
                # Handle categorical columns using original data's categories
                data_original[column] = data_original[column].astype('category')
                
                # Ensure unique categories
                unique_categories = list(set(data_original[column].cat.categories))

                # Set categories in train and test to match original data
                self.data_train[column] = self.data_train[column].astype('category').cat.set_categories(unique_categories)
                self.data_test[column] = self.data_test[column].astype('category').cat.set_categories(unique_categories)
                
            else:
                # Convert numerical-like columns to nullable integer type, coercing non-numeric (e.g., '?') to NA
                # Use Dask to_numeric then pandas nullable Int32 dtype
                self.data_train[column] = dd.to_numeric(
                    self.data_train[column], errors='coerce'
                ).astype('Float64').astype('Int32')
                self.data_test[column] = dd.to_numeric(
                    self.data_test[column], errors='coerce'
                ).astype('Float64').astype('Int32')

            # else:
            #     # Try to convert numerical columns to int32 for efficiency
            #     try:
            #         self.data_train[column] = self.data_train[column].astype('int32')
            #         self.data_test[column] = self.data_test[column].astype('int32')
            #     except ValueError:
            #         print(f"Column {column} could not be converted to int32.")
