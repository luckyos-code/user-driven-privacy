import os
import pandas as pd
import dask.dataframe as dd
from src.PreparingMethod import PreparingMethod
from src.DatasetManager import DatasetManager
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
            percentages: Percentage string for subfolder (e.g., '33-33-34')
        """
        # Store configuration parameters
        self.dataset = dataset
        self.config = DatasetManager.get_config(dataset)
        self.dataset_path = os.path.join(data_dir, dataset)
        self.percentages = percentages
        
        # Load original data
        train_path = os.path.join(self.dataset_path, f"{dataset}_train.csv")
        test_path = os.path.join(self.dataset_path, f"{dataset}_test.csv")
        self.data_train_original = pd.read_csv(train_path)
        self.data_test_original = pd.read_csv(test_path)
        
        # Convert to Dask DataFrames for parallel processing
        self.config["partition_size"] = partition_size
        self.data_train = dd.from_pandas(self.data_train_original, npartitions=partition_size)
        self.data_test = dd.from_pandas(self.data_test_original, npartitions=partition_size)
        
        # Track which methods have been applied
        self.applied_methods = {
            DatasetPart.TRAIN: None,
            DatasetPart.TEST: None
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
        if "no_preprocessing" in method.name:
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

    def _merge_preprocessed_data(self, preprocessed_df: pd.DataFrame, part: DatasetPart):
        """
        Helper method to merge preprocessed data with specified dataset part.
        
        Args:
            preprocessed_df: Preprocessed DataFrame to merge
            part: Which part of the dataset to process
        """
        # Get columns to merge (excluding record_id)
        merge_columns = [col for col in preprocessed_df.columns if col != self.config["record_id_column"]]
        
        # Convert to Dask DataFrame with proper partitioning
        preprocessed_ddf = dd.from_pandas(preprocessed_df, npartitions=self.config["partition_size"])
        
        # Process train data if requested
        if part in [DatasetPart.TRAIN, DatasetPart.BOTH]:
            # Drop columns from original data that will be replaced
            for column in merge_columns:
                self.data_train = self.data_train.drop(column, axis=1, errors='ignore')
            
            # Merge preprocessed data
            self.data_train = dd.merge(self.data_train, preprocessed_ddf, on=self.config["record_id_column"])
            self.data_train = self.data_train.repartition(npartitions=self.config["partition_size"])
            
        # Process test data if requested
        if part in [DatasetPart.TEST, DatasetPart.BOTH]:
            # Drop columns from original data that will be replaced
            for column in merge_columns:
                self.data_test = self.data_test.drop(column, axis=1, errors='ignore')
            
            # Merge preprocessed data
            self.data_test = dd.merge(self.data_test, preprocessed_ddf, on=self.config["record_id_column"])
            self.data_test = self.data_test.repartition(npartitions=self.config["partition_size"])
        
    # TODO
    def calculate_anonymization_ratios(self, anonymization, train_method: PreparingMethod = None, test_method: PreparingMethod = None):
        # Handle original method for train data
        if train_method and train_method.name == "original":
            # For original method, use the original data (no anonymization)
            df_train = self.data_train_original.copy()
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
        else:
            anon_test_method = 'generalization'
            if self.percentages and anon_test_method in ['generalization']:
                test_path = os.path.join(self.dataset_path, anon_test_method, self.percentages, f"{self.dataset}_test.csv")
            else:
                test_path = os.path.join(self.dataset_path, anon_test_method, f"{self.dataset}_test.csv")
            df_test = pd.read_csv(test_path)
        
        df_train.drop(columns=[self.config["record_id_column"], self.config["label_column"]], inplace=True)
        df_test.drop(columns=[self.config["record_id_column"], self.config["label_column"]], inplace=True)
        
        total_features = len(df_train.columns)
        total_generalized_train = 0
        total_missing_train = 0
        total_generalized_test = 0
        total_missing_test = 0
        
        for column in anonymization.value:
            generalized_train, missing_train = self.calculate_data_analysis_by_column(column, df_train)
            total_generalized_train += generalized_train
            total_missing_train += missing_train
            generalized_test, missing_test = self.calculate_data_analysis_by_column(column, df_test)
            total_generalized_test += generalized_test
            total_missing_test += missing_test
    
        generalized_ratio_train = total_generalized_train/total_features
        missing_ratio_train = total_missing_train/total_features
        original_ratio_train = 1 - generalized_ratio_train - missing_ratio_train
        generalized_ratio_test = total_generalized_test/total_features
        missing_ratio_test = total_missing_test/total_features
        original_ratio_test = 1 - generalized_ratio_test - missing_ratio_test
        generalized_ratio = (generalized_ratio_train + generalized_ratio_test)/2
        missing_ratio = (missing_ratio_train + missing_ratio_test)/2
        original_ratio = (original_ratio_train + original_ratio_test)/2

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

    # TODO
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
        print(f"Anzahl Einträge: {df[column.name].count()}")
        print(f"Original: {original/total_values}, Generalized: {generalized/total_values}, Missing: {missing/total_values}")
        
        return generalized/total_values, missing/total_values
    

    def get_data(self) -> Tuple[dd.DataFrame, dd.DataFrame]:
        """
        Finalize data preparation and return train/test datasets.
        
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
        needs_computed_categories = any(method and "no_preprocessing" in method.name or "forced_generalization" in method.name 
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
                # Try to convert numerical columns to int32 for efficiency
                try:
                    self.data_train[column] = self.data_train[column].astype('int32')
                    self.data_test[column] = self.data_test[column].astype('int32')
                except ValueError:
                    print(f"Column {column} could not be converted to int32.")
