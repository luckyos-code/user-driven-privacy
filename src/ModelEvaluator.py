from xgboost.dask import DaskXGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from dask.distributed import Client
import dask.dataframe as dd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
import time
from datetime import timedelta
from src.DatasetManager import DatasetManager
import pandas as pd

class ModelEvaluator:
    """
    Class for training and evaluating XGBoost models using Dask for distributed computation.
    
    This class handles distributed model training, prediction, and evaluation with specific 
    support for weighted samples and record-level aggregation. It uses DatasetManager for
    dataset-specific configuration.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test_with_record_ids, client: Client, dataset: str):
        """Initialize with training/test data and Dask client"""
        # Store input data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test_with_record_ids = y_test_with_record_ids.drop_duplicates()
        self.client = client
        
        # Setup logging and results storage
        self.logger = self._setup_logger()
        self.y_true = None  # Will store true labels after prediction
        self.pred_proba = None  # Will store predicted probabilities
        
        # Configure dataset-specific settings
        self.dataset = dataset
        self.label_column = DatasetManager.get_label_column(dataset)
        self.record_id_column = DatasetManager.get_record_id_column(dataset)
        
        # Log initialization
        self.logger.info(f"Initialized with dataset: {dataset}")
        self.logger.info(f"Label column: {self.label_column}")
    
    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance"""
        # logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # if not logger.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     handler.setFormatter(formatter)
        #     logger.addHandler(handler)
        #     logger.setLevel(logging.INFO)
        # return logger

        class PrintingHandler(logging.Handler):
            def emit(self, record):
                message = self.format(record)
                print(message)
        
        # In your _setup_logger method:
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            handler = PrintingHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def train_model(self, weighted=False, highest_confidence=False, group_duplicates=False, use_gpu=False, calc_feature_importance='built_in',
                    params: Optional[Dict[str, Any]]=None) -> Tuple[float, float, float]:
        """Train XGBoost model and evaluate performance"""
        self.logger.info(f"Starting model training for dataset: {self.dataset}")
        
        # Use provided data or default to instance variables
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        
        # Configure model parameters
        model_params = {
            'n_estimators': 40,
            'max_depth': 5,
            'enable_categorical': True,
            'verbosity': 1
        }
        
        if params:
            model_params.update(params)
        
        # Create classifier and configure client
        cl = DaskXGBClassifier(**model_params)

        # set GPU stuff if needed
        if use_gpu:
            self.logger.info("Using GPU acceleration")
            cl.set_params(device="cuda")
            
        cl.client = self.client
        
        try:
            weights = None
            
            # Apply sample weighting if requested
            if weighted:
                self.logger.info("Training with sample weights for specializations")
                start_time = time.time()
                X_train, weights = self.get_weights(X_train)
                X_train = X_train.persist()
                end_time = time.time() 
                
                weighting_time = end_time - start_time
                self.logger.info(f"Calculated weights in {str(timedelta(seconds=weighting_time))}")
            
            # Group duplicate records if requested
            if group_duplicates:
                self.logger.info("Deduplicating training data with summed weights")
                start_time = time.time()
                X_train, weights, y_train = self.group_duplicates(X_train, y_train, weights)
                X_train = X_train.persist()
                end_time = time.time()
                
                dedup_time = end_time - start_time
                self.logger.info(f"Deduplicated in {str(timedelta(seconds=dedup_time))}")
            
            if weights is None:
                self.logger.info("Training without sample weights for specializations")
            
            # Remove record_id before training
            X_train = X_train.drop(columns=[self.record_id_column], errors='ignore')

            # persist in memory for better performance
            X_train = X_train.persist()
            y_train = y_train.persist()
            
            # Train the model
            self.logger.info("Started training")
            start_time = time.time()
            cl.fit(X_train, y_train, sample_weight=weights)
            end_time = time.time() 
            
            training_time = end_time - start_time
            self.logger.info(f"Model trained successfully in {str(timedelta(seconds=training_time))}")

            # get feature importance
            feature_importance_df = None
            if calc_feature_importance:
                self.logger.info(f"Calculating feature importance using {calc_feature_importance} method")
                if calc_feature_importance == 'built_in':
                    feature_importance_df = self.calculate_feature_importance(cl, X_train.compute())
                elif calc_feature_importance == 'permutation':
                    feature_importance_df = self.calculate_permutation_importance(cl, X_test, y_test)
                elif calc_feature_importance == 'shap':
                    feature_importance_df, _, _ = self.calculate_shap_importance(cl, X_test)
            
                self.logger.info(f"Feature importance ({calc_feature_importance}): \n{feature_importance_df}")

                # Visualize
                # plt.figure(figsize=(10, 6))
                # plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
                # plt.xlabel('Importance')
                # plt.title(f'Feature Importance ({calc_feature_importance})')
                # plt.tight_layout()
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise
        
        # Make predictions and evaluate
        try:
            # Get predictions and true values
            start_time = time.time()
            y_test, y_pred_proba = self.get_y_test_and_y_pred_proba(cl, X_test, highest_confidence)
            self.y_true = y_test
            self.pred_proba = y_pred_proba
            
            # Convert probabilities to binary predictions
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_score_0 = f1_score(y_test, y_pred, pos_label=0)
            f1_score_1 = f1_score(y_test, y_pred, pos_label=1)
            end_time = time.time() 

            inference_time = end_time - start_time
            self.logger.info(f"Evaluation metrics - Accuracy: {accuracy:.4f}, F1: {((f1_score_0+f1_score_1)/2):.4f} (in {str(timedelta(seconds=inference_time))})")
            
            return X_train, X_test, accuracy, f1_score_0, f1_score_1, training_time, inference_time, feature_importance_df
            
        except Exception as e:
            self.logger.error(f"Error during prediction or evaluation: {str(e)}")
            raise

    def calculate_feature_importance(self, model, X_train):
        # Get feature importance
        importance = model.feature_importances_
        
        # Create DataFrame for better visualization
        feature_names = X_train.columns
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        return feature_importance_df

    def calculate_permutation_importance(self, model, X_test, y_test):
        from sklearn.inspection import permutation_importance
        # Need to compute to numpy arrays for sklearn
        X_test_computed = X_test.compute() 
        y_test_computed = y_test.compute()
        
        # Calculate permutation importance
        perm_importance = permutation_importance(model, X_test_computed, y_test_computed, n_repeats=10)
        
        # Create DataFrame
        feature_names = X_test_computed.columns
        perm_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean
        }).sort_values(by='Importance', ascending=False)
        
        return perm_importance_df

    def calculate_shap_importance(self, model, X_test):
        import shap # TODO install shap
        
        # Compute X_test to numpy array
        X_test_computed = X_test.compute()
        
        # Create explainer and calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_computed)
        
        # Summarize feature importance based on SHAP
        shap_importance = np.abs(shap_values).mean(axis=0)
        feature_names = X_test_computed.columns
        
        shap_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': shap_importance
        }).sort_values(by='Importance', ascending=False)
        
        return shap_importance_df, explainer, shap_values

    
    def get_true_values_and_pred_proba(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return stored true labels and predicted probabilities"""
        if self.y_true is None or self.pred_proba is None:
            self.logger.warning("No predictions available. Train a model first.")
            return None, None
            
        return self.y_true, self.pred_proba

    def get_weights(self, X_train) -> Tuple[dd.DataFrame, dd.Series]:
        """Calculate inverse frequency weights based on record_id"""
        self.logger.info("Calculating sample weights for specialized entries")
        try:
            # Count occurrences of each record_id
            record_counts = X_train.groupby(self.record_id_column).size().reset_index()
            record_counts.columns = [self.record_id_column, 'count']
            
            # Merge counts back with data
            X_train = X_train.merge(record_counts, on=self.record_id_column, how='left')
            
            # Calculate weights as inverse of count
            X_train['weights'] = 1 / X_train['count']
            weights = X_train['weights']
            
            # Clean up temporary columns
            X_train = X_train.drop(columns=['count', 'weights'])
            return X_train, weights
            
        except Exception as e:
            self.logger.error(f"Error calculating weights: {str(e)}")
            raise

    def group_duplicates(self, X_train, y_train=None, weights=None):
        """Fixed two-step approach that avoids the KeyError"""
        
        self.logger.info("Aggregating duplicate rows with fixed approach")
        
        # Create a copy of the input data
        X = X_train.copy()
        
        # Add target and weights columns
        if y_train is not None:
            X['_target'] = y_train
        
        if weights is not None:
            X['_weight'] = weights
        else:
            X['_weight'] = 1
        
        # Handle record_id column if present
        record_id_col = None
        if self.record_id_column in X.columns:
            record_id_col = self.record_id_column
            X = X.drop(columns=[record_id_col])
        
        # Define columns to group by
        group_cols = [col for col in X.columns if col not in ['_weight']]
        
        # Better approach: Hash the group columns directly
        # This creates an immediately available column we can use for shuffling
        X['_group_id'] = dd.hash_pandas_object(X[group_cols], index=False)
        
        # Use the hash column for shuffling and aggregation
        result = (X.set_index('_group_id')  # Use set_index instead of shuffle
                  .map_partitions(lambda df: df.reset_index()
                                 .groupby('_group_id')
                                 .agg({col: 'first' for col in group_cols if col != '_group_id'}, 
                                      **{'_weight': 'sum'})
                                 .reset_index()))
        
        # Drop the temporary group ID column
        result = result.drop(columns=['_group_id'])
        
        # Extract target column if it exists
        if y_train is not None:
            dedup_y = result['_target']
            dedup_X = result.drop(columns=['_target', '_weight'])
        else:
            dedup_y = None
            dedup_X = result.drop(columns=['_weight'])
        
        dedup_weights = result['_weight']
        
        # Add back record_id if needed
        if record_id_col:
            # Generate new IDs - we'll create a simple index-based ID
            dedup_X = dedup_X.reset_index(drop=True)
            dedup_X[record_id_col] = dedup_X.index.map(lambda i: f"dedup_{i}")
        
        # For reporting only - estimate without triggering computation
        self.logger.info("Deduplication in process (actual counts computed after operation)")
        
        return dedup_X, dedup_weights, dedup_y



    # def group_duplicates(self, X_train, y_train=None, weights=None):
    #     """Deduplicate records by aggregating identical feature sets - optimized version"""
        
    #     self.logger.info("Aggregating duplicate rows and assigning weights")
    #     try:
    #         # Prepare data with targets and weights
    #         X = X_train.copy()

    #         # add needed data columns
    #         if y_train is not None:
    #             X['_target'] = y_train
                
    #         if weights is not None:
    #             X['_weight'] = weights
    #         else:
    #             X['_weight'] = 1
            
    #         # Calculate hash in chunks to avoid memory issues
    #         record_id_col = None
    #         if self.record_id_column in X.columns:
    #             record_id_col = self.record_id_column
    #             X = X.drop(columns=[record_id_col])
            
    #         # Define columns to group by - include target in grouping criteria
    #         group_cols = [col for col in X.columns if col not in ['_weight']]
            
    #         # 1. Modified hash and group approach to prevent memory explosion
    #         def hash_and_group(df):
    #             if len(df) == 0:
    #                 return df
                
    #             # Create a hash for grouping by all relevant columns using only numeric representation
    #             df['_hash'] = pd.util.hash_pandas_object(df[group_cols], index=False).astype('int64')
                
    #             # Group by hash with optimized aggregation
    #             result = df.groupby('_hash').agg({
    #                 '_weight': 'sum'
    #             }).reset_index()
                
    #             # Reattach the feature values using merge instead of drop_duplicates
    #             features_df = df[group_cols + ['_hash']].drop_duplicates('_hash')
    #             result = pd.merge(result, features_df, on='_hash', how='left')
                
    #             # Drop the hash column
    #             return result.drop(columns=['_hash'])
            
    #         # Apply the hash and group function to each partition
    #         result = X.map_partitions(hash_and_group)
                
    #         # 2. Instead of doing a second groupby, compute the result and perform the operation in pandas
    #         # This avoids the memory explosion issue
    #         self.logger.info("Computing deduplicated result")
    #         result_pd = result.compute()
            
    #         # 3. Do the final grouping in pandas instead of Dask
    #         self.logger.info(f"Final grouping of {len(result_pd)} rows")
    #         result_pd = result_pd.groupby(group_cols, observed=True)['_weight'].sum().reset_index()
                
    #         # Extract target column back if it exists
    #         if y_train is not None:
    #             dedup_y = result_pd['_target']
    #             dedup_X = result_pd.drop(columns=['_target', '_weight'])
    #         else:
    #             dedup_y = None
    #             dedup_X = result_pd.drop(columns=['_weight'])
                
    #         dedup_weights = result_pd['_weight']
            
    #         # Add back record_id if it was removed
    #         if record_id_col:
    #             dedup_X[self.record_id_column] = [f"dedup_{i}" for i in range(len(dedup_X))]

    #         # Convert back to Dask DataFrames
    #         dedup_X = dd.from_pandas(dedup_X, npartitions=X_train.npartitions)
    #         dedup_weights = dd.from_pandas(dedup_weights, npartitions=weights.npartitions)
    #         if dedup_y is not None:
    #             dedup_y = dd.from_pandas(dedup_y, npartitions=y_train.npartitions)
            
    #         # Log the reduction rate
    #         orig_len = len(X)
    #         new_len = len(dedup_X)
    #         self.logger.info(f"Deduplicated from ~{orig_len} to ~{new_len} rows "
    #                          f"({100 * (1 - new_len / orig_len):.2f}% reduction)")
            
    #         return dedup_X, dedup_weights, dedup_y
            
    #     except Exception as e:
    #         self.logger.error(f"Error aggregating duplicates: {str(e)}")
    #         raise

    
    def get_y_test_and_y_pred_proba(self, model, X_test, highest_confidence) -> Tuple[np.ndarray, np.ndarray]:
        """Get test labels and corresponding predictions"""
        self.logger.info("Getting predictions and test labels")
        
        try:
            # Get predictions by record ID
            predicted_values_by_record_id = self.get_predictions_by_record_ids(model, X_test, highest_confidence)
            
            # Merge with actual outcomes
            df_outcome = self.y_test_with_record_ids
            df_outcome = dd.merge(df_outcome, predicted_values_by_record_id, on=self.record_id_column)
            df_outcome = df_outcome.compute()
            
            # Extract labels and predictions
            y_test = df_outcome[self.label_column]
            y_pred_proba = df_outcome['predicted_values']
            
            return y_test, y_pred_proba
            
        except Exception as e:
            self.logger.error(f"Error getting test predictions: {str(e)}")
            raise

    def get_predictions_by_record_ids(self, model, X_test, highest_confidence) -> dd.DataFrame:
        """Generate and aggregate predictions by record_id"""
        self.logger.info(f"Generating predictions (highest_confidence={highest_confidence})")
        
        try:
            # Generate predictions (probability of positive class)
            X_test_features = X_test.drop(columns=[self.record_id_column])
            y_predicted = model.predict_proba(X_test_features)[:, 1]
            X_test['predicted_values'] = y_predicted
            
            # Aggregate predictions by record_id
            if highest_confidence:
                # Select prediction with highest confidence (furthest from 0.5)
                y_predicted_with_record_id = X_test[[self.record_id_column, 'predicted_values']].groupby([self.record_id_column])['predicted_values'].apply(
                    lambda x: max(x, key=lambda v: abs(v - 0.5))
                )
            else:
                # Take mean of predictions for each record_id
                y_predicted_with_record_id = X_test[[self.record_id_column, 'predicted_values']].groupby([self.record_id_column])['predicted_values'].mean()
            
            # Reset index to get record_id as a column
            y_predicted_with_record_id = y_predicted_with_record_id.reset_index()
            
            return y_predicted_with_record_id
            
        except Exception as e:
            self.logger.error(f"Error in prediction aggregation: {str(e)}")
            raise
            