import h5py
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve
from src.PreparingMethod import PreparingMethod
from src.DatasetManager import DatasetManager


class ResultsIOHandler:

    def __init__(self, save_dir: str, dataset: str, train_method: PreparingMethod, test_method: PreparingMethod, group_duplicates: bool, filter_by_record_id: bool, percentages: str = None):
        # Store configuration parameters including dataset name
        self.train_method_name = train_method.name if train_method else "original"
        self.test_method_name = test_method.name if test_method else "original"
        self.experiment_name = f"{self.train_method_name}_train_{self.test_method_name}_test" if not group_duplicates else f"{self.train_method_name}_train_{self.test_method_name}_test_group_duplicates"
        self.experiment_name = self.experiment_name if not filter_by_record_id else f"{self.experiment_name}_filter"
        
        # Create subdirectory for percentages if provided
        if percentages:
            save_dir = os.path.join(save_dir, percentages)
            os.makedirs(save_dir, exist_ok=True)
        
        self.file_path_h5 = os.path.join(save_dir, f"results_{self.experiment_name}.h5")
        self.file_path_csv = os.path.join(save_dir, f"results_{self.experiment_name}.csv")
        self.file_path_json = os.path.join(save_dir, f"results_{self.experiment_name}.json")
        self.last_id = self.get_last_id(self.file_path_csv)
        self.dataset = dataset  # Store the dataset name
        self.percentages = percentages  # Store the percentages
        self.experiment_config = None  # Will store the experiment configuration

    def set_experiment_config(self, config):
        """Set the experiment configuration dictionary"""
        self.experiment_config = config

    def get_last_id(self, file_path_csv):
        """Determine the next available ID by checking existing CSV file or starting fresh"""
        if os.path.exists(file_path_csv):
            results_df = pd.read_csv(file_path_csv, sep=';')
            return results_df['id'].max() + 1
        else:
            # Create directory if it doesn't exist
            outdir = os.path.dirname(file_path_csv)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            return 0

    def get_column_combination_string(self, anonymization):
        """Convert anonymization enum to string representation of columns"""
        if not anonymization.value:  # Empty list for "no" anonymization
            return "No anonymization"
            
        column_combination_string = ""
        for column in anonymization.value:
            column_combination_string += column.name + "_ANON, "
        # Remove trailing comma and space
        return column_combination_string[:-2]

    def save_model_results(self, anonymization, probas, true_labels, accuracy, f1_score_0, f1_score_1, training_time,
                           row_counts, group_duplicates, anonymization_ratios, slurm_id=None, h5_saving=False):
        """Save model evaluation results to CSV and optionally to H5"""
        # Get string representation of anonymized columns
        column_combination = self.get_column_combination_string(anonymization)

        # Save detailed data to H5 if enabled
        if h5_saving:
            with h5py.File(self.file_path_h5, 'a') as file:
                group = file.require_group(str(self.last_id))
                group.create_dataset('probas', data=probas)
                group.create_dataset('true_labels', data=true_labels)
                group.create_dataset('column_combination', data=column_combination)

        # Create dictionary with metrics
        result = {
            'id': str(self.last_id),
            'slurm_id': slurm_id,
            'dataset': self.dataset,
            'train_method': self.train_method_name,
            'test_method': self.test_method_name,
            'column_combination': column_combination,
            'accuracy': accuracy,
            'f1_score_class_0': f1_score_0,
            'f1_score_class_1': f1_score_1,
            'f1': (f1_score_0 + f1_score_1) / 2,  # calculation of average f1
            'training_time': training_time,
            'group_duplicates': group_duplicates,
            'row_counts': json.dumps(row_counts),
            'anonymization_ratios': json.dumps(anonymization_ratios),
            'anonymization': anonymization.name
        }
        self.write_result_to_csv(result)
        
        self.last_id += 1

    def write_result_to_csv(self, result):
        """Write evaluation result to CSV file, appending if file exists"""
        if os.path.exists(self.file_path_csv):
            # Append to existing CSV
            results_df = pd.read_csv(self.file_path_csv, sep=';')
            results_df = pd.concat([results_df, pd.DataFrame(result, index=[0])], ignore_index=True)
            results_df.to_csv(self.file_path_csv, index=False, sep=';')
        else:
            # Create new CSV
            result_df = pd.DataFrame(result, index=[0])
            result_df.to_csv(self.file_path_csv, index=False, sep=';')

    def save_experiment_config(self, config=None):
        """Save the experiment configuration as a JSON file"""
        if config is not None:
            self.experiment_config = config
            
        if self.experiment_config is None:
            print("Warning: No experiment configuration to save")
            return
            
        # Ensure the output directory exists
        outdir = os.path.dirname(self.file_path_json)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            
        # Convert any non-JSON serializable objects to strings
        def json_serialize(obj):
            try:
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                return str(obj)
                
        # Process the config to ensure JSON serializability
        def process_config(config_dict):
            if isinstance(config_dict, dict):
                return {k: process_config(v) for k, v in config_dict.items()}
            elif isinstance(config_dict, list):
                return [process_config(item) for item in config_dict]
            else:
                return json_serialize(config_dict)
                
        serializable_config = process_config(self.experiment_config)
        
        # Save the configuration to a JSON file
        with open(self.file_path_json, 'w') as f:
            json.dump(serializable_config, f, indent=4)
            
        print(f"Current state of experiment configuration saved to {self.file_path_json}")

    def load(self, key):
        """Load detailed results from H5 file by key"""
        with h5py.File(self.file_path_h5, 'r') as file:
            group = file[key]
            probas = group['probas'][:]
            true_labels = group['true_labels'][:]
            return {'probas': probas, 'true_labels': true_labels}
    
    def get_results(self):
        """Get combined results from CSV and H5 files"""
        results = pd.read_csv(self.file_path_csv, sep=';')
        results_dict = results.to_dict(orient='records')
        return results_dict

    def load_experiment_config(self):
        """Load an experiment configuration from JSON file"""
        if os.path.exists(self.file_path_json):
            with open(self.file_path_json, 'r') as f:
                return json.load(f)
        else:
            print(f"No experiment configuration file found at {self.file_path_json}")
            return None
    
    # def show_roc_curve(self, title=None):
    #     """Generate and display ROC curve for all anonymization levels"""
    #     results = self.get_results()

    #     plt.figure(figsize=(10, 8))
    #     for result in results:
    #         try:
    #             true_labels = result.get("true_labels")
    #             probas = result.get("probas")
    #             if true_labels is not None and probas is not None:
    #                 fpr, tpr, thresholds = roc_curve(true_labels, probas)
    #                 plt.plot(fpr, tpr, label=result["anonymization"])
    #         except (KeyError, TypeError) as e:
    #             print(f"Skipping ROC for {result.get('anonymization')}: {e}")

    #     plt.plot([0, 1], [0, 1], 'k--', label='Random')
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.legend(loc='best')
        
    #     if title is None:
    #         title = f"ROC Curve for {self.experiment_name}"
    #     plt.title(title)
    #     plt.grid(True, alpha=0.3)
    #     plt.show()
        
    # def show_probability_distribution(self):
    #     """Show probability distribution of model predictions for each anonymization level"""
    #     results = self.get_results()
        
    #     plt.figure(figsize=(12, 8))
    #     for result in results:
    #         try:
    #             probas = result.get("probas")
    #             if probas is not None:
    #                 sns.kdeplot(probas, label=result["anonymization"])
    #         except (KeyError, TypeError) as e:
    #             print(f"Skipping distribution for {result.get('anonymization')}: {e}")
            
    #     plt.legend(loc='best')
    #     plt.xlabel("Probability")
    #     plt.ylabel("Density")
    #     plt.title(f"Probability Distribution for {self.experiment_name}")
    #     plt.grid(True, alpha=0.3)
    #     plt.show()
    

def compare_train_test_matrix_csv(results_dir: str, dataset: str, train_methods: list, test_methods: list, group_duplicates: bool = False, 
                             anonymization_level=None, accuracy: bool = False):
    """
    Generate a matrix comparison showing results for combinations of train/test methods
    
    Args:
        dataset: Name of the dataset
        results_dir: Directory where results are stored
        train_methods: List of training methods to compare
        test_methods: List of testing methods to compare
        anonymization_level: Specific anonymization level to analyze (None = compare all)
        accuracy: If True, display accuracy values; otherwise display F1 scores
    """
    # Get dataset-specific anonymization class
    Anonymization = DatasetManager.get_anonymization_class(dataset)
    
    # If no specific anonymization level provided, use all levels
    if anonymization_level is None:
        anonymization_levels = [v for k, v in Anonymization.__dict__.items() if not k.startswith('_')]
    else:
        # Make sure we have a proper Anonymization enum
        if isinstance(anonymization_level, str):
            try:
                anonymization_level = getattr(Anonymization, anonymization_level)
            except AttributeError:
                raise ValueError(f"Invalid anonymization level: {anonymization_level}")
        anonymization_levels = [anonymization_level]
    
    # Create one matrix per anonymization level
    for anon_level in anonymization_levels:
        # Create results matrix
        results_matrix = pd.DataFrame(index=[m.value if hasattr(m, 'value') else str(m) for m in train_methods],
                                     columns=[m.value if hasattr(m, 'value') else str(m) for m in test_methods])
        
        # Fill matrix with results
        for train_method in train_methods:
            train_name = train_method.name if hasattr(train_method, 'name') else str(train_method)
            
            for test_method in test_methods:
                test_name = test_method.name if hasattr(test_method, 'name') else str(test_method)
                
                # Create experiment name
                experiment_name = f"{train_name}_train_{test_name}_test"
                if group_duplicates: experiment_name = experiment_name + "_group_duplicates"
                experiment_path = os.path.join(results_dir, f"results_{experiment_name}.csv")
                
                # Check if results exist
                if not os.path.exists(experiment_path):
                    results_matrix.loc[
                        train_method.value if hasattr(train_method, 'value') else str(train_method),
                        test_method.value if hasattr(test_method, 'value') else str(test_method)
                    ] = "---"
                    continue
                
                # Load results
                results_df = pd.read_csv(experiment_path, sep=';')
                
                # Find the entry for the current anonymization level
                result_row = results_df[results_df['anonymization'] == anon_level.name]
                
                if result_row.empty:
                    cell_value = "---"
                else:
                    # Get metric value based on parameter
                    if accuracy:
                        value = result_row['accuracy'].iloc[0]
                    else:
                        value = result_row['f1'].iloc[0]
                    
                    # Format with two decimal places
                    cell_value = f"{value:.2f}"
                
                # Add to matrix
                results_matrix.loc[
                    train_method.value if hasattr(train_method, 'value') else str(train_method),
                    test_method.value if hasattr(test_method, 'value') else str(test_method)
                ] = cell_value
        
        # Plot matrix as table
        fig, ax = plt.subplots(figsize=(len(test_methods) * 1.5 + 2, len(train_methods) + 2))
        ax.axis('off')
        
        # Create table with row and column headers
        table = ax.table(
            cellText=results_matrix.values,
            rowLabels=results_matrix.index,
            colLabels=results_matrix.columns,
            cellLoc='center',
            loc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.2)
        
        # Add better titles and styling
        metric = "Accuracy" if accuracy else "F1 Score"
        plt.title(f"{metric} for {anon_level.name} Anonymization\n(Train Methods × Test Methods)")
        
        plt.tight_layout()
        plt.show()
        

def compare_train_test_matrix_ext(results_dir: str, dataset: str, train_methods: list, test_methods: list,
                             anonymization_level=None, accuracy: bool = False):
    """
    Generate a consolidated matrix comparison showing results for combinations of train/test methods
    
    Args:
        dataset: Name of the dataset
        results_dir: Directory where results are stored
        train_methods: List of training methods to compare
        test_methods: List of testing methods to compare
        anonymization_level: Specific anonymization level to analyze (None = compare all)
        accuracy: If True, display accuracy values; otherwise display F1 scores
    """
    # Get dataset-specific anonymization class
    Anonymization = DatasetManager.get_anonymization_class(dataset)
    
    # If no specific anonymization level provided, use all levels
    if anonymization_level is None:
        anonymization_levels = [v for k, v in Anonymization.__dict__.items() if not k.startswith('_')]
    else:
        # Make sure we have a proper Anonymization enum
        if isinstance(anonymization_level, str):
            try:
                anonymization_level = getattr(Anonymization, anonymization_level)
            except AttributeError:
                raise ValueError(f"Invalid anonymization level: {anonymization_level}")
        anonymization_levels = [anonymization_level]
    
    # Process each anonymization levelƒ
    for anon_level in anonymization_levels:
        # Dictionary to store results: (train_method, test_method, filter_n, filter_mode) -> value
        results = {}
        # Set to keep track of unique filter configurations
        filter_configs = set()
        
        # Process all combinations of train and test methods
        for train_method in train_methods:
            train_name = train_method.name if hasattr(train_method, 'name') else str(train_method)
            
            for test_method in test_methods:
                test_name = test_method.name if hasattr(test_method, 'name') else str(test_method)
                
                # Create base experiment name
                base_experiment_name = f"{train_name}_train_{test_name}_test"
                
                # 1. Process base results from JSON file
                json_path = os.path.join(results_dir, f"results_{base_experiment_name}.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        experiment_data = json.load(f)
                    
                    # Find results for the current anonymization level
                    for anon_result in experiment_data.get('anonymization_results', []):
                        if anon_result.get('level') == anon_level.name:
                            metrics = anon_result.get('metrics', {})
                            if metrics:
                                value = metrics.get('accuracy', 0) if accuracy else metrics.get('f1_score_avg', 0)
                                results[(train_name, test_name, None, None)] = value

                # 2. Process filter results from filter JSON file
                filter_json_path = os.path.join(results_dir, f"results_{base_experiment_name}_filter.json")
                if os.path.exists(filter_json_path):
                    with open(filter_json_path, 'r') as f:
                        filter_data = json.load(f)

                    # Find results for the current anonymization level
                    for anon_result in filter_data.get('anonymization_results', []):
                        if anon_result.get('level') == anon_level.name:
                            filtering_results = anon_result.get('filtering', {})
                            for filter_result in filtering_results.get('results', []):
                                n_duplicates = filter_result.get('n_duplicates')
                                mode = filter_result.get('mode')
                                metrics = filter_result.get('metrics', {})
                                
                                if metrics:
                                    value = metrics.get('accuracy', 0) if accuracy else metrics.get('f1_score_avg', 0)
                                    results[(train_name, test_name, n_duplicates, mode)] = value
                                    filter_configs.add((n_duplicates, mode))
        
        # If we have no results, skip this anonymization level
        if not results:
            print(f"No results found for anonymization level {anon_level.name}")
            continue
        
        # Sort filter configurations for consistent display
        filter_configs = sorted(list(filter_configs))
        has_base_results = any(key[2] is None for key in results.keys())
        
        # Create a master figure with subplots for base and each filter configuration
        num_cols = (1 if has_base_results else 0) + len(filter_configs)
        
        if num_cols == 0:
            print(f"No valid configurations found for anonymization level {anon_level.name}")
            continue
        
        # Create figure with appropriate size
        fig_width = max(12, num_cols * 4)  # Ensure minimum width while scaling with # of matrices
        fig_height = 8  # Fixed height
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Add overall title
        metric_name = "Accuracy" if accuracy else "F1 Score"
        fig.suptitle(f"{dataset} - {metric_name} Results for {anon_level.name} Anonymization", 
                    fontsize=16, fontweight='bold')
        
        # Create grid for subplots
        grid = plt.GridSpec(1, num_cols, figure=fig, wspace=1.5)
        
        # Function to create and populate a single matrix subplot
        def create_matrix_subplot(ax, filter_config=None):
            # Create DataFrame to hold the results matrix
            matrix = pd.DataFrame(
                index=[m.value if hasattr(m, 'value') else str(m) for m in train_methods],
                columns=[m.value if hasattr(m, 'value') else str(m) for m in test_methods]
            )
            
            # Fill the matrix with values from the results
            for train_method in train_methods:
                train_name = train_method.name if hasattr(train_method, 'name') else str(train_method)
                train_label = train_method.value if hasattr(train_method, 'value') else str(train_method)
                
                for test_method in test_methods:
                    test_name = test_method.name if hasattr(test_method, 'name') else str(test_method)
                    test_label = test_method.value if hasattr(test_method, 'value') else str(test_method)
                    
                    # Retrieve the value from results dictionary
                    if filter_config is None:
                        # Base results
                        key = (train_name, test_name, None, None)
                    else:
                        # Filter results
                        key = (train_name, test_name, filter_config[0], filter_config[1])
                    
                    value = results.get(key)
                    if value is not None:
                        matrix.at[train_label, test_label] = f"{value:.2f}"
                    else:
                        matrix.at[train_label, test_label] = "---"
            
            # Turn off axes for the plot
            ax.axis('off')
            
            # Create and style the table
            table = ax.table(
                cellText=matrix.values,
                rowLabels=matrix.index,
                colLabels=matrix.columns,
                cellLoc='center',
                loc='center'
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            
            # Set cell colors based on values (only for numeric cells)
            for i in range(len(matrix.index)):
                for j in range(len(matrix.columns)):
                    cell = table[(i+1, j)]  # +1 because first row is header
                    value_str = matrix.iloc[i, j]
                    try:
                        value = float(value_str)
                        # Apply color scale: higher values = better = more green
                        intensity = min(1.0, max(0.0, value))  # Ensure between 0 and 1
                        cell.set_facecolor((1.0 - intensity * 0.5, 1.0, 1.0 - intensity * 0.5))
                    except ValueError:
                        # Not a numeric cell, leave default color
                        pass
            
            # Add subtitle for the matrix
            if filter_config is None:
                title = "Base Results"
            else:
                n_duplicates, mode = filter_config
                if mode == 'unique':
                    title = "Unique Records"
                else:
                    title = f"{mode.capitalize()}, n={n_duplicates}"
            
            ax.set_title(title)
            
        # Create subplots for each configuration
        col_idx = 0
        
        # First add base results if they exist
        if has_base_results:
            ax_base = fig.add_subplot(grid[0, col_idx])
            create_matrix_subplot(ax_base, filter_config=None)
            col_idx += 1
        
        # Then add filter configurations
        for filter_config in filter_configs:
            ax_filter = fig.add_subplot(grid[0, col_idx])
            create_matrix_subplot(ax_filter, filter_config=filter_config)
            col_idx += 1
        
        # Adjust layout
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for the title
        plt.show()


