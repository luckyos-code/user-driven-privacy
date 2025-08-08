import os
import numpy as np
import random
from src.DataLoader import DataLoader
from src.ModelEvaluator import ModelEvaluator
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from src.ResultsIOHandler import ResultsIOHandler
from src.PreparingMethod import PreparingMethod
from src.DatasetManager import DatasetManager
from src.FilteringHandler import FilteringHandler
import time
from datetime import timedelta

# Set global seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def run_evaluation(n_workers: int = 1, use_gpu: bool = False, save_dir: str = None, data_dir: str = None,
                dataset: str = None, train_method: PreparingMethod = None, test_method: PreparingMethod = None,
                group_duplicates: bool = False, filter_by_record_id: bool = False, random_seed: int = 42, percentages: str = None):
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
    """
    # Set global seeds
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Start timing the full experiment
    experiment_start_time = time.time()

    # Create experiment configuration dictionary to track all settings and results
    experiment_config = {
        # Basic configuration
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "dataset": dataset,
        "training_method": train_method.name if train_method else "original",
        "testing_method": test_method.name if test_method else "original",
        "group_duplicates": group_duplicates,
        "filter_by_record_id": filter_by_record_id,
        "random_seed": random_seed,
        "percentages": percentages,
        
        # Hardware configuration
        "n_workers": int(n_workers),
        "use_gpu": use_gpu,
        "cores": 8,
        "memory": '128GB',
        "queue": 'paul',
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
                             random_seed=random_seed)
    client = Client(cluster)
    print("Dask cluster dashboard:", client.dashboard_link)
    experiment_config["dask_dashboard_link"] = client.dashboard_link
    
    # Initialize results handler with experiment config
    results_io_handler = ResultsIOHandler(save_dir, dataset, train_method, test_method, group_duplicates, filter_by_record_id)
    results_io_handler.set_experiment_config(experiment_config)

    # Determine if we're using weighted methods
    weighted = any(method and "weighted" in method.name 
                for method in [train_method, test_method] if method)
    experiment_config["weighted_methods"] = weighted

    # Determine if highest_confidence should be used
    absolute_results = any(method and "highest_confidence" in method.name
                        for method in [train_method, test_method] if method)
    experiment_config["absolute_results"] = absolute_results

    # Get dataset-specific anonymization enum
    Anonymization = DatasetManager.get_anonymization_class(dataset)
    spalten_dict, spalten_list = DatasetManager.get_spalten_classes(dataset)
    
    # Run evaluation for each anonymization level
    for anonymization in Anonymization:
        # TODO testing
        if anonymization.name not in ["full"]: continue
        
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
        data_loader.preprocess(columns, train_method, test_method)
     
        # Get processed data
        data_train, data_test = data_loader.get_data()

        # track original row counts right after loading
        original_train_count = data_train.shape[0].compute()
        original_test_count = data_test.shape[0].compute()
        
        anonymization_config["row_counts_original"] = {
            "train": int(original_train_count),
            "test": int(original_test_count)
        }

        print("Row counts original: "+str(anonymization_config["row_counts_original"]))
        
        # Record preprocessing time
        preprocessing_time = time.time() - preprocessing_start_time
        anonymization_config["preprocessing_time"] = preprocessing_time
        print(f"Preprocessing finished in {timedelta(seconds=preprocessing_time)}")

        # apply filtering for specialized entries if wanted
        if (any(method and "specialization" in method.name
            for method in [train_method] if method) # TODO also for test?!
                and filter_by_record_id):
            record_id_column = DatasetManager.get_record_id_column(dataset)

            # Define all configurations you want to try
            if anonymization.name == 'no':
                filtering_configs = [(0, None)]
            else:
                filtering_configs = [
                    (0, None),       # Keep only unique records
                    
                    (1, 'random'),   # Keep 1 random duplicate
                    (3, 'random'),   # Keep 3 random duplicates
                    (5, 'random'),   # Keep 5 random duplicates
                    
                    (1, 'imputation'),  # Keep 1 duplicate by imputation
                    (3, 'imputation'),  # Keep 3 duplicates by imputation
                    (5, 'imputation'),  # Keep 5 duplicates by imputation
                    
                    (1, 'knn'),      # Keep 1 duplicate using KNN similarity
                    (3, 'knn'),      # Keep 3 duplicates using KNN similarity
                    (5, 'knn'),      # Keep 5 duplicates using KNN similarity
                    
                    (1, 'autoencoder'),  # Keep 1 duplicate using autoencoder reconstruction
                    (3, 'autoencoder'),  # Keep 3 duplicates using autoencoder reconstruction
                    (5, 'autoencoder'),  # Keep 5 duplicates using autoencoder reconstruction
                ]
            
            # Track filtering configurations in experiment config
            anonymization_config["filtering"] = {
                "enabled": True,
                "configs": [{"n_duplicates": n, "mode": mode} for n, mode in filtering_configs],
                "results": []
            }
           
            # Get filtered datasets for all configurations using the FilteringHandler
            filtering_start_time = time.time()
            filtered_datasets = FilteringHandler.filter_specialized_data(
                data_train, record_id_column, filtering_configs, random_seed
            )
            filtering_time = time.time() - filtering_start_time
            anonymization_config["filtering_time"] = filtering_time
            print(f"Filtering finished in {timedelta(seconds=filtering_time)}")
           
            # Now you can use each filtered dataset as needed
            for (n_duplicates, mode), filtered_data in filtered_datasets.items():
                # Create a config for this specific filtering configuration
                filtering_result_config = {
                    "n_duplicates": n_duplicates,
                    "mode": mode or "unique",
                    "metrics": {},
                    "row_counts_used": {}
                }

                print(f"Filtered run (n_duplicates, mode, #filtered_data): {n_duplicates}, {mode}, {filtered_data.shape[0].compute()}")
                
                # Run your model evaluation with this filtered dataset
                evaluate_model(
                    dataset, filtered_data, data_test, client, weighted=weighted,
                    absolute_results=absolute_results, group_duplicates=group_duplicates,
                    use_gpu=use_gpu, config=filtering_result_config, random_seed=random_seed
                )
                
                # Add filtering result to the anonymization config
                anonymization_config["filtering"]["results"].append(filtering_result_config)
        else:
            # Run evaluation and save results without filtering
            anonymization_config["filtering"] = {"enabled": False}
            evaluate_model(
                dataset, data_train, data_test, client, weighted=weighted, absolute_results=absolute_results,
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
    
def create_cluster(n_workers: int = 1, use_gpu: bool = False, cores: int = 8, memory: str = '64GB', queue: str = 'clara', random_seed: int = 42):
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
            job_extra.append("--gres=gpu:rtx2080ti:1")  # Request 1 GPU per job, change to correct type
            print("GPU support enabled: Requesting 1 GPU per job.")

        # Gave all worker jobs inherit the original job's ID as an environment variable
        job_extra.append(f"--export=ALL,PARENT_SLURM_JOB_ID={job_id}")
     
        cluster = SLURMCluster(
            cores=cores,
            memory=memory,
            processes=1,
            walltime='48:00:00',  # Adjust based on your cluster's limits
            queue=queue,  # Change to the correct SLURM queue
            job_extra=job_extra,
            log_directory=f'slurm_logs/{job_id}/dask_logs',
            local_directory='dask-worker-space'
        )
        cluster.scale(jobs=n_workers)  # Scale number of jobs to workers
        return cluster
    else:
        print("SLURM environment not detected. Falling back to LocalCluster.")
        return LocalCluster(n_workers=n_workers, threads_per_worker=cores, memory_limit=memory)


def evaluate_model(dataset, data_train, data_test, client, weighted=False, absolute_results=False,
                   group_duplicates=False, use_gpu=False, config=None, random_seed=42):
    """
    Evaluate model performance and save results for a specific configuration.
    
    Args:
        dataset: Name of the dataset
        data_train: Training data Dask DataFrame
        data_test: Test data Dask DataFrame
        client: Dask client
        weighted: Whether to use weighted samples
        absolute_results: Whether to use highest confidence predictions
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
    
    # Get record_id column for the dataset
    record_id_col = DatasetManager.get_record_id_column(dataset)
    y_test_with_record_ids = data_test[[record_id_col, label_column]]

    # Persist the feature and target dataframes for better performance during model training
    X_train = X_train.persist()
    Y_train = Y_train.persist()
    X_test = X_test.persist()
    y_test_with_record_ids = y_test_with_record_ids.drop_duplicates().persist()
    
    # Initialize and run model evaluation
    model_evaluator = ModelEvaluator(X_train, X_test, Y_train, y_test_with_record_ids, client, dataset)
    _X_train, _X_test, accuracy, f1_score_0, f1_score_1, training_time, inference_time, feature_importance_df = model_evaluator.train_model(
        weighted, absolute_results, group_duplicates, use_gpu
    )
    
    # Get prediction results
    true_labels, pred_proba = model_evaluator.get_true_values_and_pred_proba()
    
    # Store metrics
    metrics = {
        "accuracy": float(accuracy),
        "f1_score_0": float(f1_score_0),
        "f1_score_1": float(f1_score_1),
        "f1_score_avg": float((f1_score_0 + f1_score_1) / 2),
        "training_time": float(training_time),
        "inference_time": float(inference_time),
        "feature_importance": feature_importance_df.to_dict(orient='records'),
    }
    
    # Update config
    config["metrics"] = metrics

    # track used row counts right after training
    used_train_count = _X_train.shape[0].compute()
    used_test_count = _X_test.shape[0].compute()
    
    config["row_counts_used"] = {
        "train": int(used_train_count),
        "test": int(used_test_count)
    }

    print("Row counts used: "+str(config["row_counts_used"]))
    
    # Return metrics for potential further use
    return metrics
