# User-Driven Privacy Research Project

This project is designed to conduct experiments on the impact of different data preparation techniques on machine learning model performance, particularly in the context of user-driven data privacy. The core idea is to simulate scenarios where users can decide the level of privacy for their data, resulting in a dataset with a mix of original, generalized, and missing values.

The project is structured to automate the process of dataset creation, model training, evaluation, and results aggregation.

## Project Structure

-   `run.py`: The main script to run a single experiment.
-   `run_all.sh`: A shell script to automate running a batch of experiments with different configurations.
-   `job.sh`: A Slurm job script that is called by `run_all.sh` to submit jobs to a Slurm cluster.
-   `src/`: Contains the core Python source code for all steps of the pipeline.
    -   `DatasetCreation.py`: Handles the creation of dataset versions based on specified privacy preferences (percentages of original, generalized, and missing data).
    -   `PreparingMethod.py`: Defines the different data preparation methods that can be applied in `DataLoader.py`.
    -   `Main.py`: Contains the main evaluation loop (`run_evaluation`) which trains and tests the model.
    -   Other modules for data loading, model evaluation, etc.
-   `data/`: Should contain the input datasets (e.g., `adult`, `diabetes`).
-   `results/`: The default directory where the results of the experiments are stored.
-   `auswertung.ipynb`: A Jupyter notebook to analyze and visualize the results from the `results` directory.

## Workflow

The main workflow consists of the following steps:

1.  **Dataset Version Creation**: Based on a given set of percentages, a version of the dataset is created that contains a mix of original, generalized, and missing data points. This is handled by `src/DatasetCreation.py`.

2.  **Data Preparation**: The training and test sets are preprocessed using one of the specified `PreparingMethod`s. These methods define how to handle the mixed-privacy data in `DataLoader.py`.

3.  **Model Training and Evaluation**: A machine learning model is trained on the prepared training data and evaluated on the prepared test data. The performance metrics (like F1-score) are calculated. This is orchestrated by `src/Main.py`.

4.  **Running Experiments**: The `run.py` script is the main entry point to run a single experiment with a specific configuration. To run a series of experiments, the `run_all.sh` script can be used, which systematically iterates through different datasets, preparation methods, and privacy settings, submitting each as a job to a Slurm cluster via `job.sh`.

5.  **Results Analysis**: The results of all experiments are saved in the `results/` directory. The `auswertung.ipynb` notebook can be used to load, aggregate, and visualize these results to draw conclusions.

## How to Run

### Running a Single Experiment

You can run a single experiment using the `run.py` script. It requires several command-line arguments to specify the configuration.

**Arguments for `run.py`:**

-   `--save_dir`: Directory to save results.
-   `--data_dir`: Base directory containing datasets.
-   `--dataset`: Name of the dataset to use (e.g., `adult`).
-   `--train_method`: The preparation method for the training data (e.g., `weighted_specialization`).
-   `--test_method`: The preparation method for the test data.
-   `--percentages`: A string of three space-separated float values for the dataset split (original, generalized, missing), e.g., `"0.6 0.2 0.2"`.
-   `--n_workers`: The number of Dask workers to use.
-   `--group_duplicates`: (Optional) A flag to enable deduplication of records.
-   `--filter_by_record_id`: (Optional) A flag to enable filtering by record ID, relevant for specialization methods.
-   `--use_gpu`: (Optional) A flag to enable GPU acceleration.

**Example command:**

```bash
python run.py \
    --save_dir ./results/my_experiment \
    --data_dir ./data \
    --dataset adult \
    --train_method weighted_specialization \
    --test_method weighted_specialization \
    --percentages "0.5 0.3 0.2" \
    --n_workers 4
```

### Running Batch Experiments

To run a comprehensive set of experiments, you can configure and use the `run_all.sh` script.

1.  **Configure `run_all.sh`**: Open the `run_all.sh` script and modify the arrays (`datasets`, `percentages_list`, `train_methods`, `test_methods`, etc.) to define the parameter space for your experiments.

2.  **Execute the script**:
    ```bash
    ./run_all.sh
    ```
    This will start submitting jobs to the Slurm cluster based on the configurations. Each job will execute the `run.py` script with a different combination of parameters.

## Analyzing Results

The results of each run are stored in the directory specified by `--save_dir`. After the experiments are complete, you can use the `auswertung.ipynb` notebook to analyze the output. The notebook will typically read the result files (e.g., CSVs with F1-scores), aggregate them, and generate plots and tables for comparison.