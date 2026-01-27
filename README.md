# User-Driven Privacy Research Project

This repository contains the codebase for the paper:

**[Paper Title]**  
[Authors]  
arXiv: [LINK]

This project investigates the impact of different data preparation techniques on machine learning model performance in the context of user-driven data privacy. The core idea is to simulate scenarios where users can decide the level of privacy for their data, resulting in datasets with a mix of original, generalized, and missing values. We explore both traditional anonymization techniques and LLM-based approaches for data imputation and generation.

## Reference Data

The datasets and results used in our experiments are publicly available:

- **Anonymized datasets**: [LINK]
- **LLM results and imputed datasets**: [LINK]

Alternatively, you can generate the datasets yourself by following the instructions below.

## Project Structure

-   `run.py`: The main script to run a single experiment.
-   `run_all.sh`: A shell script to automate running batches of experiments with different configurations.
-   `run.job`: A Slurm job script called by `run_all.sh` to submit jobs to a Slurm cluster.
-   `src/`: Contains the core Python source code for all pipeline steps.
    -   `DatasetCreation.py`: Creates dataset versions based on specified privacy preferences (percentages of original, generalized, and missing data).
    -   `PreparingMethod.py`: Defines data preparation methods applied in `DataLoader.py`, including generalization and specialization techniques.
    -   `Main.py`: Contains the main evaluation loop (`run_evaluation`) for model training and testing.
    -   Additional modules for data loading, LLM-based imputation, model evaluation, etc.
-   `data/`: Input datasets (e.g., `adult`, `diabetes`).
-   `results/`: Run-specific subfolders containing experiment results.
-   `results_collected/`: Aggregated final results organized by dataset and distribution.
-   `auswertung.ipynb`: Jupyter notebook for analyzing and visualizing results.

## Environment Setup
We used and recommend using a Slurm + Dask combination on a cluster for the general and LLM experiments.
Local mode is ok for creating anonymized datasets, local LLM setups, and testing.

### Requirements

-   **Slurm**: For distributed job scheduling (required for batch experiments)
-   **Dask**: For parallel processing (not required for anonymized dataset generation and LLM outputs)
-   **Python 3.x**: With dependencies listed in requirements files

### Installation

The project provides different requirement files for different use cases:

-   **`requirements_local.txt`**: For local development (Python 3.9.6, includes LLM packages)
-   **`requirements_cluster.txt`**: For runs in cluster environment (without LLM packages, as they're loaded via their own job)
-   **`requirements_llm.txt`**: Only packages needed for LLM experiments (Python 3.12.3)
- For Slurm runs on a cluster, we load the needed packages and module in the two job files, adjust this for your environment.

**Install dependencies:**

```bash
# For local development
pip install -r requirements-local.txt

# For cluster (additional packages loaded via job scripts)
pip install -r requirements-cluster.txt

# For LLM evaluation (in addition to local/cluster requirements)
pip install -r requirements_llm.txt
```

## Instructions

This section covers how to run specific parts of the experiments and how to extend the codebase with new functionality.

### 1. Generating Anonymized Datasets

The dataset creation process generates anonymized versions of the input datasets with varying levels of privacy. This includes:

-   **Anonymized datasets** with specified distributions of original, generalized, and missing data
-   **Prepared datasets** for generalization and specialization methods

**Note on specialization outputs**: Earlier versions generated per-column files for specialization. The codebase has since shifted to record-based in-memory computation for efficiency. The column-based files are kept for reference to illustrate the output column data from specialization.

**Run dataset generation**:

```bash
python src/DatasetCreation.py \
    --dataset adult \
    --data_dir ./data \
    --percentages "0.5 0.3 0.2"
```

### 2. Generating LLM-Based Outputs

The project includes LLM-based approaches for data imputation and generation. These outputs can be used to compare traditional anonymization techniques with modern generative approaches.

#### Prerequisites

**Environment Configuration:**

1. Configure the LLM API settings in `.env_llm` (in project root):
   ```bash
   LLM_API_BASE_URL="..."
   LLM_API_KEY="..."
   LLM_MODEL="..."
   ```

2. Install LLM-specific dependencies or check cluster dependencies in job file:
   ```bash
   pip install -r requirements_llm.txt
   ```

3. Ensure anonymized datasets exist (from Step 1 or downloaded from Zenodo)

**What LLM evaluation does:**

For each dataset's training and test sets:
1. **Value Imputation**: Predicts missing/generalized values and creates imputed dataset versions (saved as CSV)
2. **Label Prediction**: Directly predicts target labels from anonymized records (results saved as CSV)

Results are saved in `llm_evaluation/{percentage}_results/` directories. When finished, rename to `llm_evaluation/{percentage}/` for use in model evaluation.

#### Running LLM Evaluation

**Option 1: Interactive Shell Wrapper (Recommended)**

```bash
# Single percentage, multiple datasets
bash llm_runs.sh launch "33-33-34" "Adult-Diabetes-German" 1 10

# Multiple percentages, with partitioning for large datasets
bash llm_runs.sh launch "33-33-34,66-17-17" "Employment" 3 10

# Check job status
bash llm_runs.sh status

# Merge partitioned results after completion
bash llm_runs.sh merge "33-33-34" "Employment" 3
```

**Option 2: Direct SLURM Submission**

```bash
sbatch --export=ALL,PERCENTAGE=33-33-34,DATASETS=Adult-Diabetes-German,BATCH_SIZE=10,INPUT_DIR=./data,RESULTS_BASE=llm_evaluation llm.job
```

**Option 3: Python Script Directly**
Collecting and Analyzing Results

After running experiments, gather the final results from the `results/` folder into a `results_collected/` folder. The collection script organizes results following the structure: `dataset -> distribution -> json files`.

**Run results collection**:

```bash
cd evaluation
python generate_results.py
```

This will aggregate all experiment outputs and organize them for analysis.LM Evaluation Scripts:**
- `llm_runs.sh`: Shell wrapper for launching multiple LLM-related jobs with extra options
- `llm.job`: SLURM job definition file
- `llm_src/llm_evaluation.py`: Main LLM evaluation script
- `llm_src/llm_evaluation_test.ipynb`: Jupyter notebook for small test runs
- `llm_src/llm_evaluation_launcher.py`: Job launcher with partitioning support
- `llm_src/llm_evaluation_merger.py`: Merges results from partitioned runs
- `llm_src/merge_llm_results.py`: Utility for merging partitioned results

### 3. Overall Evaluation Runs

Run the complete evaluation pipeline to train and test models on prepared datasets. Results are automatically saved to run-specific subfolders in the `results/` directory.

#### Running a Single Experiment

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

#### Running Batch Experiments

To run a comprehensive set of experiments, you can configure and use the `run_all.sh` script.

1.  **Configure `run_all.sh`**: Open the `run_all.sh` script and modify the arrays (`datasets`, `percentages_list`, `train_methods`, `test_methods`, etc.) to define the parameter space for your experiments.

2.  **Execute the script**:
    ```bash
    ./run_all.sh
    ```
    This will start submitting jobs to the Slurm cluster based on the configurations. Each job will execute the `run.py` script with a different combination of parameters.

### 4. Generating Results

After running experiments, gather the final results from the `results/` folder into a `results_collected/` folder. The collection script organizes results following the structure: `dataset -> distribution -> json files`.

**Run results collection**:

```bash
# Command to collect and organize results
# [Add specific script and parameters]
```

## Extending the Codebase

### Adding a New Dataset

To a**Create column specifications**: Add a new class in `src/spalten/` folder (following the pattern of existing datasets) that defines generalizations for each column. Import the class in `src/spalten/__init__.py`.

2.  **Configure dataset download**: Add dataset information to `download_dataset_if_missing()` in `src/DatasetCreation.py` for automatic download, or add a custom download function (like for Employment dataset), or manually place files in the `data/` folder.

3.  **Add dataset configuration**: Create a config entry in `src/DatasetManager.py` with dataset information, attributes, and anonymization levels.

4.  **Implement data cleaning**: Review `clean_and_split_data()` in `src/Vorverarbeitung.py` and add any dataset-specific cleaning or splitting logic required.

5.  **Update batch scripts**: Add the dataset name to the `datasets` array in `run_all.sh` for batch processing.
Additional Notes

### Directory Structure Reference

**Data Directories:**
- `data/`: Raw and processed datasets
  - `{dataset}/`: Original dataset files
  - `{dataset}/generalization/{percentage}/`: Anonymized datasets for each privacy distribution
  - `{dataset}/specialization/`: Specialization intermediate files (reference only)
- `llm_evaluation/`: LLM evaluation outputs
  - `{percentage}_results/`: Active evaluation results (rename to `{percentage}/` when complete)
  - `{percentage}/`: Final results used by model evaluation
- `llm_slurm_logs/`: SLURM job logs for LLM experiments

**Results Directories:**
- `results/`: Individual experiment run outputs
- `results_collected/`: Aggregated results organized by dataset and distribution

**Source Code:**
- `src/`: Core pipeline modules
- `src/spalten/`: Dataset column specifications and generalization hierarchies
- `llm_src/`: LLM evaluation scripts
- `evaluation/`: Results collection and analysis scripts

### Working with LLM Results

LLM evaluation creates two types of outputs per dataset split:

1. **Imputation results**: `{dataset}_{split}_imputation_results.csv` - Evaluation of how well the LLM can recover original values
2. **Imputed datasets**: `{dataset}_{split}_imputed_dataset.csv` - Complete datasets with LLM-imputed values
3. **Prediction results**: `{dataset}_{split}_prediction_results.csv` - Direct label prediction from anonymized data

These outputs are used in the main evaluation pipeline by setting the training/test methods to `llm_imputation` and ensuring the results are in `llm_evaluation/{percentage}/`.

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{...,
  title={...},
  author={...},
  journal={arXiv preprint arXiv:...},
  year={2025}
}
```

add a dataset
- spalten class: add a new class for the dataset to the src/spalten/ folder like the others that gives generalizations and add the class to the __init__
- datasetcreation: add the dataset infos to download_dataset_if_missing for automatic download (or special function like for employment or add files to data folder by hand) 
- datasetmanager: add a matching config with infos, attributes, anonymization levels
- vorverarbeitung: look at the function clean_and_split_data and see if the dataset needs special cleaning or splitting for common usage

add a privacy distribution
- just run experiments or dataset creation with the percentage and everything will be created accordingly
- for llm extra runs are needed for imputation and prediction results

requirements_local: Python/3.9.6 packages needed for the project locally (cluster dependencies instead loaded and installed in job script)
requirements_cluster: package list used on the cluster but without required llm packages (as loaded and installed on the cluster from the job script)



llm:
_.env_llm: fill this configuration file before running and then rename to just .env_llm. Add following info:
    LLM_API_BASE_URL="..."
    LLM_API_KEY="..."
    LLM_MODEL="..."
Pre-requisite: the original and anonymized datasets need to be stored already for use by the llm. either by downloading from zenodo or running the creation script
given a dataset llm evaluation does two things to each the training and test sets:
1. impute anonymized values and create resulting imputed dataset version (both steps saved as csv)
2. directly predict the label of anonymized records (results saved as csv)
result folders will be saved named as percentage_results to not overwrite existing results. when finished rename folder to only percentage as name for later use

LLM Evaluation Scripts (run from project root):
- llm_runs.sh: Launch multiple individual jobs with extra options
- llm.job: SLURM job file for direct submission
- llm_src/llm_evaluation_launcher.py: Starts jobs with extra functionality like partitioning
- llm_src/llm_evaluation_merger.py: Takes finished partitioned runs and puts them together
- llm_src/llm_evaluation.py: Main script for llm experiments
- llm_src/llm_evaluation_test.ipynb: Notebook for smaller test runs
- llm_src/merge_llm_results.py: Utility to merge partitioned results
- .env_llm: Environment variables (LLM API URL, key, model)
- requirements_llm.txt: Python packages needed for the llm run (assuming datasets are present)

