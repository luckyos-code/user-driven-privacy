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
-   `job.sh`: A Slurm job script called by `run_all.sh` to submit jobs to a Slurm cluster.
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

### Requirements

-   **Slurm**: For distributed job scheduling (required for batch experiments)
-   **Dask**: For parallel processing (not required for anonymized dataset generation and LLM outputs)
-   **Python 3.x**: With dependencies listed in requirements file

### Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
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

**Run LLM-based generation**:

```bash
# Command for LLM output generation
# [Add specific script and parameters]
```

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

To add a new dataset to the experiments:

1.  Place the dataset files in the `data/` directory following the expected format
2.  Update the dataset configuration in the relevant scripts
3.  Ensure the dataset has the required columns and preprocessing requirements
4.  Add the dataset name to the `datasets` array in `run_all.sh` for batch processing

### Adding Filtering or Pre-filtering Methods for Specialization

To extend the specialization methods with new filtering approaches:

1.  Define the new filtering method in `src/PreparingMethod.py`
2.  Implement the filtering logic to process records or columns appropriately
3.  Update the method selection logic to include your new filter
4.  Test the method with a single experiment before including it in batch runs

## Analyzing Results

The results of each run are stored in run-specific subfolders within the `results/` directory. After experiments complete, use the `auswertung.ipynb` notebook to analyze the output. The notebook reads result files (e.g., CSVs with F1-scores and other metrics), aggregates them, and generates plots and tables for comparison.

Results are also collected into the `results_collected/` folder for easier analysis across multiple experiment runs.

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
- for llm extra runs are needed for imputation and prediciton results


llm:
.env_llm
data need to be there already
predition results