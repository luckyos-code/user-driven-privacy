# Learning from Anonymized and Incomplete Tabular Data

### Paper

This repository contains the codebase for the paper: **[Learning from Anonymized and Incomplete Tabular Data](TODO)**

```bibtex
@article{TODO,
  title={...},
  author={...},
  journal={...},
  year={...}
}
```
### Abstract TODO
```
User-driven privacy allows individuals to control whether and at what granularity their data is shared, leading to datasets that mix original, generalized, and missing values within the same records and attributes. While such representations are intuitive for privacy, they pose challenges for machine learning, which typically treats non-original values as new categories or as missing, thereby discarding generalization semantics. For learning from such tabular data, we propose novel data transformation strategies that account for heterogeneous anonymizations and evaluate them alongside standard imputation and LLM–based approaches. We employ multiple datasets, privacy configurations, and deployment scenarios, demonstrating that our method reliably regains utility. Our results show that generalized values are preferable to pure suppression, that the best preparation strategy depends on the scenario, and that consistent data representations are crucial for maintaining downstream utility. Overall, our findings highlight that effective learning is tied to the appropriate handling of anonymized values.
```

### Reference Data

The (anonymized) datasets and LLM imputation/prediction results used in our experiments are publicly available at: [ZENODO](https://doi.org/10.5281/zenodo.18405312)

Alternatively, generate the datasets anew by following the instructions below.


## Project Structure

**Main Scripts and Locations:**
-   `run_all.sh`, `run.job`, `run.py`: The main scripts to running the general experiments with different configurations.
-   `ll_runs.sh`, `llm.job`, `llm_src/`: scripts for LLM evaluation.
-   `requirements_*.txt`: Needed packages for installation.
-   `DatasetCreation.ipynb`: Notebook for individual generation of original and anonymized datasets
-   `DatasetCreation_Specialization.ipynb`: Notebook for individual generation of specialized datasets with filtering (optional, normally handled in-memory during experiments)
-   `src/`: Contains the core Python source code for all pipeline steps.
    -   `DatasetManager.py`: Collection of dataset important information used in the workflows.
    -   `spalten/`: Dataset column specifications and generalization hierarchies
    -   `DatasetCreation.py`: Creates dataset versions based on specified privacy preferences (percentages of original, generalized, and missing data).
    -   `Main.py`: Contains the main evaluation loop (`run_evaluation`) for model training and testing.
    -   Additional modules for data loading, model evaluation, etc.


**Data and Output Directory Reference:**
- `data/`: Raw and processed datasets
  - `{dataset}/`: Original dataset files
  - `{dataset}/generalization/{percentage}/`: Anonymized datasets for each privacy distribution
  - `{dataset}/forced_generalization/{percentage}/`: Forced generalization preparation output
  - `{dataset}/specialization/{percentage}/`: Optional pre-generated specialization datasets (normally created in-memory during experiments)
- `llm_evaluation/`: LLM evaluation outputs
  - `{percentage}_results/`: Intermediate evaluation results (rename to `{percentage}/` when complete)
  - `{percentage}/`: Final results used by model evaluation
- `llm_slurm_logs/`: Slurm job logs for LLM experiments
- `results/`: Experiment run outputs
- `slurm_logs/`: Slurm job logs for overall experiments

## Environment and Installation
We used and recommend using a Slurm + Dask combination on a cluster for the general and LLM experiments.
Local mode is ok for creating anonymized datasets, local LLM setups, and testing.

-   **Slurm**: For distributed job scheduling (not required for dataset generation and small LLM outputs)
-   **Dask**: For parallel processing (not required for dataset generation and LLM outputs but included in requirements file)
-   **Python 3.x**: Depending on environment, with dependencies listed in requirements files

The project provides different requirement files for different use cases:

-   **`requirements_local.txt`**: For local development (Python 3.9.19, includes LLM packages)
-   **`requirements_cluster.txt`**: As reference for cluster environment as loaded by `run.job`, adjust job files to your environment (Python/3.9.6 on cluster, without LLM packages, as they're loaded via `llm.job` for such experiments)
-   **`requirements_llm.txt`**: Only packages needed for LLM experiments (Python 3.12.3 on cluster)

```bash
# Installing packages for local development (for cluster see run.job and llm.job)

# using a conda environment
conda create -n requirements-local python=3.9.19 -y && \
conda run -n requirements-local pip install -r requirements_local.txt

# if pip from file not working:
#   for some platforms with compatibility issues, use several conda distributed packages for ensuring support (all versions copied from the file)
conda create -n requirements-local python=3.9.19 -y && \
conda install -n requirements-local numpy=1.23.5 scipy=1.11.3 pandas=1.5.3 scikit-learn=1.2.2 matplotlib=3.5.2 -c conda-forge -y && \
conda run -n requirements-local pip install 'dask[distributed]==2023.1.0' dask-jobqueue==0.8.1 dask-mpi==2022.4.0 aiohttp==3.11.11 fastparquet==2024.11.0 folktables==0.0.12 h5py==3.11.0 jupyterlab==3.6.0 python-dotenv==1.0.0 requests==2.32.3 seaborn==0.13.2 xgboost==2.0.3
```

# Instructions

This section covers how to run specific parts of the experiments and how to extend the codebase with new functionality.

## 1. Generating Anonymized Datasets
This step is only needed if creating datasets separately, otherwise it is done and ensured on the start of each run in `run_all.sh` / `run.job` / `run.py` in Step 3. You may also download them from the [Zenodo](https://doi.org/10.5281/zenodo.18405312) repository instead of running yourself.

### 1a. Original and Anonymized Datasets

Use [DatasetCreation.ipynb](DatasetCreation.ipynb), which provides an easy wrapper for running the needed script:
* Downloads, cleans, and splits the original datasets
* Creates anonymized versions depending on the set percentages
* Creates prepared versions for the forced generalization method

### 1b. Specialization Datasets (Optional)

**Note:** Specialization datasets are normally created in-memory during experiments (Step 3). This step is only needed if you want to pre-generate specialization data for analysis or reuse. Actual specialization workflow still creates it in memory and does not read from or write to disk.

Use [DatasetCreation_Specialization.ipynb](DatasetCreation_Specialization.ipynb) to create specialized datasets with record-based approach:
* Loads generalized data from `data/{dataset}/generalization/{percentages}/`
* For each record, generates variants by expanding generalized values and applies filtering to select best variants if wanted
* Saves filtered datasets with descriptive names: `specialization_{mode}_n{duplicates}.csv`

### Data Folder
Assuming the default `data/` location we get the following structure:
* `data/`: original and anonymized data for four datasets sorted into subfolders by dataset name
    * `original`: at the toplevel of each dataset folder we find a `train.csv` and `test.csv` with original data in 80/20 split and the following subfolders:
    * `generalization/`: contains the anonymized data that again is sorted into subfolders for the different privacy distributions, with train and test data for each 
    * `forced_generalization/`: similar structure to `generalization/` but contains the prepared data under forced generalization method on the anonymized datasets, contains the whole dataset (no split) which will then be joined into the respective train and test parts by record ids in the experiments
    * `specialization/`: optional folder for pre-generated specialization datasets (see Step 1b above), normally specialization is created and handled in-memory during experiments
    * Note: resulting datasets of other methods are not saved to disk as they are created and handled in-memory only

## 2. Generating LLM-Based Outputs

The project includes LLM-based approaches for data imputation and prediction. While prediction is a method on its own, outputs for imputation through an LLM are needed if using llm_imputation in Step 3 runs. You may also download from the [Zenodo](https://doi.org/10.5281/zenodo.18405312) repository instead of running yourself.

**What LLM evaluation does:**

For each dataset's training and test sets:
1. **Value Imputation**: Predicts missing/generalized values and creates imputed dataset versions (both parts saved as CSV)
2. **Label Prediction**: Directly predicts target labels from anonymized records (results saved as CSV)

Results are saved in `llm_evaluation/{percentage}_results/` directories. When finished, rename to `llm_evaluation/{percentage}/` for use in model evaluation.

### Environment Configuration

1. Configure the LLM API settings in `_.env_llm` and rename to `.env_llm` (in project root):
   ```bash
   LLM_API_BASE_URL="..."
   LLM_API_KEY="..."
   LLM_MODEL="..."
   ```

2. Install (at least) LLM-specific dependencies or check cluster dependencies in `llm.job` file:
   ```bash
   pip install -r requirements_llm.txt
   ```

3. Ensure original and anonymized datasets exist, assuming the default `data/` location (from Step 1, Step 3 (except llm_imputation method), or downloaded from [Zenodo](https://doi.org/10.5281/zenodo.18405312) repository)

### Running LLM Evaluation

**Option 1: Local Execution for Testing or Smaller Tasks**

* Use [llm_evaluation_test.ipynb](llm_src/llm_evaluation_test.ipynb) in `llm_src/`

**Option 2: Interactive Shell Wrapper (Slurm)**
Starts jobs on cluster that interact with the LLM API. See `llm_runs.sh` and `llm.job`.

```bash
# command: bash llm_runs.sh launch <percentage> <datasets> <partitions> [batch_size] [input_dir]

# Single percentage, multiple datasets
bash llm_runs.sh launch "33-33-34" "Adult-Diabetes-German" 1 10

# Multiple percentages, with partitioning for large datasets
bash llm_runs.sh launch "33-33-34,66-17-17" "Employment" 3 10

# Check job status
bash llm_runs.sh status

# Merge partitioned results after completion (if applicable)
bash llm_runs.sh merge "33-33-34" "Employment" 3
```

Other than the top-level job scripts, the subfolder `llm_src/` contains the LLM-related scripts:
- `.env_llm`: Environment variables (LLM API URL, key, model), fill and rename from `_.env_llm`
- `llm_runs.sh`: Shell wrapper for launching multiple LLM-related jobs with extra options
- `llm.job`: Slurm job definition file
- `llm_src/llm_evaluation.py`: Main LLM evaluation script
- `llm_src/llm_evaluation_test.ipynb`: Jupyter notebook for small test runs
- `llm_src/llm_evaluation_launcher.py`: Job launcher with partitioning support
- `llm_src/llm_evaluation_merger.py`: Merges results from partitioned runs
- `llm_src/merge_llm_results.py`: Utility scripts for merging partitioned results

## 3. Overall Evaluation Runs
**Highly recommend Slurm+Dask on a cluster.** Local is still possible with a LocalCluster as implemented in `Main.py`.

Run the complete evaluation pipeline to train and test models on prepared datasets. Results are automatically saved to run-specific subfolders (based on the run_all.sh session) in the `results/` directory and organized following the structure: `dataset -> distribution -> json files`.

### Running Batch Experiments

To run a comprehensive set of experiments, you can configure and use the `run_all.sh` script.

1.  **Configure `run_all.sh`**: Open the `run_all.sh` script and modify the arrays (`datasets`, `percentages_list`, `train_methods`, `test_methods`, etc.) by commenting out unwanted runs to define the parameter space for your experiments. Also consider the variables `DATA_DIR` and `batch_size` (also `require_matching_methods` if needed). Cluster-related settings for the DaskSlurmCluster constructor need to be changed in the `Main.py`.

2.  **Execute the script**:
    ```bash
    bash run_all.sh
    ```
    This will start submitting jobs through the `run.job` file to the Slurm cluster based on the configurations. Each job will execute the `run.py` script with a different combination of parameters.


### Running a Single Experiment

You may use `run.py` for starting single experiments outside of submitting a Slurm job. See `run.py` for parameters and for our used configurations see `run_all.sh` and `run.job`. Local runs use LocalCluster, see `Main.py`, which we generally did not use for the experiments.

**Example command:**

```bash
python run.py \
    --save_dir ./results/my_experiment \
    --data_dir ./data \
    --dataset german \
    --train_method original \
    --test_method original \
    --percentages "1.0 0.0 0.0" \
    --n_workers 1
```

## Extending the Codebase

### Adding New Privacy Distribution

1. Just run overall experiments with the dataset and percentage and everything will be created accordingly (or run the individual dataset creation)
2. LLM runs on the new distributions, needed for LLM-based imputation and prediction results

### Adding New Dataset

1. **Add dataset configuration**: Create a config entry in `src/DatasetManager.py` with dataset information, attributes, and anonymization levels.

2. **Create column specifications**: Add a new class in `src/spalten/` folder (following the pattern of existing datasets) that defines generalizations for each column. Import the class in `src/spalten/__init__.py`.

3.  **Configure dataset download**: Add dataset information to `download_dataset_if_missing()` in `src/DatasetCreation.py` for automatic download, or add a custom download function (like for Employment dataset), or manually place files in the `data/` folder.

4.  **Implement data cleaning**: Review `clean_and_split_data()` in `src/Vorverarbeitung.py` and add any dataset-specific cleaning or splitting logic required.

5.  **Update batch scripts**: Add the dataset name to the `datasets` array in `run_all.sh` for batch processing.
