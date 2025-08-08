import os
import urllib.request
from src.Vorverarbeitung import clean_and_split_data, prepare_specialization, prepare_forced_generalization, prepare_extended_specialization
from src.Generalisierung import generalize_train_and_test_data_with_ratios
import pandas as pd


def download_dataset_if_missing(dataset_name, data_dir='data'):
    """
    Download the dataset CSV if it does not exist in the expected folder.
    """

    dataset_urls = {
        'adult': 'https://github.com/luckyos-code/user-driven-privacy/raw/original/adult/dataset/adult.csv',
        'diabetes': 'https://github.com/luckyos-code/user-driven-privacy/raw/original/diabetes/dataset/diabetes.csv',
        'german_credit': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
    }
    dataset_filenames = {
        'adult': 'adult.csv',
        'diabetes': 'diabetes.csv',
        'german_credit': 'german_credit.csv',
    }
    if dataset_name not in dataset_urls:
        print(f"No download URL for dataset: {dataset_name}")
        return
    dataset_folder = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    dataset_path = os.path.join(dataset_folder, dataset_filenames[dataset_name])
    if not os.path.exists(dataset_path):
        print(f"Downloading {dataset_name} dataset...")
        url = dataset_urls[dataset_name]
        try:
            urllib.request.urlretrieve(url, dataset_path)
            print(f"Downloaded {dataset_name} to {dataset_path}")
            # for adult dataset, drop the education-num column
            if dataset_name == 'adult':
                df = pd.read_csv(dataset_path)
                if 'education-num' in df.columns:
                    df = df.drop(columns=['education-num'])
                    df.to_csv(dataset_path, index=False)
                    print("Removed 'education-num' column from adult dataset")
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")
    else:
        print(f"{dataset_name} dataset already exists at {dataset_path}")


def create_dataset_versions(dataset, original_pct, generalized_pct, missing_pct, seed=42, data_dir='data'):
    # Download dataset if missing
    download_dataset_if_missing(dataset, data_dir)

    def pct_folder(o, g, m):
        return f"{int(round(o*100))}-{int(round(g*100))}-{int(round(m*100))}"

    pct_str = pct_folder(original_pct, generalized_pct, missing_pct)

    dataset_dir = os.path.join(data_dir, dataset)
    train_file = os.path.join(dataset_dir, f'{dataset}_train.csv')
    test_file = os.path.join(dataset_dir, f'{dataset}_test.csv')

    # 1. Clean and split data (no percentage subfolder)
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print(f'Splitting and cleaning raw data for {dataset}...')
        clean_and_split_data(dataset, data_dir)
    else:
        print(f'Split data for {dataset} already exists, skipping.')

    # 2. Anonymize data (all use percentage subfolder)
    gen_train = os.path.join(dataset_dir, 'generalization', pct_str, f'{dataset}_train.csv')
    gen_test = os.path.join(dataset_dir, 'generalization', pct_str, f'{dataset}_test.csv')
    if not (os.path.exists(gen_train) and os.path.exists(gen_test)):
        print(f'Generalizing data for {dataset}...')
        generalize_train_and_test_data_with_ratios(f'{dataset}_train', os.path.join(dataset_dir, 'generalization', pct_str), original_pct, generalized_pct, missing_pct, seed, data_dir)
        generalize_train_and_test_data_with_ratios(f'{dataset}_test', os.path.join(dataset_dir, 'generalization', pct_str), original_pct, generalized_pct, missing_pct, seed, data_dir)
    else:
        print(f'Generalized data for {dataset} already exists, skipping.')

    # 3. Forced generalization (use percentage subfolder)
    forced_gen_file = os.path.join(dataset_dir, 'forced_generalization', pct_str, 'vorverarbeitet.csv')
    if not os.path.exists(forced_gen_file):
        print(f'Forced generalization for {dataset}...')
        prepare_forced_generalization(dataset, data_dir, pct_str)
    else:
        print(f'Forced generalization for {dataset} already exists, skipping.')


    # 4. Specialization and extended/weighted variants (use percentage subfolder)
    def check_and_run_preprocessing(folder, prep_func, pct):
        proxy_file = os.path.join(dataset_dir, folder, pct, 'age_vorverarbeitet.csv')
        if not os.path.exists(proxy_file):
            print(f'Preprocessing: {folder} for {dataset}...')
            prep_func(dataset, data_dir, pct)
        else:
            print(f'Preprocessed data for {folder} already exists, skipping.')

    check_and_run_preprocessing('specialization', prepare_specialization, pct_str)
    check_and_run_preprocessing('extended_weighted_specialization', prepare_extended_specialization, pct_str)

    print(f'Data preparation workflow for {dataset} finished.')
