from src.DatasetManager import DatasetManager
import numpy as np
import pandas as pd
import os

def generalize_train_and_test_data_with_ratios(dataset_name, output_subfolder, original_pct, generalized_pct, missing_pct, seed=42, data_dir='data'):
    np.random.seed(seed)
    assert abs(original_pct + generalized_pct + missing_pct - 1.0) < 1e-6, "Percentages must sum to 1.0"
    base_name = dataset_name.split('_')[0]
    input_path = os.path.join(data_dir, base_name, f'{dataset_name}.csv')
    data = pd.read_csv(input_path)
    # Remove education-num if present
    data.drop(columns=['education-num'], inplace=True, errors='ignore')
    spalten_dict, spalten_list = DatasetManager.get_spalten_classes(base_name)
    PrivacyLevel = __import__("src.spalten.PrivacyLevel", fromlist=['PrivacyLevel']).PrivacyLevel
    for column in spalten_list:
        col_values = data[column.name].astype(object)
        n = len(col_values)
        indices = np.arange(n)
        np.random.shuffle(indices)
        n_original = int(original_pct * n)
        n_generalized = int(generalized_pct * n)
        n - n_original - n_generalized
        indices[:n_original]
        generalized_idx = indices[n_original:n_original+n_generalized]
        missing_idx = indices[n_original+n_generalized:]
        for i in generalized_idx:
            privacy_levels = [pl for pl in column.dict_all.keys() if pl != PrivacyLevel.LEVEL0]
            if privacy_levels:
                chosen_level = np.random.choice(privacy_levels)
                col_values.iloc[i] = column.get_key(col_values.iloc[i], chosen_level)
        for i in missing_idx:
            col_values.iloc[i] = '?'
        data[column.name] = col_values
    out_dir = os.path.join(data_dir, base_name, output_subfolder)
    os.makedirs(out_dir, exist_ok=True)
    write_to_csv(data, os.path.join(out_dir, f'{dataset_name}.csv'))

def write_to_csv(data, path:str):
    outdir = os.path.dirname(path)
    os.makedirs(outdir, exist_ok=True)
    data.to_csv(path, index=False)