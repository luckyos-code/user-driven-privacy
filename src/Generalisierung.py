from src.DatasetManager import DatasetManager
import numpy as np
import pandas as pd
import os

def generalize_train_and_test_data_with_ratios(dataset_name, output_subfolder, original_pct, generalized_pct, missing_pct, seed=42, data_dir='data'):
    np.random.seed(seed)
    assert abs(original_pct + generalized_pct + missing_pct - 1.0) < 1e-6, "Percentages must sum to 1.0"
    
    # Extract base dataset name (handles 'adult_train', 'german_train', etc.)
    # For 'german_train', we want 'german', not just the first part before underscore
    if '_train' in dataset_name:
        base_name = dataset_name.replace('_train', '')
    elif '_test' in dataset_name:
        base_name = dataset_name.replace('_test', '')
    else:
        base_name = dataset_name
    
    input_path = os.path.join(data_dir, base_name, f'{dataset_name}.csv')
    data = pd.read_csv(input_path)
    # Remove education-num if present
    data.drop(columns=['education-num'], inplace=True, errors='ignore')
    spalten_dict, spalten_list = DatasetManager.get_spalten_classes(base_name)
    PrivacyLevel = __import__("src.spalten.PrivacyLevel", fromlist=['PrivacyLevel']).PrivacyLevel
    
    print(f"Generating data with ratios: Original={original_pct:.2%}, Generalized={generalized_pct:.2%}, Missing={missing_pct:.2%}")
    
    for column in spalten_list:
        col_values = data[column.name].astype(object)
        n = len(col_values)
        indices = np.arange(n)
        np.random.shuffle(indices)
        n_original = int(original_pct * n)
        n_generalized = int(generalized_pct * n)
        n_missing = n - n_original - n_generalized
        
        print(f"\nColumn '{column.name}': n={n}, original={n_original}, generalized={n_generalized}, missing={n_missing}")
        
        original_idx = indices[:n_original]
        generalized_idx = indices[n_original:n_original+n_generalized]
        missing_idx = indices[n_original+n_generalized:]
        
        # Apply generalization
        # Exclude LEVEL0 (original) and any level that only contains "?" (full suppression)
        # Only use privacy levels that provide actual generalization
        privacy_levels = []
        for pl in column.dict_all.keys():
            if pl == PrivacyLevel.LEVEL0:
                continue
            # Check if this level only contains "?" - if so, skip it (it's suppression, not generalization)
            level_keys = list(column.dict_all[pl].keys())
            if level_keys == ['?']:
                continue
            privacy_levels.append(pl)
        
        if privacy_levels:
            # Column has generalization levels - apply them
            for i in generalized_idx:
                chosen_level = np.random.choice(privacy_levels)
                try:
                    col_values.iloc[i] = column.get_key(col_values.iloc[i], chosen_level)
                except Exception as e:
                    print(f"  Error generalizing value '{col_values.iloc[i]}' at index {i}: {e}")
        else:
            # No generalization available - convert "generalized" portion to missing
            print(f"  No generalization levels available - converting generalized portion to missing")
            for i in generalized_idx:
                col_values.iloc[i] = '?'
        
        # Apply missing marker
        for i in missing_idx:
            col_values.iloc[i] = '?'
        
        data[column.name] = col_values
        
        # Verify what we got
        actual_missing = (data[column.name] == '?').sum()
        actual_generalized = data[column.name].apply(lambda x: column.is_generalized(str(x)) and str(x) != '?').sum()
        actual_original = n - actual_missing - actual_generalized
        print(f"  Result: original={actual_original}, generalized={actual_generalized}, missing={actual_missing}")
    
    # output_subfolder is already a full path from DatasetCreation.py
    out_dir = output_subfolder
    os.makedirs(out_dir, exist_ok=True)
    write_to_csv(data, os.path.join(out_dir, f'{dataset_name}.csv'))

def write_to_csv(data, path:str):
    outdir = os.path.dirname(path)
    os.makedirs(outdir, exist_ok=True)
    data.to_csv(path, index=False)