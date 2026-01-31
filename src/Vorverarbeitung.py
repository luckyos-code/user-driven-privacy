import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.DatasetManager import DatasetManager

def prepare_specialization(dataset, data_dir='data', percentages: str = None, limit_to_observed_values: bool = True):
    """
    Prepare specialization data by expanding generalized values.
    
    Args:
        dataset: Name of the dataset
        data_dir: Base data directory
        percentages: Percentage split string (e.g., "33-33-34")
        limit_to_observed_values: If True, only use values observed in original data (REALISTIC)
                                  If False, use all possible hierarchy values (UNREALISTIC but exhaustive)
    """
    df = get_anonymized_data(dataset, data_dir, percentages)
    folder_name = "specialization"
    specialize_data_and_save_to_csv(df, folder_name, dataset, data_dir, percentages, 
                                    extended=False, limit_to_observed_values=limit_to_observed_values)

def prepare_forced_generalization(dataset_name, data_dir='data', percentages: str = None):
    df = get_anonymized_data(dataset_name, data_dir, percentages)
    folder_name = "forced_generalization"
    preprocess_data_to_highest_privacy_level(df, folder_name, dataset_name, data_dir, percentages)

def prepare_extended_specialization(dataset, data_dir='data', percentages: str = None, limit_to_observed_values: bool = True):
    """
    Prepare extended specialization data (special handling for age).
    
    Args:
        dataset: Name of the dataset
        data_dir: Base data directory  
        percentages: Percentage split string (e.g., "33-33-34")
        limit_to_observed_values: If True, only use values observed in original data (REALISTIC)
                                  If False, use all possible hierarchy values (UNREALISTIC but exhaustive)
    """
    df = get_anonymized_data(dataset, data_dir, percentages)
    folder_name = "extended_weighted_specialization"
    specialize_data_and_save_to_csv(df, folder_name, dataset, data_dir, percentages, 
                                    extended=True, limit_to_observed_values=limit_to_observed_values)


def get_anonymized_data(dataset_name, data_dir='data', percentages: str = None):
    if percentages:
        train_path = os.path.join(data_dir, dataset_name, 'generalization', percentages, f'{dataset_name}_train.csv')
        test_path = os.path.join(data_dir, dataset_name, 'generalization', percentages, f'{dataset_name}_test.csv')
    else:
        train_path = os.path.join(data_dir, dataset_name, 'generalization', f'{dataset_name}_train.csv')
        test_path = os.path.join(data_dir, dataset_name, 'generalization', f'{dataset_name}_test.csv')
    manipulated_data_train = pd.read_csv(train_path)
    manipulated_data_test = pd.read_csv(test_path)
    df = pd.concat([manipulated_data_train, manipulated_data_test])
    df = df.reset_index(drop=True)
    return df

# helper function specialization
def ersetze_durch_mittelwert(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    for column in columns:
        for idx, row in df.iterrows():
            value = row[column.name]
            # Only process generalized values (not original numeric values)
            if column.is_generalized(value):
                intervall = column.get_value(value)
                if intervall is not None:
                    new_value = intervall[len(intervall) // 2]
                    df.at[idx, column.name] = new_value
    return df

# helper function specialization
def create_rows_for_values(row, attributeName, values):
    new_rows = []
    for value in values:
        new_row = row.copy()
        new_row[attributeName] = value
        new_rows.append(new_row)
    return new_rows

# helper function specialization
def erstelle_neue_zeilen(df: pd.DataFrame, columns: list, limit_to_observed_values: bool = False, observed_values_dict: dict = None) -> pd.DataFrame:
    """
    Create new rows by specializing generalized values.
    
    Args:
        df: DataFrame with generalized values
        columns: List of column classes to process
        limit_to_observed_values: If True, only use values that were observed in original data
        observed_values_dict: Dict mapping column_name -> set of observed original values
                             Required if limit_to_observed_values=True
    
    Returns:
        DataFrame with specialized rows
    """
    df = df.reset_index(drop=True)
    for column in columns:
        condition = df[column.name].apply(column.is_generalized)
        rows_to_clone = df[condition].copy().reset_index(drop=True)
        base_df = df[~condition].reset_index(drop=True)
        new_rows = []
        for _, row in rows_to_clone.iterrows():
            generalized_value = row[column.name]
            
            # Get possible values for this generalized value
            if limit_to_observed_values and observed_values_dict and column.name in observed_values_dict:
                # REALISTIC: Only use values that were actually observed in original data
                all_possible_values = column.get_value(generalized_value)
                observed_values = observed_values_dict[column.name]
                # Type-safe filtering: convert both to strings for comparison
                # This handles cases where hierarchy has ints [0, 1] but observed has strings ['0', '1']
                observed_str = {str(v) for v in observed_values}
                intervall = [v for v in all_possible_values if str(v) in observed_str]
                
                # Fallback: If no observed values match (shouldn't happen with good data),
                # keep original generalized value to avoid losing the row
                if not intervall:
                    intervall = [generalized_value]
            else:
                # UNREALISTIC (current behavior): Use all possible values from hierarchy
                intervall = column.get_value(generalized_value)
            
            new_rows.extend(create_rows_for_values(row, column.name, intervall))
    df = pd.concat([base_df, pd.DataFrame(new_rows)], ignore_index=True)
    # old adult education-num strategy
    # if column.value.name == Spalten.EDUCATION.value.name:
    #     # Füge Spalte education-num hinzu. Die Werte von education-num sind die Integer Values von education_num_mapping für die Werte von education.
    #     df["education-num"] = df["education"].apply(lambda x: EducationNum.education_num_mapping[x])
    return df

def specialize_data_and_save_to_csv(df: pd.DataFrame, folder_name: str, dataset_name, data_dir='data', percentages: str = None, extended: bool = False, limit_to_observed_values: bool = False):
    """
    Specialize generalized data by creating variants for each generalized value.
    
    Args:
        df: DataFrame with generalized data
        folder_name: Output folder name
        dataset_name: Name of the dataset
        data_dir: Base data directory
        percentages: Percentage split string (e.g., "33-33-34")
        extended: Whether to use extended specialization (age handling)
        limit_to_observed_values: If True, only create variants with values observed in original data
                                  If False, use all possible values from hierarchies (unrealistic but exhaustive)
    """
    import sys
    spalten_dict, spalten_list = DatasetManager.get_spalten_classes(dataset_name)
    numerical_columns = DatasetManager.get_numerical_columns(dataset_name)
    record_id_col = DatasetManager.get_record_id_column(dataset_name)
    if percentages:
        out_dir = os.path.join(data_dir, dataset_name, folder_name, percentages)
    else:
        out_dir = os.path.join(data_dir, dataset_name, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Extract observed values from original (non-generalized, non-missing) data if needed
    # ONLY for categorical columns (numerical columns use mean imputation, not specialization)
    observed_values_dict = None
    if limit_to_observed_values:
        observed_values_dict = extract_observed_values(dataset_name, df, data_dir)
    
    for column in spalten_list:
        try:
            actual_df = df[[record_id_col, column.name]]
            # Only exclude "AGE" if extended is True
            if numerical_columns and column.name in numerical_columns and not (extended and column.name.lower() == "age"):
                processed_df = ersetze_durch_mittelwert(actual_df, [column])
            else:
                processed_df = erstelle_neue_zeilen(actual_df, [column], limit_to_observed_values, observed_values_dict)
            
            output_path = os.path.join(out_dir, f"{column.name}_vorverarbeitet.csv")
            write_to_csv(processed_df, output_path)
            
            # Flush stdout before printing to avoid buffering issues
            sys.stdout.flush()
            print(f"Dataframe {column.name} wurde vorverarbeitet und gespeichert")
            sys.stdout.flush()
        except OSError as e:
            # Handle stale file handle errors gracefully
            print(f"Warning: OSError beim Verarbeiten von {column.name}: {e}", file=sys.stderr)
            sys.stderr.flush()
            # Try to continue with other columns
            continue
        except Exception as e:
            print(f"Error beim Verarbeiten von {column.name}: {e}", file=sys.stderr)
            sys.stderr.flush()
            raise

def preprocess_data_to_highest_privacy_level(df: pd.DataFrame, folder_name: str, dataset_name, data_dir='data', percentages: str = None):
    spalten_dict, spalten_list = DatasetManager.get_spalten_classes(dataset_name)
    DatasetManager.get_record_id_column(dataset_name)
    if percentages:
        out_dir = os.path.join(data_dir, dataset_name, folder_name, percentages)
    else:
        out_dir = os.path.join(data_dir, dataset_name, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    for column in spalten_list:
        for index, row in df.iterrows():
            df.at[index, column.name] = column.get_highest_privacy_value(row[column.name])
        print(f"Spalte {column.name} wurde vorverarbeitet")
    write_to_csv(df, os.path.join(out_dir, "vorverarbeitet.csv"))


def get_anonymized_data_analysis(anonymization_level: str, dataset_name, data_dir='data'):
    spalten_dict, spalten_list = DatasetManager.get_spalten_classes(dataset_name)
    Anonymization = DatasetManager.get_anonymization_class(dataset_name)
    label_col = DatasetManager.get_label_column(dataset_name)
    record_id_col = DatasetManager.get_record_id_column(dataset_name)
    df_train = pd.read_csv(os.path.join(data_dir, dataset_name, 'generalization', f'{dataset_name}_train.csv'))
    df_train.drop(columns=[record_id_col, label_col], inplace=True, errors='ignore')
    df_test = pd.read_csv(os.path.join(data_dir, dataset_name, 'generalization', f'{dataset_name}_test.csv'))
    df_test.drop(columns=[record_id_col, label_col], inplace=True, errors='ignore')
    
    # Count actual values (not ratios) across all anonymized columns
    total_values_train = 0
    total_original_train = 0
    total_generalized_train = 0
    total_missing_train = 0
    
    total_values_test = 0
    total_original_test = 0
    total_generalized_test = 0
    total_missing_test = 0
    
    for column in getattr(Anonymization, anonymization_level).value:
        original_train, generalized_train, missing_train, count_train = get_data_analysis_by_column(spalten_dict[column.name], df_train)
        total_values_train += count_train
        total_original_train += original_train
        total_generalized_train += generalized_train
        total_missing_train += missing_train
        
        original_test, generalized_test, missing_test, count_test = get_data_analysis_by_column(spalten_dict[column.name], df_test)
        total_values_test += count_test
        total_original_test += original_test
        total_generalized_test += generalized_test
        total_missing_test += missing_test
    
    # Calculate ratios from total counts
    generalized_ratio_train = total_generalized_train / total_values_train if total_values_train > 0 else 0
    missing_ratio_train = total_missing_train / total_values_train if total_values_train > 0 else 0
    original_ratio_train = total_original_train / total_values_train if total_values_train > 0 else 0
    
    generalized_ratio_test = total_generalized_test / total_values_test if total_values_test > 0 else 0
    missing_ratio_test = total_missing_test / total_values_test if total_values_test > 0 else 0
    original_ratio_test = total_original_test / total_values_test if total_values_test > 0 else 0
    
    # Calculate overall ratios
    total_values_all = total_values_train + total_values_test
    generalized_ratio = (total_generalized_train + total_generalized_test) / total_values_all if total_values_all > 0 else 0
    missing_ratio = (total_missing_train + total_missing_test) / total_values_all if total_values_all > 0 else 0
    original_ratio = (total_original_train + total_original_test) / total_values_all if total_values_all > 0 else 0
    
    print(f"Train: Original: {original_ratio_train}, Generalized: {generalized_ratio_train}, Missing: {missing_ratio_train}")
    print(f"Test: Original: {original_ratio_test}, Generalized: {generalized_ratio_test}, Missing: {missing_ratio_test}")
    print(f"Total: Original: {original_ratio}, Generalized: {generalized_ratio}, Missing: {missing_ratio}")


def get_data_analysis_by_column(column, df):
    print(f"Spalte {column.name} wird analysiert")
    print(f"Anzahl Einträge: {df[column.name].count()}")
    total_values = df[column.name].count()
    original = 0
    generalized = 0
    missing = 0
    for index, row in df.iterrows():
        value = row[column.name]
        if value == "?":
            missing += 1
        elif column.is_generalized(value):
            generalized += 1
        else:
            original += 1
    print(f"Original: {original/total_values}, Generalized: {generalized/total_values}, Missing: {missing/total_values}")
    # Return actual counts, not ratios
    return original, generalized, missing, total_values



def clean_and_split_data(dataset_name, data_dir='data'):
    """
    This function cleans the dataset and splits it into a training and a test set.
    Handles special logic for german (assigns column names, maps codes if needed).
    Handles employment dataset which already has record_id and clean data.
    """
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    if dataset_name == 'employment':
        # Employment data is already clean from folktables, just needs splitting
        data = pd.read_csv(os.path.join(dataset_dir, f'{dataset_name}.csv'))
        # Data already has record_id column and clean binary labels
        data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)
        data.to_csv(os.path.join(dataset_dir, f'{dataset_name}_cleaned.csv'), index=False)
        data_train.to_csv(os.path.join(dataset_dir, f'{dataset_name}_train.csv'), index=False)
        data_test.to_csv(os.path.join(dataset_dir, f'{dataset_name}_test.csv'), index=False)
        return
    
    if dataset_name == 'german':
        # Use correct column names for german
        columns = DatasetManager.get_config('german')['all_columns']
        data = pd.read_csv(os.path.join(dataset_dir, f'{dataset_name}.csv'), sep=' ', header=None, names=columns)
        # Map categorical codes to readable values
        mappings = DatasetManager.get_config('german').get('mappings', {})
        for col, mapping in mappings.items():
            if col in data.columns:
                data[col] = data[col].map(mapping).fillna(data[col])
        # Convert label: 1 (good) -> 0, 2 (bad) -> 1 for binary classification
        label_col = DatasetManager.get_label_column(dataset_name)
        data[label_col] = data[label_col].apply(lambda x: 0 if x == 1 else 1)
    else:
        data = pd.read_csv(os.path.join(dataset_dir, f'{dataset_name}.csv'), na_values=['?'])
        data.dropna(inplace=True)

        if dataset_name == 'adult':
            new_output = []
            for value in data[DatasetManager.get_label_column(dataset_name)]:
                if value == '<=50K' or value == '<=50K.':
                    new_output.append(0)
                elif value == '>50K' or value == '>50K.':
                    new_output.append(1)
            data[DatasetManager.get_label_column(dataset_name)] = new_output
        elif dataset_name == 'diabetes':
            for col in data.columns:
                # Nur auf float-Spalten anwenden
                if pd.api.types.is_float_dtype(data[col]):
                    # Bedingung: Wert ist float und gleich dem gerundeten Wert
                    data[col] = data[col].apply(lambda x: int(x) if x.is_integer() else x)

    data.to_csv(os.path.join(dataset_dir, f'{dataset_name}_cleaned.csv'), index_label='record_id')
    data = pd.read_csv(os.path.join(dataset_dir, f'{dataset_name}_cleaned.csv'))
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)
    data_train.to_csv(os.path.join(dataset_dir, f'{dataset_name}_train.csv'), index=False)
    data_test.to_csv(os.path.join(dataset_dir, f'{dataset_name}_test.csv'), index=False)

def write_to_csv(data, path):
    #split path in directory and filename
    outdir = os.path.dirname(path)
    os.makedirs(outdir, exist_ok=True)
    data.to_csv(path, index=False)


def extract_observed_values(dataset_name, df, data_dir='data'):
    """
    Extract observed values from original (non-generalized, non-missing) data.
    Only extracts for categorical columns (numerical use mean imputation).
    
    Args:
        dataset_name: Name of the dataset
        df: DataFrame with generalized data
        data_dir: Base data directory
        
    Returns:
        Dict mapping column_name -> set of observed original values
    """
    spalten_dict, spalten_list = DatasetManager.get_spalten_classes(dataset_name)
    numerical_columns = DatasetManager.get_numerical_columns(dataset_name)
    
    print("Extracting observed values from original data...")
    print("(Only categorical columns - numerical use mean imputation)")
    
    observed_values_dict = {}
    for column in spalten_list:
        # Skip numerical columns - they use mean imputation, not specialization
        if numerical_columns and column.name in numerical_columns:
            continue
        
        if column.name in df.columns:
            # Get values that are NOT generalized and NOT missing
            original_values = df[column.name][
                ~df[column.name].apply(column.is_generalized) & 
                df[column.name].notna() &
                (df[column.name] != '?') &
                (df[column.name] != '')
            ].unique()
            observed_values_dict[column.name] = set(original_values)
            print(f"  {column.name}: {len(observed_values_dict[column.name])} observed values → {sorted(list(observed_values_dict[column.name]))}")
    
    return observed_values_dict
