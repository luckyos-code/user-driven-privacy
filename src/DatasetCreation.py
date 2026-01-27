import os
import urllib.request
import time
import fcntl
import sys
from src.Vorverarbeitung import clean_and_split_data, prepare_specialization, prepare_forced_generalization, prepare_extended_specialization
from src.Generalisierung import generalize_train_and_test_data_with_ratios
import pandas as pd


def download_employment_dataset(data_dir='data', state='CA', survey_year='2018'):
    """
    Download and prepare the ACS Employment dataset using folktables.
    
    Args:
        data_dir: Base data directory
        state: US state code (default: 'CA' for California)
        survey_year: ACS survey year (default: '2018')
    """
    try:
        from folktables import ACSDataSource, ACSEmployment
    except ImportError:
        print("folktables package not installed. Install with: pip install folktables")
        return
    
    dataset_folder = os.path.join(data_dir, 'employment')
    os.makedirs(dataset_folder, exist_ok=True)
    dataset_path = os.path.join(dataset_folder, 'employment.csv')
    
    if os.path.exists(dataset_path):
        print(f"employment dataset already exists at {dataset_path}")
        return
    
    print(f"Downloading ACS Employment data for {state} ({survey_year})...")
    try:
        # Download ACS data
        data_source = ACSDataSource(survey_year=survey_year, horizon='1-Year', survey='person')
        acs_data = data_source.get_data(states=[state], download=True)
        print(f"Downloaded {len(acs_data)} records from ACS")
        
        # Extract features, labels, and group
        features, labels, groups = ACSEmployment.df_to_numpy(acs_data)
        
        # Create DataFrame with proper column names
        feature_names = ACSEmployment.features
        df = pd.DataFrame(features, columns=feature_names)
        df['label'] = labels.astype(int)  # Convert boolean to 0/1
        df['record_id'] = range(len(df))  # Add record IDs
        
        # Save to CSV
        df.to_csv(dataset_path, index=False)
        print(f"Saved employment dataset to {dataset_path}")
        print(f"  Features: {len(feature_names)}")
        print(f"  Records: {len(df)}")
        print(f"  Employed: {labels.sum()} ({labels.mean()*100:.1f}%)")
        
        # Clean up the downloaded year folder
        import shutil
        year_folder = os.path.join(data_dir, survey_year)
        if os.path.exists(year_folder):
            shutil.rmtree(year_folder)
            print(f"Cleaned up temporary folder: {year_folder}")
        
    except Exception as e:
        print(f"Failed to download employment dataset: {e}")
        import traceback
        traceback.print_exc()


def download_dataset_if_missing(dataset_name, data_dir='data'):
    """
    Download the dataset CSV if it does not exist in the expected folder.
    Simple check without locking (downloads are rare and fast).
    """
    # Handle employment dataset separately
    if dataset_name == 'employment':
        download_employment_dataset(data_dir=data_dir)
        return
    
    dataset_urls = {
        'adult': 'https://github.com/luckyos-code/user-driven-privacy/raw/original/adult/dataset/adult.csv',
        'diabetes': 'https://github.com/luckyos-code/user-driven-privacy/raw/original/diabetes/dataset/diabetes.csv',
        'german': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
    }
    dataset_filenames = {
        'adult': 'adult.csv',
        'diabetes': 'diabetes.csv',
        'german': 'german.csv',
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
    """
    Create all required dataset versions for a given dataset and percentage combination.
    Uses two locks:
    1. Dataset-level lock for download + clean/split (shared across all percentages)
    2. Percentage-level lock for generalization and preprocessing (per percentage)
    """
    def pct_folder(o, g, m):
        return f"{int(round(o*100))}-{int(round(g*100))}-{int(round(m*100))}"

    pct_str = pct_folder(original_pct, generalized_pct, missing_pct)
    dataset_dir = os.path.join(data_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # === PART 1: Download + Clean/Split (shared across all percentages) ===
    base_lock_path = os.path.join(dataset_dir, '.base_preparation.lock')
    base_complete_marker = os.path.join(dataset_dir, '.base_preparation.complete')
    
    if not os.path.exists(base_complete_marker):
        base_lock_file = None
        base_lock_acquired = False
        try:
            base_lock_file = open(base_lock_path, 'w')
            try:
                fcntl.flock(base_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                base_lock_acquired = True
            except IOError:
                base_lock_acquired = False
            
            if base_lock_acquired:
                # Double-check after acquiring lock
                if os.path.exists(base_complete_marker):
                    print(f'Base data for {dataset} already prepared (found after lock).')
                else:
                    print(f'=== Preparing base data for {dataset} ===')
                    sys.stdout.flush()
                    
                    # Download dataset if missing (now safely inside lock)
                    dataset_filenames = {
                        'adult': 'adult.csv',
                        'diabetes': 'diabetes.csv',
                        'german': 'german.csv',
                    }
                    dataset_path = os.path.join(dataset_dir, dataset_filenames.get(dataset, f'{dataset}.csv'))
                    
                    if not os.path.exists(dataset_path):
                        download_dataset_if_missing(dataset, data_dir)
                    else:
                        print(f"{dataset} dataset already exists at {dataset_path}")
                    
                    # Clean and split data
                    train_file = os.path.join(dataset_dir, f'{dataset}_train.csv')
                    test_file = os.path.join(dataset_dir, f'{dataset}_test.csv')
                    if not (os.path.exists(train_file) and os.path.exists(test_file)):
                        print(f'Splitting and cleaning raw data for {dataset}...')
                        sys.stdout.flush()
                        clean_and_split_data(dataset, data_dir)
                    else:
                        print(f'Split data for {dataset} already exists.')
                    
                    # Mark as complete
                    with open(base_complete_marker, 'w') as f:
                        f.write(f'Completed at {time.time()}\n')
                    print(f'Base preparation for {dataset} completed.')
                    sys.stdout.flush()
            else:
                # Wait for another process to complete base preparation
                print(f'Another process is preparing base data for {dataset}, waiting...')
                sys.stdout.flush()
                max_wait = 600  # 10 minutes for download + split
                wait_interval = 5
                waited = 0
                
                while waited < max_wait:
                    time.sleep(wait_interval)
                    waited += wait_interval
                    
                    if os.path.exists(base_complete_marker):
                        print(f'Base data for {dataset} ready.')
                        sys.stdout.flush()
                        break
                    
                    if waited % 30 == 0:
                        print(f'Still waiting for base data... ({waited}s elapsed)')
                        sys.stdout.flush()
                else:
                    raise TimeoutError(f'Timeout waiting for base data preparation after {max_wait}s')
        
        finally:
            if base_lock_file:
                try:
                    if base_lock_acquired:
                        fcntl.flock(base_lock_file.fileno(), fcntl.LOCK_UN)
                    base_lock_file.close()
                    if base_lock_acquired and os.path.exists(base_lock_path):
                        os.remove(base_lock_path)
                except Exception as e:
                    print(f'Warning: Error cleaning up base lock: {e}', file=sys.stderr)
    else:
        print(f'Base data for {dataset} already prepared.')
    
    # === PART 2: Percentage-specific processing ===
    pct_lock_path = os.path.join(dataset_dir, f'.pct_preparation_{pct_str}.lock')
    pct_complete_marker = os.path.join(dataset_dir, f'.pct_preparation_{pct_str}.complete')
    
    # Check if everything is already done
    if os.path.exists(pct_complete_marker):
        print(f'All data for {dataset} ({pct_str}) already exists, skipping.')
        return
    
    # Try to acquire percentage-specific lock
    pct_lock_acquired = False
    pct_lock_file = None
    try:
        pct_lock_file = open(pct_lock_path, 'w')
        try:
            fcntl.flock(pct_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            pct_lock_acquired = True
        except IOError:
            pct_lock_acquired = False
        
        if pct_lock_acquired:
            # Double-check after acquiring lock
            if os.path.exists(pct_complete_marker):
                print(f'All data for {dataset} ({pct_str}) already exists (found after lock), skipping.')
                return
            
            print(f'=== Creating percentage-specific data for {dataset} ({pct_str}) ===')
            sys.stdout.flush()
            
            # 2. Generalize data
            gen_dir = os.path.join(dataset_dir, 'generalization', pct_str)
            gen_train = os.path.join(gen_dir, f'{dataset}_train.csv')
            gen_test = os.path.join(gen_dir, f'{dataset}_test.csv')
            if not (os.path.exists(gen_train) and os.path.exists(gen_test)):
                print(f'Generalizing data for {dataset}...')
                sys.stdout.flush()
                generalize_train_and_test_data_with_ratios(
                    f'{dataset}_train', gen_dir, original_pct, generalized_pct, missing_pct, seed, data_dir
                )
                generalize_train_and_test_data_with_ratios(
                    f'{dataset}_test', gen_dir, original_pct, generalized_pct, missing_pct, seed, data_dir
                )
            else:
                print(f'Generalized data for {dataset} already exists, skipping.')
            
            # 3. Forced generalization
            forced_gen_file = os.path.join(dataset_dir, 'forced_generalization', pct_str, 'vorverarbeitet.csv')
            if not os.path.exists(forced_gen_file):
                print(f'Forced generalization for {dataset}...')
                sys.stdout.flush()
                prepare_forced_generalization(dataset, data_dir, pct_str)
            else:
                print(f'Forced generalization for {dataset} already exists, skipping.')
            
            # 4. Specialization
            spec_check_file = os.path.join(dataset_dir, 'specialization', pct_str, 'age_vorverarbeitet.csv')
            if not os.path.exists(spec_check_file):
                print(f'Preprocessing: specialization for {dataset}...')
                sys.stdout.flush()
                prepare_specialization(dataset, data_dir, pct_str)
            else:
                print(f'Preprocessed data for specialization already exists, skipping.')
            
            # Create completion marker
            with open(pct_complete_marker, 'w') as f:
                f.write(f'Completed at {time.time()}\n')
                f.write(f'Dataset: {dataset}\n')
                f.write(f'Percentages: {pct_str}\n')
            
            print(f'=== Data preparation for {dataset} ({pct_str}) completed successfully ===')
            sys.stdout.flush()
            
        else:
            # Another process has the lock, wait for it
            print(f'Another process is creating data for {dataset} ({pct_str}), waiting...')
            sys.stdout.flush()
            max_wait = 3600  # 60 minutes max wait
            wait_interval = 10
            waited = 0
            
            while waited < max_wait:
                time.sleep(wait_interval)
                waited += wait_interval
                
                if os.path.exists(pct_complete_marker):
                    print(f'Data creation for {dataset} ({pct_str}) completed by another process.')
                    sys.stdout.flush()
                    return
                
                if waited % 60 == 0:  # Print every minute
                    print(f'Still waiting for {dataset} ({pct_str}) data creation... ({waited}s elapsed)')
                    sys.stdout.flush()
            
            raise TimeoutError(f'Timeout waiting for {dataset} ({pct_str}) data creation after {max_wait}s')
            
    finally:
        if pct_lock_file:
            try:
                if pct_lock_acquired:
                    fcntl.flock(pct_lock_file.fileno(), fcntl.LOCK_UN)
                pct_lock_file.close()
                # Clean up lock file only if we acquired it
                if pct_lock_acquired and os.path.exists(pct_lock_path):
                    os.remove(pct_lock_path)
            except Exception as e:
                print(f'Warning: Error cleaning up pct lock file: {e}', file=sys.stderr)
                sys.stderr.flush()
    
    print(f'Data preparation workflow for {dataset} finished.')
