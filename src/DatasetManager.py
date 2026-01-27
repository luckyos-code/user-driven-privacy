from enum import Enum
import importlib

class DatasetManager:
    """Unified manager for dataset-specific configurations and Spalten enums"""
    
    # Define configurations for each supported dataset
    _CONFIGS = {
        "adult": {
            "categorical_columns": ['workclass', 'education', 'marital-status', 
                                   'occupation', 'relationship', 'race', 
                                   'sex', 'native-country'],
            "numerical_columns": ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
            "label_column": 'income',
            "record_id_column": 'record_id',
            "target_description": "Target values: 0 (<=50K income) or 1 (>50K income)",
            "anonymization": {
                "no": [], # No anonymization
                "basic": ["age", "sex", "race"], # Basic selection of sensitive attributes
                "all": None  # Will be populated dynamically with all columns - all attributes are sensitive
            }
        },
        "diabetes": {
            "categorical_columns": ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
                                    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
                                    "NoDocbcCost", "DiffWalk", "Sex"],
            "numerical_columns": ["Age", "BMI", "GenHlth", "MentHlth", "PhysHlth", "Education", "Income"],
            "label_column": 'Diabetes_binary',
            "record_id_column": 'record_id',
            "target_description": "Target values: 0 (no diabetes) or 1 (has diabetes)",
            "anonymization": {
                "no": [], 
                "basic": ["Age", "Sex", "Income"],
                "all": None
            }
        },
        "german": {
            # UCI German Credit columns, in order
            "all_columns": [
                "status", "duration", "credit_history", "purpose", "credit_amount", "savings", "employment_since",
                "installment_rate", "personal_status_sex", "other_debtors", "present_residence", "property",
                "age", "other_installment_plans", "housing", "number_credits", "job", "people_liable",
                "telephone", "foreign_worker", "label"
            ],
            "categorical_columns": [
                "status", "credit_history", "purpose", "savings", "employment_since", "personal_status_sex",
                "other_debtors", "property", "other_installment_plans", "housing", "job", "telephone", "foreign_worker"
            ],
            "numerical_columns": ["duration", "credit_amount", "installment_rate", "present_residence", "age", "number_credits", "people_liable"],
            "label_column": "label",
            "record_id_column": "record_id",
            "anonymization": {
                "no": [],
                "basic": ["age", "foreign_worker", "housing"],
                "all": None
            },
            "target_description": "Target values: 1 (good credit) or 2 (bad credit)",
            # Complete mapping for all categorical codes from UCI documentation
            "mappings": {
                "status": {
                    "A11": "<0_DM",
                    "A12": "0-200_DM",
                    "A13": ">=200_DM",
                    "A14": "no_account"
                },
                "credit_history": {
                    "A30": "no_credits",
                    "A31": "all_paid",
                    "A32": "existing_paid",
                    "A33": "delay",
                    "A34": "critical"
                },
                "purpose": {
                    "A40": "car_new",
                    "A41": "car_used",
                    "A42": "furniture",
                    "A43": "radio_tv",
                    "A44": "appliances",
                    "A45": "repairs",
                    "A46": "education",
                    "A48": "retraining",
                    "A49": "business",
                    "A410": "others"
                },
                "savings": {
                    "A61": "<100_DM",
                    "A62": "100-500_DM",
                    "A63": "500-1000_DM",
                    "A64": ">=1000_DM",
                    "A65": "unknown"
                },
                "employment_since": {
                    "A71": "unemployed",
                    "A72": "<1_year",
                    "A73": "1-4_years",
                    "A74": "4-7_years",
                    "A75": ">=7_years"
                },
                "personal_status_sex": {
                    "A91": "male_divorced",
                    "A92": "female_divorced_married",
                    "A93": "male_single",
                    "A94": "male_married",
                    "A95": "female_single"
                },
                "other_debtors": {
                    "A101": "none",
                    "A102": "co_applicant",
                    "A103": "guarantor"
                },
                "property": {
                    "A121": "real_estate",
                    "A122": "building_society",
                    "A123": "car_other",
                    "A124": "unknown"
                },
                "other_installment_plans": {
                    "A141": "bank",
                    "A142": "stores",
                    "A143": "none"
                },
                "housing": {
                    "A151": "rent",
                    "A152": "own",
                    "A153": "free"
                },
                "job": {
                    "A171": "unemployed_unskilled",
                    "A172": "unskilled",
                    "A173": "skilled",
                    "A174": "management"
                },
                "telephone": {
                    "A191": "none",
                    "A192": "yes"
                },
                "foreign_worker": {
                    "A201": "yes",
                    "A202": "no"
                }
            }
        },
        "employment": {
            # ACS Employment task from folktables
            # Features from 2018 ACS PUMS (American Community Survey Public Use Microdata Sample)
            "categorical_columns": [
                "SCHL",      # Educational attainment
                "MAR",       # Marital status
                "RELP",      # Relationship to reference person
                "DIS",       # Disability recode
                "ESP",       # Employment status of parents
                "CIT",       # Citizenship status
                "MIG",       # Mobility status (lived here 1 year ago)
                "MIL",       # Military service
                "ANC",       # Ancestry recode
                "NATIVITY",  # Nativity
                "DEAR",      # Hearing difficulty
                "DEYE",      # Vision difficulty
                "DREM",      # Cognitive difficulty
                "SEX",       # Sex
                "RAC1P"      # Race
            ],
            "numerical_columns": [
                "AGEP"       # Age
            ],
            "label_column": "label",  # Employment status (1=employed, 0=not employed)
            "record_id_column": "record_id",
            "state": "CA",  # California - can be changed for different distributions
            "survey_year": "2018",
            "target_description": "Target values: 0 (not employed) or 1 (employed)",
            "anonymization": {
                "no": [],
                "basic": ["AGEP", "SEX", "RAC1P"], 
                "all": None 
            }
        }
    }
    

    @classmethod
    def get_available_datasets(cls):
        """Dynamically detect available datasets from spalten directory."""
        import os
        from pathlib import Path
        
        # Get the spalten directory path
        spalten_dir = Path(__file__).parent / "spalten"
        
        if not spalten_dir.exists():
            return []
        
        datasets = []
        for file in spalten_dir.glob("*.py"):
            # Skip __init__ and PrivacyLevel
            if file.stem in ["__init__", "PrivacyLevel"]:
                continue
            # Convert to lowercase (e.g., Adult.py -> adult)
            datasets.append(file.stem.lower())
        
        return sorted(datasets)

    @classmethod
    def get_config(cls, dataset_name):
        """Get configuration for a specific dataset"""
        if dataset_name not in cls._CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(cls._CONFIGS.keys())}")
        
        return cls._CONFIGS[dataset_name]

    @classmethod
    def get_categorical_columns(cls, dataset_name):
        """Get categorical columns for a specific dataset"""
        return cls.get_config(dataset_name)["categorical_columns"]
    
    @classmethod
    def get_numerical_columns(cls, dataset_name):
        """Get numerical columns for a specific dataset"""
        return cls.get_config(dataset_name)["numerical_columns"]
    
    @classmethod
    def get_label_column(cls, dataset_name):
        """Get label column for a specific dataset"""
        return cls.get_config(dataset_name)["label_column"]

    @classmethod
    def get_record_id_column(cls, dataset_name):
        """Get label column for a specific dataset"""
        return cls.get_config(dataset_name)["record_id_column"]
    
    @classmethod
    def get_spalten_classes(cls, dataset_name):
        """Get a dict and list of all column classes for the given dataset."""
        available = cls.get_available_datasets()
        if dataset_name not in available:
            raise ValueError(f"No Spalten mapping for dataset: {dataset_name}. Available: {available}")
        module = importlib.import_module(f"src.spalten.{dataset_name.capitalize()}")
        spalten_dict = {getattr(obj, 'name'): obj for obj in module.__dict__.values() if hasattr(obj, 'name') and not getattr(obj, 'name').startswith('src.spalten.')}
        spalten_list = list(spalten_dict.values())
        return spalten_dict, spalten_list

    @classmethod
    def get_spalten_class(cls, dataset_name):
        """Legacy: Return just the list of column classes for compatibility."""
        return cls.get_spalten_classes(dataset_name)[1]
    
    @classmethod
    def get_anonymization_class(cls, dataset_name):
        """Get dataset-specific Anonymization enum class (using new spalten_dict)."""
        # Get spalten mapping
        spalten_dict, spalten_list = cls.get_spalten_classes(dataset_name)
        # Get anonymization config
        anon_config = cls._CONFIGS[dataset_name]["anonymization"]
        # Dynamically set 'all' to all available column names
        anon_config["all"] = list(spalten_dict.keys())
        # Create mapping for Anonymization enum
        mapping = {}
        for level, columns in anon_config.items():
            if columns is None:
                columns = anon_config["all"]
            # Convert column names to actual column classes
            spalten_values = [spalten_dict[col] for col in columns] if columns else []
            mapping[level] = spalten_values
        # Create the Anonymization enum
        anon_class = Enum(f"{dataset_name.capitalize()}Anonymization", mapping)
        return anon_class
