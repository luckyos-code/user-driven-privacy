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
            "anonymization": {
                "no": [],                                    # No anonymization
                "basic": ["age", "sex", "race"],             # Basic demographic info
                # "moderate": ["AGE", "SEX", "RACE", "NATIVE_COUNTRY", "EDUCATION", "OCCUPATION"],  # Extended demographics
                # "strong": ["AGE", "SEX", "RACE", "NATIVE_COUNTRY", "EDUCATION", "OCCUPATION",
                #           "MARITAL_STATUS", "CAPITAL_GAIN", "CAPITAL_LOSS", "HOURS_PER_WEEK"],    # Most attributes
                "all": None  # Will be populated dynamically with all columns
            }
        },
        "diabetes": {
            "categorical_columns": ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
                                    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
                                    "NoDocbcCost", "DiffWalk", "Sex"],
            "numerical_columns": ["Age", "BMI", "GenHlth", "MentHlth", "PhysHlth", "Education", "Income"],
            "label_column": 'Diabetes_binary',
            "record_id_column": 'record_id',
            "anonymization": {
                "no": [], 
                "basic": ["Age", "Sex", "Income", "Education"],
                # "moderate": ["AGE", "SEX", "INCOME", "EDUCATION", "BMI", "HIGH_BP", "HIGH_CHOL", "STROKE",
                #              "HEART_DISEASEOR_ATTACK"],      # Demographics + key health indicators
                # "strong": ["AGE", "SEX", "INCOME", "EDUCATION", "BMI", "HIGH_BP", "HIGH_CHOL", "STROKE",
                #            "HEART_DISEASEOR_ATTACK", "SMOKER", "PHYS_ACTIVITY", "FRUITS", "VEGGIES",
                #            "HVY_ALCOHOL_CONSUMP", "GEN_HLTH", "MENT_HLTH", "PHYS_HLTH", "DIFF_WALK"],
                "all": None
            }
        },
        "german_credit": {
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
                "basic": ["AGE", "JOB", "HOUSING"],
                "all": None
            },
            # Optional: mapping for categorical codes (example, not exhaustive)
            "mappings": {
                "status": {
                    "A11": "...<0 DM", "A12": "0<=X<200 DM", "A13": ">=200 DM", "A14": "no checking account"
                },
                "credit_history": {
                    "A30": "no credits/all paid", "A31": "all paid", "A32": "existing paid", "A33": "delay", "A34": "critical"
                },
                # ... add more mappings as needed ...
            }
        }
    }
    

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
        if dataset_name not in ['adult', 'diabetes']:
            raise ValueError(f"No Spalten mapping for dataset: {dataset_name}")
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
