from enum import Enum

class PreparingMethod(Enum):
    original = "original"
    no_preprocessing = "no preprocessing"
    forced_generalization = "forced generalization"
    specialization = "specialization" # deprecated
    weighted_specialization = "weighted specialization"
    weighted_specialization_highest_confidence = "highest confidence" # deprecated
    extended_weighted_specialization = "extended weighted specialization" # deprecated
    baseline_imputation = "baseline_imputation"  # Runs all imputation baselines
    llm_imputation = "llm_imputation"  # Loads LLM-imputed data from files
    # Note: LLM prediction is handled by LLM evaluation code and just merged into the results later
    