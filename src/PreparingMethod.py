from enum import Enum

class PreparingMethod(Enum):
    original = "original"
    no_preprocessing = "no preprocessing"
    forced_generalization = "forced generalization"
    specialization = "specialization"
    weighted_specialization = "weighted specialization"
    weighted_specialization_highest_confidence = "highest confidence"
    extended_weighted_specialization = "extended weighted specialization"
    