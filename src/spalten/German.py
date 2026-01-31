from .PrivacyLevel import PrivacyLevel

# --- Age ---
class Age:
    name = "age"
    # Level 1: Fine-grained age ranges
    dict_level1 = {
        "[0-9]": range(int(0), int(9)+1),
        "[10-19]": range(int(10), int(19)+1),
        "[20-29]": range(int(20), int(29)+1),
        "[30-39]": range(int(30), int(39)+1),
        "[40-49]": range(int(40), int(49)+1),
        "[50-59]": range(int(50), int(59)+1),
        "[60-69]": range(int(60), int(69)+1),
        "[70-79]": range(int(70), int(79)+1),
        "[80-90]": range(int(80), int(90)+1)
    }
    # Level 2: Coarse age groups with semantic labels
    dict_level2 = {
        "young": range(int(0), int(29)+1),
        "middle": range(int(30), int(59)+1),
        "old": range(int(60), int(90)+1)
    }
    # Level 3: Complete suppression
    dict_level3 = {
        "?": range(int(0), int(90)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2,
        PrivacyLevel.LEVEL3: dict_level3
    }
    @staticmethod
    def get_value(key):
        for d in Age.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Age.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Age.name}")
    @staticmethod
    def is_generalized(key):
        for d in Age.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Age.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Age.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL3 or privacy_level == PrivacyLevel.LEVEL2:
            return value
        if privacy_level == PrivacyLevel.LEVEL1:
            original_values = Age.get_value(value)
            if original_values is not None:
                original_value = list(original_values)[0]
                return Age.get_key(original_value, PrivacyLevel.LEVEL2)
            else:
                raise ValueError(f"Value {value} not found in LEVEL1 for Age")
        if privacy_level == PrivacyLevel.LEVEL0:
            return Age.get_key(value, PrivacyLevel.LEVEL2)

# --- Duration ---
class Duration:
    name = "duration"
    dict_level1 = {
        "[4-12]": range(int(4), int(12)+1),
        "[13-24]": range(int(13), int(24)+1),
        "[25-36]": range(int(25), int(36)+1),
        "[37-48]": range(int(37), int(48)+1),
        "[49-72]": range(int(49), int(72)+1)
    }
    dict_level2 = {
        "?": range(int(4), int(72)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2,
    }
    @staticmethod
    def get_value(key):
        for d in Duration.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Duration.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Duration.name}")
    @staticmethod
    def is_generalized(key):
        for d in Duration.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Duration.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Duration.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL3 or privacy_level == PrivacyLevel.LEVEL2:
            return value
        if privacy_level == PrivacyLevel.LEVEL1:
            original_values = Duration.get_value(value)
            if original_values is not None:
                original_value = list(original_values)[0]
                return Duration.get_key(original_value, PrivacyLevel.LEVEL2)
            else:
                raise ValueError(f"Value {value} not found in LEVEL1 for Duration")
        if privacy_level == PrivacyLevel.LEVEL0:
            return Duration.get_key(value, PrivacyLevel.LEVEL2)

# --- CreditAmount ---
class CreditAmount:
    name = "credit_amount"
    dict_level1 = {
        "[250-2000]": range(int(250), int(2000)+1),
        "[2001-5000]": range(int(2001), int(5000)+1),
        "[5001-10000]": range(int(5001), int(10000)+1),
        "[10001-20000]": range(int(10001), int(20000)+1)
    }
    dict_level2 = {
        "?": range(int(250), int(20000)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2,
    }
    @staticmethod
    def get_value(key):
        for d in CreditAmount.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in CreditAmount.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {CreditAmount.name}")
    @staticmethod
    def is_generalized(key):
        for d in CreditAmount.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in CreditAmount.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = CreditAmount.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL3 or privacy_level == PrivacyLevel.LEVEL2:
            return value
        if privacy_level == PrivacyLevel.LEVEL1:
            original_values = CreditAmount.get_value(value)
            if original_values is not None:
                original_value = list(original_values)[0]
                return CreditAmount.get_key(original_value, PrivacyLevel.LEVEL2)
            else:
                raise ValueError(f"Value {value} not found in LEVEL1 for CreditAmount")
        if privacy_level == PrivacyLevel.LEVEL0:
            return CreditAmount.get_key(value, PrivacyLevel.LEVEL2)

# --- InstallmentRate ---
class InstallmentRate:
    name = "installment_rate"
    dict_level1 = {
        "low": range(int(1), int(2)+1),
        "high": range(int(3), int(4)+1)
    }
    dict_level2 = {
        "?": range(int(1), int(4)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in InstallmentRate.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in InstallmentRate.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {InstallmentRate.name}")
    @staticmethod
    def is_generalized(key):
        for d in InstallmentRate.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in InstallmentRate.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = InstallmentRate.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return InstallmentRate.get_key(value, PrivacyLevel.LEVEL1)

# --- PresentResidence ---
class PresentResidence:
    name = "present_residence"
    dict_level1 = {
        "short": range(int(1), int(2)+1),
        "long": range(int(3), int(4)+1)
    }
    dict_level2 = {
        "?": range(int(1), int(4)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in PresentResidence.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in PresentResidence.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {PresentResidence.name}")
    @staticmethod
    def is_generalized(key):
        for d in PresentResidence.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in PresentResidence.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = PresentResidence.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return PresentResidence.get_key(value, PrivacyLevel.LEVEL1)

# --- NumberCredits ---
class NumberCredits:
    name = "number_credits"
    dict_level1 = {
        "low": range(int(1), int(2)+1),
        "high": range(int(3), int(4)+1)
    }
    dict_level2 = {
        "?": range(int(1), int(4)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in NumberCredits.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in NumberCredits.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {NumberCredits.name}")
    @staticmethod
    def is_generalized(key):
        for d in NumberCredits.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in NumberCredits.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = NumberCredits.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return NumberCredits.get_key(value, PrivacyLevel.LEVEL1)

# --- PeopleLiable ---
class PeopleLiable:
    name = "people_liable"
    dict_level1 = {
        "?": range(int(1), int(2)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in PeopleLiable.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in PeopleLiable.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {PeopleLiable.name}")
    @staticmethod
    def is_generalized(key):
        for d in PeopleLiable.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in PeopleLiable.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = PeopleLiable.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- Status (Checking account status) ---
class Status:
    name = "status"
    dict_level1 = {
        "low": ["no_account", "<0_DM", "0-200_DM"],
        "good": [">=200_DM"],
    }
    dict_level2 = {
        "?": ["<0_DM", "0-200_DM", ">=200_DM", "no_account"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in Status.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Status.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Status.name}")
    @staticmethod
    def is_generalized(key):
        for d in Status.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Status.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Status.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return Status.get_key(value, PrivacyLevel.LEVEL1)

# --- CreditHistory ---
class CreditHistory:
    name = "credit_history"
    dict_level1 = {
        "good": ["no_credits", "all_paid", "existing_paid"],
        "problematic": ["delay", "critical"]
    }
    dict_level2 = {
        "?": ["no_credits", "all_paid", "existing_paid", "delay", "critical"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in CreditHistory.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in CreditHistory.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {CreditHistory.name}")
    @staticmethod
    def is_generalized(key):
        for d in CreditHistory.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in CreditHistory.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = CreditHistory.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return CreditHistory.get_key(value, PrivacyLevel.LEVEL1)

# --- Purpose ---
class Purpose:
    name = "purpose"
    dict_level1 = {
        "vehicle": ["car_new", "car_used"],
        "household": ["furniture", "radio_tv", "appliances"],
        "personal": ["repairs", "education", "retraining"],
        "other": ["business", "others"],
    }
    dict_level2 = {
        "?": ["car_new", "car_used", "furniture", "radio_tv", "appliances", 
              "repairs", "education", "retraining", "business", "others"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in Purpose.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Purpose.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Purpose.name}")
    @staticmethod
    def is_generalized(key):
        for d in Purpose.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Purpose.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Purpose.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return Purpose.get_key(value, PrivacyLevel.LEVEL1)

# --- Savings ---
class Savings:
    name = "savings"
    dict_level1 = {
        "low": ["<100_DM", "unknown", "100-500_DM"],
        "high": ["500-1000_DM", ">=1000_DM"],
    }
    dict_level2 = {
        "?": ["<100_DM", "100-500_DM", "500-1000_DM", ">=1000_DM", "unknown"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in Savings.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Savings.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Savings.name}")
    @staticmethod
    def is_generalized(key):
        for d in Savings.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Savings.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Savings.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return Savings.get_key(value, PrivacyLevel.LEVEL1)

# --- EmploymentSince ---
class EmploymentSince:
    name = "employment_since"
    dict_level1 = {
        "short": ["unemployed", "<1_year", "1-4_years"],
        "long": ["4-7_years", ">=7_years"]
    }
    dict_level2 = {
        "?": ["unemployed", "<1_year", "1-4_years", "4-7_years", ">=7_years"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in EmploymentSince.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in EmploymentSince.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {EmploymentSince.name}")
    @staticmethod
    def is_generalized(key):
        for d in EmploymentSince.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in EmploymentSince.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = EmploymentSince.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return EmploymentSince.get_key(value, PrivacyLevel.LEVEL1)

# --- PersonalStatusSex ---
class PersonalStatusSex:
    name = "personal_status_sex"
    dict_level1 = {
        "male": ["male_divorced", "male_single", "male_married"],
        "female": ["female_divorced_married", "female_single"]
    }
    dict_level2 = {
        "?": ["male_divorced", "female_divorced_married", "male_single", "male_married", "female_single"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in PersonalStatusSex.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in PersonalStatusSex.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {PersonalStatusSex.name}")
    @staticmethod
    def is_generalized(key):
        for d in PersonalStatusSex.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in PersonalStatusSex.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = PersonalStatusSex.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return PersonalStatusSex.get_key(value, PrivacyLevel.LEVEL1)

# --- OtherDebtors ---
class OtherDebtors:
    name = "other_debtors"
    dict_level1 = {
        "?": ["none", "co_applicant", "guarantor"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in OtherDebtors.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in OtherDebtors.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {OtherDebtors.name}")
    @staticmethod
    def is_generalized(key):
        for d in OtherDebtors.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in OtherDebtors.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = OtherDebtors.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- Property ---
class Property:
    name = "property"
    dict_level1 = {
        "?": ["real_estate", "building_society", "car_other", "unknown"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
    }
    @staticmethod
    def get_value(key):
        for d in Property.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Property.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Property.name}")
    @staticmethod
    def is_generalized(key):
        for d in Property.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Property.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Property.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return Property.get_key(value, PrivacyLevel.LEVEL1)

# --- OtherInstallmentPlans ---
class OtherInstallmentPlans:
    name = "other_installment_plans"
    dict_level1 = {
        "?": ["bank", "stores", "none"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in OtherInstallmentPlans.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in OtherInstallmentPlans.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {OtherInstallmentPlans.name}")
    @staticmethod
    def is_generalized(key):
        for d in OtherInstallmentPlans.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in OtherInstallmentPlans.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = OtherInstallmentPlans.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- Housing ---
class Housing:
    name = "housing"
    dict_level1 = {
        "?": ["rent", "own", "free"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in Housing.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Housing.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Housing.name}")
    @staticmethod
    def is_generalized(key):
        for d in Housing.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Housing.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Housing.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- Job ---
class Job:
    name = "job"
    dict_level1 = {
        "unskilled": ["unemployed_unskilled", "unskilled"],
        "skilled": ["skilled", "management"]
    }
    dict_level2 = {
        "?": ["unemployed_unskilled", "unskilled", "skilled", "management"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in Job.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Job.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Job.name}")
    @staticmethod
    def is_generalized(key):
        for d in Job.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Job.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Job.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return Job.get_key(value, PrivacyLevel.LEVEL1)

# --- Telephone ---
class Telephone:
    name = "telephone"
    dict_level1 = {
        "?": ["none", "yes"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in Telephone.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Telephone.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Telephone.name}")
    @staticmethod
    def is_generalized(key):
        for d in Telephone.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Telephone.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Telephone.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- ForeignWorker ---
class ForeignWorker:
    name = "foreign_worker"
    dict_level1 = {
        "?": ["yes", "no"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in ForeignWorker.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in ForeignWorker.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {ForeignWorker.name}")
    @staticmethod
    def is_generalized(key):
        for d in ForeignWorker.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in ForeignWorker.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = ForeignWorker.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value
