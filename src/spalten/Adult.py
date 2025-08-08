from .PrivacyLevel import PrivacyLevel

# --- Age ---
class Age:
    name = "age"
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
    dict_level2 = {
        "jung": range(int(0), int(29)+1),
        "mittel": range(int(30), int(59)+1),
        "alt": range(int(60), int(90)+1)
    }
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

# --- CapitalLoss ---
class CapitalLoss:
    name = "capital-loss"
    dict_level1 = {
        "[0-99]": range(int(0), int(99)+1),
        "[100-999]": range(int(100), int(999)+1),
        "[1000-1999]": range(int(1000), int(1999)+1),
        "[2000-4356]": range(int(2000), int(4356)+1)
    }
    dict_level2 = {
        "?": range(int(0), int(4356)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in CapitalLoss.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in CapitalLoss.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {CapitalLoss.name}")
    @staticmethod
    def is_generalized(key):
        for d in CapitalLoss.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in CapitalLoss.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = CapitalLoss.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return CapitalLoss.get_key(value, PrivacyLevel.LEVEL1)

# --- CapitalGain ---
class CapitalGain:
    name = "capital-gain"
    dict_level1 = {
        "[0-99]": range(int(0), int(99)+1),
        "[100-999]": range(int(100), int(999)+1),
        "[1000-1999]": range(int(1000), int(1999)+1),
        "[2000-4999]": range(int(2000), int(4999)+1),
        "[5000-9999]": range(int(5000), int(9999)+1),
        "[10000-19999]": range(int(10000), int(19999)+1),
        "[20000-99999]": range(int(20000), int(99999)+1)
    }
    dict_level2 = {
        "?": range(int(0), int(99999)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in CapitalGain.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in CapitalGain.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {CapitalGain.name}")
    @staticmethod
    def is_generalized(key):
        for d in CapitalGain.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in CapitalGain.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = CapitalGain.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return CapitalGain.get_key(value, PrivacyLevel.LEVEL1)

# --- Education ---
class Education:
    name = "education"
    dict_level1 = {
        "elementary_school": ["Preschool", "1st-4th", "5th-6th", "7th-8th"],
        "high_school": ["9th", "10th", "11th", "12th"],
        "high_school_graduate": ["HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm"],
        "bachelor_and_higher": ["Bachelors", "Masters", "Prof-school", "Doctorate"]
    }
    dict_level2 = {
        "?": ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in Education.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Education.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Education.name}")
    @staticmethod
    def is_generalized(key):
        for d in Education.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Education.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Education.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return Education.get_key(value, PrivacyLevel.LEVEL1)

# --- EducationNum ---
# class EducationNum:
#     name = "education-num"
#     dict_level1 = {
#         "[1-4]": range(int(1), int(4)+1),
#         "[5-8]": range(int(5), int(8)+1),
#         "[9-12]": range(int(9), int(12)+1),
#         "[13-16]": range(int(13), int(16)+1)
#     }
#     dict_level2 = {
#         "?": range(int(1), int(16)+1)
#     }
#     dict_all = {
#         PrivacyLevel.LEVEL1: dict_level1,
#         PrivacyLevel.LEVEL2: dict_level2
#     }
#     @staticmethod
#     def get_value(key):
#         for d in EducationNum.dict_all.values():
#             if key in d:
#                 return d[key]
#         return None
#     @staticmethod
#     def get_key(value, privacy_level):
#         for key, rng in EducationNum.dict_all[privacy_level].items():
#             if int(value) in rng:
#                 return key
#         raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {EducationNum.name}")
#     @staticmethod
#     def is_generalized(key):
#         for d in EducationNum.dict_all.values():
#             if key in d:
#                 return True
#         return False
#     @staticmethod
#     def get_privacy_level_for_value(value):
#         for privacy_level, values in EducationNum.dict_all.items():
#             if value in values.keys():
#                 return privacy_level
#         return PrivacyLevel.LEVEL0
#     # education_num_mapping is not a privacy dict, but a mapping for education string to num
#     education_num_mapping = {
#         "Preschool": 1,
#         "1st-4th": 2,
#         "5th-6th": 3,
#         "7th-8th": 4,
#         "9th": 5,
#         "10th": 6,
#         "11th": 7,
#         "12th": 8,
#         "HS-grad": 9,
#         "Some-college": 10,
#         "Assoc-voc": 11,
#         "Assoc-acdm": 12,
#         "Bachelors": 13,
#         "Masters": 14,
#         "Prof-school": 15,
#         "Doctorate": 16
#     }

# --- FnlWgt ---
class FnlWgt:
    name = "fnlwgt"
    dict_level1 = {
        "[13492-49999]": range(int(13492), int(49999)+1),
        "[50000-99999]": range(int(50000), int(99999)+1),
        "[100000-499999]": range(int(100000), int(499999)+1),
        "[500000-1490400]": range(int(500000), int(1490400)+1)
    }
    dict_level2 = {
        "?": range(int(13492), int(1490400)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in FnlWgt.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in FnlWgt.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {FnlWgt.name}")
    @staticmethod
    def is_generalized(key):
        for d in FnlWgt.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in FnlWgt.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = FnlWgt.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return FnlWgt.get_key(value, PrivacyLevel.LEVEL1)

# --- HoursPerWeek ---
class HoursPerWeek:
    name = "hours-per-week"
    dict_level1 = {
        "[0-9]": range(int(0), int(9)+1),
        "[10-19]": range(int(10), int(19)+1),
        "[20-29]": range(int(20), int(29)+1),
        "[30-39]": range(int(30), int(39)+1),
        "[40-49]": range(int(40), int(49)+1),
        "[50-59]": range(int(50), int(59)+1),
        "[60-69]": range(int(60), int(69)+1),
        "[70-79]": range(int(70), int(79)+1),
        "[80-89]": range(int(80), int(89)+1),
        "[90-99]": range(int(90), int(99)+1)
    }
    dict_level2 = {
        "wenig": range(int(0), int(29)+1),
        "mittel": range(int(30), int(59)+1),
        "viel": range(int(60), int(99)+1)
    }
    dict_level3 = {
        "?": range(int(0), int(99)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2,
        PrivacyLevel.LEVEL3: dict_level3
    }
    @staticmethod
    def get_value(key):
        for d in HoursPerWeek.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in HoursPerWeek.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {HoursPerWeek.name}")
    @staticmethod
    def is_generalized(key):
        for d in HoursPerWeek.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in HoursPerWeek.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = HoursPerWeek.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL3 or privacy_level == PrivacyLevel.LEVEL2:
            return value
        if privacy_level == PrivacyLevel.LEVEL1:
            original_values = HoursPerWeek.get_value(value)
            if original_values is not None:
                original_value = list(original_values)[0]
                return HoursPerWeek.get_key(original_value, PrivacyLevel.LEVEL2)
            else:
                raise ValueError(f"Value {value} not found in LEVEL1 for HoursPerWeek")
        if privacy_level == PrivacyLevel.LEVEL0:
            return HoursPerWeek.get_key(value, PrivacyLevel.LEVEL2)

# --- MaritalStatus ---
class MaritalStatus:
    name = "marital-status"
    dict_level1 = {
        "married": ['Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent'],
        "not_married": ['Never-married', 'Widowed'],
        "separeted_divorced": ['Separated', 'Divorced']
    }
    dict_level2 = {
        "?": ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in MaritalStatus.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in MaritalStatus.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {MaritalStatus.name}")
    @staticmethod
    def is_generalized(key):
        for d in MaritalStatus.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in MaritalStatus.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = MaritalStatus.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return MaritalStatus.get_key(value, PrivacyLevel.LEVEL1)

# --- NativeCountry ---
class NativeCountry:
    name = "native-country"
    dict_level1 = {
        "Europa": ['Germany', 'Greece', 'England', 'Italy', 'Poland', 'Portugal', 'Ireland', 'France', 'Hungary', 'Holand-Netherlands', 'Scotland', 'Yugoslavia', ],
        "Amerika": ['United-States', 'Canada', 'Cuba', 'Mexico', 'Ecuador', 'Puerto-Rico', 'Outlying-US(Guam-USVI-etc)', 'Columbia', 'Nicaragua', 'Peru', 'Honduras', 'Jamaica', 'Dominican-Republic', 'El-Salvador', 'Haiti', 'Guatemala', 'Trinadad&Tobago'],
        "Asien": ['India', 'Japan', 'China', 'Iran', 'Philippines', 'Vietnam', 'Taiwan', 'Cambodia', 'Thailand', 'Hong', 'Laos', ],
        "Afrika": ['South']
    }
    dict_level2 = {
        "?": ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in NativeCountry.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in NativeCountry.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {NativeCountry.name}")
    @staticmethod
    def is_generalized(key):
        for d in NativeCountry.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in NativeCountry.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = NativeCountry.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return NativeCountry.get_key(value, PrivacyLevel.LEVEL1)

# --- Occupation ---
class Occupation:
    name = "occupation"
    dict_level1 = {
        "serives": ['Handlers-cleaners', 'Priv-house-serv', 'Other-service'],
        "craft": ['Craft-repair', 'Farming-fishing', 'Transport-moving'],
        "tech": ['Tech-support', 'Machine-op-inspct'],
        "professional_management": ['Prof-specialty', 'Exec-managerial'],
        "protective": ['Protective-serv', 'Armed-Forces'],
        "admin_and_sales": ['Adm-clerical', 'Sales']
    }
    dict_level2 = {
        "?": ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving']
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in Occupation.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Occupation.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Occupation.name}")
    @staticmethod
    def is_generalized(key):
        for d in Occupation.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Occupation.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Occupation.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return Occupation.get_key(value, PrivacyLevel.LEVEL1)

# --- Race ---
class Race:
    name = "race"
    dict_level1 = {
        "?": ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in Race.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Race.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Race.name}")
    @staticmethod
    def is_generalized(key):
        for d in Race.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Race.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Race.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- Relationship ---
class Relationship:
    name = "relationship"
    dict_level1 = {
        "related": ['Husband', 'Other-relative', 'Own-child', 'Wife'],
        "not_related": ['Not-in-family', 'Unmarried']
    }
    dict_level2 = {
        "?": ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in Relationship.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Relationship.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Relationship.name}")
    @staticmethod
    def is_generalized(key):
        for d in Relationship.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Relationship.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Relationship.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return Relationship.get_key(value, PrivacyLevel.LEVEL1)

# --- Sex ---
class Sex:
    name = "sex"
    dict_level1 = {
        "?": ['Male', 'Female']
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in Sex.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Sex.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Sex.name}")
    @staticmethod
    def is_generalized(key):
        for d in Sex.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Sex.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Sex.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- Workclass ---
class Workclass:
    name = "workclass"
    dict_level1 = {
        "private_sector": ['Private'],
        "self_emp": ['Self-emp-inc', 'Self-emp-not-inc'],
        "gov": ['Federal-gov', 'Local-gov', 'State-gov'],
        "without_pay": ['Without-pay']
    }
    dict_level2 = {
        "?": ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in Workclass.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Workclass.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Workclass.name}")
    @staticmethod
    def is_generalized(key):
        for d in Workclass.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Workclass.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Workclass.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return Workclass.get_key(value, PrivacyLevel.LEVEL1) 