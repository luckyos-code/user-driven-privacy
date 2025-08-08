from .PrivacyLevel import PrivacyLevel

# --- Age ---
class Age:
    name = "Age"
    dict_level1 = {
        "jung": range(int(1), int(4)+1),
        "mittel": range(int(5), int(9)+1),
        "alt": range(int(10), int(13)+1)
    }
    dict_level2 = {
        "?": range(int(1), int(13)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
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
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return Age.get_key(value, PrivacyLevel.LEVEL1)

# --- PhysHlth ---
class PhysHlth:
    name = "PhysHlth"
    dict_level1 = {
        "wenig": range(int(0), int(10)+1),
        "mittel": range(int(11), int(20)+1),
        "viel": range(int(21), int(30)+1)
    }
    dict_level2 = {
        "?": range(int(0), int(30)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in PhysHlth.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in PhysHlth.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {PhysHlth.name} und im PrivacyLevel {privacy_level}")
    @staticmethod
    def is_generalized(key):
        for d in PhysHlth.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in PhysHlth.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = PhysHlth.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return PhysHlth.get_key(value, PrivacyLevel.LEVEL1)

# --- Sex ---
class Sex:
    name = "Sex"
    dict_level1 = {
        "?": [0, 1]
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
            if int(value) in rng:
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

# --- Smoker ---
class Smoker:
    name = "Smoker"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in Smoker.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Smoker.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Smoker.name}")
    @staticmethod
    def is_generalized(key):
        for d in Smoker.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Smoker.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Smoker.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- Stroke ---
class Stroke:
    name = "Stroke"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in Stroke.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Stroke.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Stroke.name}")
    @staticmethod
    def is_generalized(key):
        for d in Stroke.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Stroke.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Stroke.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- Veggies ---
class Veggies:
    name = "Veggies"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in Veggies.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Veggies.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Veggies.name}")
    @staticmethod
    def is_generalized(key):
        for d in Veggies.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Veggies.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Veggies.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- CholCheck ---
class CholCheck:
    name = "CholCheck"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in CholCheck.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in CholCheck.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {CholCheck.name}")
    @staticmethod
    def is_generalized(key):
        for d in CholCheck.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in CholCheck.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = CholCheck.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- DiffWalk ---
class DiffWalk:
    name = "DiffWalk"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in DiffWalk.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in DiffWalk.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {DiffWalk.name}")
    @staticmethod
    def is_generalized(key):
        for d in DiffWalk.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in DiffWalk.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = DiffWalk.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- BMI ---
class BMI:
    name = "BMI"
    dict_level1 = {
        "untergewicht": range(int(12), int(19)+1),
        "normal": range(int(20), int(25)+1),
        "uebergewicht": range(int(26), int(30)+1),
        "adipositas1": range(int(31), int(35)+1),
        "adipositas2": range(int(36), int(40)+1),
        "[41-50]": range(int(41), int(50)+1),
        "[51-60]": range(int(51), int(60)+1),
        "[61-70]": range(int(61), int(70)+1),
        "[71-80]": range(int(71), int(80)+1),
        "[>80]": range(int(81), int(98)+1)
    }
    dict_level2 = {
        "gering": range(int(12), int(25)+1),
        "mittel": range(int(26), int(40)+1),
        "hoch": range(int(41), int(98)+1)
    }
    dict_level3 = {
        "?": range(int(12), int(98)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2,
        PrivacyLevel.LEVEL3: dict_level3
    }
    @staticmethod
    def get_value(key):
        for d in BMI.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in BMI.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {BMI.name}")
    @staticmethod
    def is_generalized(key):
        for d in BMI.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in BMI.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = BMI.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL3 or privacy_level == PrivacyLevel.LEVEL2:
            return value
        if privacy_level == PrivacyLevel.LEVEL1:
            original_values = BMI.get_value(value)
            if original_values is not None:
                original_value = list(original_values)[0]
                return BMI.get_key(original_value, PrivacyLevel.LEVEL2)
            else:
                raise ValueError(f"Value {value} not found in LEVEL1 for BMI")
        if privacy_level == PrivacyLevel.LEVEL0:
            return BMI.get_key(value, PrivacyLevel.LEVEL2)

# --- Education ---
class Education:
    name = "Education"
    dict_level1 = {
        "schlecht": range(int(1), int(3)+1),
        "gut": range(int(4), int(6)+1)
    }
    dict_level2 = {
        "?": range(int(1), int(6)+1)
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
            if int(value) in rng:
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
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- AnyHealthcare ---
class AnyHealthcare:
    name = "AnyHealthcare"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in AnyHealthcare.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in AnyHealthcare.dict_all[privacy_level].items():
            if value in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {AnyHealthcare.name}")
    @staticmethod
    def is_generalized(key):
        for d in AnyHealthcare.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in AnyHealthcare.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = AnyHealthcare.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- Fruits ---
class Fruits:
    name = "Fruits"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in Fruits.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Fruits.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Fruits.name}")
    @staticmethod
    def is_generalized(key):
        for d in Fruits.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Fruits.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Fruits.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- GenHlth ---
class GenHlth:
    name = "GenHlth"
    dict_level1 = {
        "?": range(int(1), int(5)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in GenHlth.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in GenHlth.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {GenHlth.name}")
    @staticmethod
    def is_generalized(key):
        for d in GenHlth.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in GenHlth.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = GenHlth.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- HeartDiseaseorAttack ---
class HeartDiseaseorAttack:
    name = "HeartDiseaseorAttack"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in HeartDiseaseorAttack.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in HeartDiseaseorAttack.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {HeartDiseaseorAttack.name}")
    @staticmethod
    def is_generalized(key):
        for d in HeartDiseaseorAttack.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in HeartDiseaseorAttack.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = HeartDiseaseorAttack.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- HighChol ---
class HighChol:
    name = "HighChol"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in HighChol.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in HighChol.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {HighChol.name}")
    @staticmethod
    def is_generalized(key):
        for d in HighChol.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in HighChol.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = HighChol.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- HighBP ---
class HighBP:
    name = "HighBP"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in HighBP.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in HighBP.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {HighBP.name}")
    @staticmethod
    def is_generalized(key):
        for d in HighBP.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in HighBP.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = HighBP.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- HvyAlcoholConsump ---
class HvyAlcoholConsump:
    name = "HvyAlcoholConsump"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in HvyAlcoholConsump.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in HvyAlcoholConsump.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {HvyAlcoholConsump.name}")
    @staticmethod
    def is_generalized(key):
        for d in HvyAlcoholConsump.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in HvyAlcoholConsump.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = HvyAlcoholConsump.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- Income ---
class Income:
    name = "Income"
    dict_level1 = {
        "wenig": range(int(1), int(4)+1),
        "viel": range(int(5), int(8)+1)
    }
    dict_level2 = {
        "?": range(int(1), int(8)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in Income.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in Income.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {Income.name}")
    @staticmethod
    def is_generalized(key):
        for d in Income.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in Income.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = Income.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- MentHlth ---
class MentHlth:
    name = "MentHlth"
    dict_level1 = {
        "wenig": range(int(0), int(10)+1),
        "mittel": range(int(11), int(20)+1),
        "viel": range(int(21), int(30)+1)
    }
    dict_level2 = {
        "?": range(int(0), int(30)+1)
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    @staticmethod
    def get_value(key):
        for d in MentHlth.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in MentHlth.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {MentHlth.name} und im PrivacyLevel {privacy_level}")
    @staticmethod
    def is_generalized(key):
        for d in MentHlth.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in MentHlth.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = MentHlth.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        if privacy_level == PrivacyLevel.LEVEL0:
            return MentHlth.get_key(value, PrivacyLevel.LEVEL1)

# --- NoDocbcCost ---
class NoDocbcCost:
    name = "NoDocbcCost"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in NoDocbcCost.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in NoDocbcCost.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {NoDocbcCost.name}")
    @staticmethod
    def is_generalized(key):
        for d in NoDocbcCost.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in NoDocbcCost.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = NoDocbcCost.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

# --- PhysActivity ---
class PhysActivity:
    name = "PhysActivity"
    dict_level1 = {
        "?": [0, 1]
    }
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    @staticmethod
    def get_value(key):
        for d in PhysActivity.dict_all.values():
            if key in d:
                return d[key]
        return None
    @staticmethod
    def get_key(value, privacy_level):
        for key, rng in PhysActivity.dict_all[privacy_level].items():
            if int(value) in rng:
                return key
        raise ValueError(f"Der Wert {value} vom Typ {type(value)} wurde nicht gefunden in der Rubrik {PhysActivity.name}")
    @staticmethod
    def is_generalized(key):
        for d in PhysActivity.dict_all.values():
            if key in d:
                return True
        return False
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in PhysActivity.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = PhysActivity.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value 