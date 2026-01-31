from .PrivacyLevel import PrivacyLevel


# ==============================================================================
# NUMERICAL SPALTE: AGE (AGEP)
# ==============================================================================

class AGEP:
    name = "AGEP"
    
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
        for d in AGEP.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        for key, val_range in AGEP.dict_all[privacy_level].items():
            if value in val_range:
                return key
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in AGEP.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in AGEP.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = AGEP.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL3 or privacy_level == PrivacyLevel.LEVEL2:
            return value
        if privacy_level == PrivacyLevel.LEVEL1:
            original_values = AGEP.get_value(value)
            if original_values is not None:
                original_value = list(original_values)[0]
                return AGEP.get_key(original_value, PrivacyLevel.LEVEL2)
        # LEVEL0 - map to LEVEL1
        return AGEP.get_key(value, PrivacyLevel.LEVEL1)


# ==============================================================================
# CATEGORICAL SPALTE: EDUCATION (SCHL)
# ==============================================================================

class SCHL:
    name = "SCHL"
    
    # Level 1: Group by education level
    dict_level1 = {
        "no_hs": [float(i) for i in range(0, 16)],
        "hs_graduate": [float(i) for i in range(16, 18)],
        "some_college": [float(i) for i in range(18, 21)],
        "bachelors": [21.0],
        "graduate": [float(i) for i in range(22, 25)]
    }
    
    # Level 2: Broad education categories
    dict_level2 = {
        "no_degree": [float(i) for i in range(0, 21)],
        "degree": [float(i) for i in range(21, 25)]
    }
    
    # Level 3: Complete suppression
    dict_level3 = {
        "?": [float(i) for i in range(0, 25)]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2,
        PrivacyLevel.LEVEL3: dict_level3
    }
    
    @staticmethod
    def get_value(key):
        for d in SCHL.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        for key, values in SCHL.dict_all[privacy_level].items():
            if value in values:
                return key
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in SCHL.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in SCHL.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = SCHL.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL3 or privacy_level == PrivacyLevel.LEVEL2:
            return value
        if privacy_level == PrivacyLevel.LEVEL1:
            return value
        # LEVEL0 - map to LEVEL1
        return SCHL.get_key(value, PrivacyLevel.LEVEL1)


# Template for 2-level categorical classes

# ==============================================================================
# CATEGORICAL SPALTE: MARITAL STATUS (MAR)
# ==============================================================================

class MAR:
    name = "MAR"
    
    dict_level1 = {'married': [1.0], 'not_married': [2.0, 3.0, 4.0, 5.0]}
    
    dict_level2 = {
        "?": [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    
    @staticmethod
    def get_value(key):
        for d in MAR.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        for key, values in MAR.dict_all[privacy_level].items():
            if value in values:
                return key
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in MAR.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in MAR.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = MAR.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        # LEVEL0 - map to LEVEL1
        return MAR.get_key(value, PrivacyLevel.LEVEL1)


# ==============================================================================
# CATEGORICAL SPALTE: RELATIONSHIP (RELP)
# ==============================================================================

class RELP:
    name = "RELP"
    
    dict_level1 = {'relative': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], 'non_relative': [0.0, 14.0, 15.0, 16.0, 17.0]}
    
    dict_level2 = {
        "?": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    
    @staticmethod
    def get_value(key):
        for d in RELP.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        for key, values in RELP.dict_all[privacy_level].items():
            if value in values:
                return key
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in RELP.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in RELP.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = RELP.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        # LEVEL0 - map to LEVEL1
        return RELP.get_key(value, PrivacyLevel.LEVEL1)


# ==============================================================================
# CATEGORICAL SPALTE: EMPLOYMENT STATUS OF PARENTS (ESP)
# ==============================================================================

class ESP:
    name = "ESP"
    
    dict_level1 = {'working': [1.0, 2.0, 3.0, 4.0, 5.0], 'not_working': [0.0, 6.0, 7.0, 8.0]}
    
    dict_level2 = {
        "?": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
        PrivacyLevel.LEVEL2: dict_level2
    }
    
    @staticmethod
    def get_value(key):
        for d in ESP.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        for key, values in ESP.dict_all[privacy_level].items():
            if value in values:
                return key
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in ESP.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in ESP.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = ESP.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        # LEVEL0 - map to LEVEL1
        return ESP.get_key(value, PrivacyLevel.LEVEL1)


# ==============================================================================
# CATEGORICAL SPALTE: CITIZENSHIP (CIT)
# ==============================================================================

class CIT:
    name = "CIT"

    dict_level1 = {
        "?": [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
    }
    
    @staticmethod
    def get_value(key):
        for d in CIT.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        for key, values in CIT.dict_all[privacy_level].items():
            if value in values:
                return key
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in CIT.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in CIT.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = CIT.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        # LEVEL0 - map to LEVEL1
        return CIT.get_key(value, PrivacyLevel.LEVEL1)


# ==============================================================================
# CATEGORICAL SPALTE: MOBILITY (MIG)
# ==============================================================================

class MIG:
    name = "MIG"

    dict_level1 = {
        "?": [0.0, 1.0, 2.0, 3.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
    }
    
    @staticmethod
    def get_value(key):
        for d in MIG.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        for key, values in MIG.dict_all[privacy_level].items():
            if value in values:
                return key
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in MIG.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in MIG.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = MIG.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        # LEVEL0 - map to LEVEL1
        return MIG.get_key(value, PrivacyLevel.LEVEL1)


# ==============================================================================
# CATEGORICAL SPALTE: MILITARY SERVICE (MIL)
# ==============================================================================

class MIL:
    name = "MIL"

    dict_level1 = {
        "?": [0.0, 1.0, 2.0, 3.0, 4.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1,
    }
    
    @staticmethod
    def get_value(key):
        for d in MIL.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        for key, values in MIL.dict_all[privacy_level].items():
            if value in values:
                return key
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in MIL.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in MIL.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = MIL.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL2 or privacy_level == PrivacyLevel.LEVEL1:
            return value
        # LEVEL0 - map to LEVEL1
        return MIL.get_key(value, PrivacyLevel.LEVEL1)


# ==============================================================================
# CATEGORICAL SPALTE: DISABILITY (DIS)
# ==============================================================================

class DIS:
    name = "DIS"
    
    dict_level1 = {
        "?": [1.0, 2.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    
    @staticmethod
    def get_value(key):
        for d in DIS.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in DIS.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in DIS.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = DIS.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value


# ==============================================================================
# CATEGORICAL SPALTE: ANCESTRY (ANC)
# ==============================================================================

class ANC:
    name = "ANC"
    
    dict_level1 = {
        "?": [1.0, 2.0, 3.0, 4.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    
    @staticmethod
    def get_value(key):
        for d in ANC.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in ANC.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in ANC.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = ANC.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value


# ==============================================================================
# CATEGORICAL SPALTE: NATIVITY (NATIVITY)
# ==============================================================================

class NATIVITY:
    name = "NATIVITY"
    
    dict_level1 = {
        "?": [1.0, 2.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    
    @staticmethod
    def get_value(key):
        for d in NATIVITY.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in NATIVITY.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in NATIVITY.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = NATIVITY.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value


# ==============================================================================
# CATEGORICAL SPALTE: HEARING DIFFICULTY (DEAR)
# ==============================================================================

class DEAR:
    name = "DEAR"
    
    dict_level1 = {
        "?": [1.0, 2.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    
    @staticmethod
    def get_value(key):
        for d in DEAR.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in DEAR.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in DEAR.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = DEAR.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value


# ==============================================================================
# CATEGORICAL SPALTE: VISION DIFFICULTY (DEYE)
# ==============================================================================

class DEYE:
    name = "DEYE"
    
    dict_level1 = {
        "?": [1.0, 2.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    
    @staticmethod
    def get_value(key):
        for d in DEYE.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in DEYE.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in DEYE.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = DEYE.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value


# ==============================================================================
# CATEGORICAL SPALTE: COGNITIVE DIFFICULTY (DREM)
# ==============================================================================

class DREM:
    name = "DREM"
    
    dict_level1 = {
        "?": [0.0, 1.0, 2.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    
    @staticmethod
    def get_value(key):
        for d in DREM.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in DREM.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in DREM.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = DREM.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value


# ==============================================================================
# CATEGORICAL SPALTE: SEX (SEX)
# ==============================================================================

class SEX:
    name = "SEX"
    
    dict_level1 = {
        "?": [1.0, 2.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    
    @staticmethod
    def get_value(key):
        for d in SEX.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in SEX.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in SEX.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = SEX.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value


# ==============================================================================
# CATEGORICAL SPALTE: RACE (RAC1P)
# ==============================================================================

class RAC1P:
    name = "RAC1P"
    
    dict_level1 = {
        "?": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    }
    
    dict_all = {
        PrivacyLevel.LEVEL1: dict_level1
    }
    
    @staticmethod
    def get_value(key):
        for d in RAC1P.dict_all.values():
            if key in d:
                return d[key]
        return None
    
    @staticmethod
    def get_key(value, privacy_level):
        return "?"
    
    @staticmethod
    def is_generalized(value):
        for d in RAC1P.dict_all.values():
            if value in d:
                return True
        return False
    
    @staticmethod
    def get_privacy_level_for_value(value):
        for privacy_level, values in RAC1P.dict_all.items():
            if value in values.keys():
                return privacy_level
        return PrivacyLevel.LEVEL0
    
    @staticmethod
    def get_highest_privacy_value(value):
        privacy_level = RAC1P.get_privacy_level_for_value(value)
        if privacy_level == PrivacyLevel.LEVEL1 or privacy_level == PrivacyLevel.LEVEL0:
            return value

