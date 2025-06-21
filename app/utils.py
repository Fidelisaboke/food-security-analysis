# Mapping for nominal inputs
def get_ordinal_mappings():
    return {
        "hhedu": {
            "Never attended": 0,
            "Primary": 1,
            "Secondary": 2,
            "Higher": 3
        },

        "windex3": {
            "Lowest": 0.0,
            "Middle": 1.0,
            "Highest": 2.0
        },

        "windex5": {
            "Lowest": 0.0,
            "Second": 1.0,
            "Middle": 2.0,
            "Fourth": 3.0,
            "Highest": 4.0
        }
    }

def get_ethnicity_booleans(ethnicity):
    mapping = {
        "Kikuyu": [1, 0, 0, 0, 0],
        "Kisii": [0, 1, 0, 0, 0],
        "Luhya": [0, 0, 1, 0, 0],
        "Luo": [0, 0, 0, 1, 0],
        "Other": [0, 0, 0, 0, 1],
    }
    
    return mapping[ethnicity]
