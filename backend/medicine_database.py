# =========================================================
# KRUSHIVAANI - AGRICULTURE MEDICINE DATABASE
# =========================================================

MEDICINE_DATABASE = [

    {
        "id": 1,
        "crop": "ಟೊಮ್ಯಾಟೊ",
        "problem": "Early Blight",
        "problem_kn": "ಎಲೆಗಳಲ್ಲಿ ಕಪ್ಪು/ಕಂದು ಕಲೆಗಳು",
        "medicine_type": "Fungicide",
        "general_info": (
            "Early Blight ರೀತಿಯ ಶಿಲೀಂಧ್ರ ಸಮಸ್ಯೆಗಳಲ್ಲಿ "
            "ನೋಂದಾಯಿತ fungicide ಅನ್ನು label ಸೂಚನೆಯಂತೆ ಬಳಸಬೇಕು."
        ),
        "usage": (
            "ಉತ್ಪನ್ನದ label ನಲ್ಲಿ ನೀಡಿರುವ crop, disease ಮತ್ತು "
            "dose ಸೂಚನೆಗಳನ್ನು ಮಾತ್ರ ಅನುಸರಿಸಿ."
        ),
        "precaution": (
            "ಔಷಧಿಯನ್ನು label ಸೂಚನೆ ಇಲ್ಲದೆ ಮಿಶ್ರಣ ಮಾಡಬೇಡಿ. "
            "ರಕ್ಷಣಾ ಕೈಗವಸು ಮತ್ತು ಅಗತ್ಯ safety equipment ಬಳಸಿ."
        )
    },

    {
        "id": 2,
        "crop": "ಟೊಮ್ಯಾಟೊ",
        "problem": "Late Blight",
        "problem_kn": "ಎಲೆ ಮತ್ತು ಹಣ್ಣುಗಳಲ್ಲಿ ಕಂದು ಕಲೆಗಳು",
        "medicine_type": "Fungicide",
        "general_info": (
            "Late Blight ಶಿಲೀಂಧ್ರ ಸಮಸ್ಯೆಯಾಗಿರಬಹುದು. "
            "ಸರಿಯಾದ diagnosis ನಂತರ ಸೂಕ್ತ registered fungicide ಆಯ್ಕೆ ಮಾಡಬೇಕು."
        ),
        "usage": (
            "ಉತ್ಪನ್ನದ label ಮತ್ತು ಕೃಷಿ ಇಲಾಖೆಯ ಸಲಹೆಯಂತೆ ಬಳಸಿ."
        ),
        "precaution": (
            "ರೋಗ ಖಚಿತವಾಗದೆ ಔಷಧಿ ಬಳಸಬೇಡಿ."
        )
    },

    {
        "id": 3,
        "crop": "ಮೆಣಸಿನಕಾಯಿ",
        "problem": "Aphids",
        "problem_kn": "ಎಲೆಗಳಲ್ಲಿ ಸಣ್ಣ ಕೀಟಗಳು",
        "medicine_type": "Insecticide",
        "general_info": (
            "Aphids ಕಾಣಿಸಿಕೊಂಡರೆ ಮೊದಲು infestation ಪ್ರಮಾಣವನ್ನು ಪರಿಶೀಲಿಸಿ."
        ),
        "usage": (
            "ಅಗತ್ಯವಿದ್ದರೆ cropಗೆ ಅನುಮೋದಿತ insecticide ಅನ್ನು "
            "label ಸೂಚನೆಯಂತೆ ಮಾತ್ರ ಬಳಸಿ."
        ),
        "precaution": (
            "ಹೂ ಬಿಡುವ ಸಮಯದಲ್ಲಿ pollinators ಗೆ ಹಾನಿಯಾಗದಂತೆ "
            "ಅಗತ್ಯ ಮುನ್ನೆಚ್ಚರಿಕೆ ತೆಗೆದುಕೊಳ್ಳಿ."
        )
    },

    {
        "id": 4,
        "crop": "ಅಕ್ಕಿ",
        "problem": "Stem Borer",
        "problem_kn": "ಕಾಂಡ ಕೊರೆಯುವ ಹುಳು",
        "medicine_type": "Insecticide",
        "general_info": (
            "Stem Borer ಸಮಸ್ಯೆಯಲ್ಲಿ ಗಿಡದ ಕಾಂಡ ಮತ್ತು ಎಲೆಗಳ ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ."
        ),
        "usage": (
            "ಅನುಮೋದಿತ ಉತ್ಪನ್ನವನ್ನು label ಸೂಚನೆಯಂತೆ ಮಾತ್ರ ಬಳಸಿ."
        ),
        "precaution": (
            "ಹೆಚ್ಚುವರಿ ಪ್ರಮಾಣದಲ್ಲಿ pesticide ಬಳಸಬೇಡಿ."
        )
    },

    {
        "id": 5,
        "crop": "ಹತ್ತಿ",
        "problem": "Aphids / Sucking Pests",
        "problem_kn": "ರಸ ಹೀರುವ ಕೀಟಗಳು",
        "medicine_type": "Insecticide",
        "general_info": (
            "ರಸ ಹೀರುವ ಕೀಟಗಳ ಪ್ರಮಾಣವನ್ನು ಮೊದಲು ಪರಿಶೀಲಿಸುವುದು ಮುಖ್ಯ."
        ),
        "usage": (
            "ಬೆಳೆ ಮತ್ತು ಕೀಟಕ್ಕೆ ನೋಂದಾಯಿತ ಉತ್ಪನ್ನವನ್ನು "
            "label ಸೂಚನೆಯಂತೆ ಬಳಸಿ."
        ),
        "precaution": (
            "ಒಂದೇ pesticide ಅನ್ನು ನಿರಂತರವಾಗಿ ಬಳಸುವುದನ್ನು ತಪ್ಪಿಸಿ "
            "ಮತ್ತು label ಸೂಚನೆಗಳನ್ನು ಪಾಲಿಸಿ."
        )
    }
]


def get_all_medicines():
    return MEDICINE_DATABASE


def get_medicine_by_id(medicine_id: int):

    for medicine in MEDICINE_DATABASE:

        if medicine["id"] == medicine_id:
            return medicine

    return None