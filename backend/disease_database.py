# KrushiVaani - Crop & Disease Database
# Safe, farmer-friendly information for AI results.

DISEASE_DATABASE = {
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "ಟೊಮ್ಯಾಟೊ (Tomato)",
        "disease": "Yellow Leaf Curl Virus",
        "summary": "ಟೊಮ್ಯಾಟೊ ಎಲೆಗಳಲ್ಲಿ ಹಳದಿ ಬಣ್ಣ ಮತ್ತು ಮಡಚಿಕೊಳ್ಳುವ ಲಕ್ಷಣಗಳು ಕಾಣಿಸಬಹುದು.",
        "do": [
            "ಪೀಡಿತ ಗಿಡಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
            "ಸ್ಪಷ್ಟವಾದ ಎಲೆ ಮತ್ತು ಗಿಡದ ಚಿತ್ರವನ್ನು ತೆಗೆದು ಮತ್ತೊಮ್ಮೆ ಪರಿಶೀಲಿಸಿ.",
            "ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ.",
            "ಹೊಲದಲ್ಲಿನ ರೋಗದ ಹರಡುವಿಕೆಯನ್ನು ಗಮನಿಸಿ."
        ],
        "dont": [
            "AI result ಮಾತ್ರ ಆಧರಿಸಿ pesticide ಅಥವಾ ಔಷಧಿ ಬಳಸಬೇಡಿ.",
            "Confidence ಕಡಿಮೆ ಇದ್ದಾಗ result ಅನ್ನು ಖಚಿತ diagnosis ಎಂದು ಪರಿಗಣಿಸಬೇಡಿ.",
            "ರೋಗ ಖಚಿತವಾಗದೆ ದುಬಾರಿ treatment ಆರಂಭಿಸಬೇಡಿ."
        ]
    },

    "Tomato___Early_blight": {
        "crop": "ಟೊಮ್ಯಾಟೊ (Tomato)",
        "disease": "Early Blight",
        "summary": "ಟೊಮ್ಯಾಟೊ ಎಲೆಗಳಲ್ಲಿ ಕಂದು ಬಣ್ಣದ ವೃತ್ತಾಕಾರದ ಕಲೆಗಳು ಕಾಣಿಸಿಕೊಳ್ಳುವುದು Early Blight ನ ಸಾಮಾನ್ಯ ಲಕ್ಷಣವಾಗಿದೆ.",
        "do": [
            "ಪೀಡಿತ ಎಲೆಗಳು ಮತ್ತು ಗಿಡಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
            "ಗಿಡಗಳ ನಡುವೆ ಸಾಕಷ್ಟು ಗಾಳಿ ಸಂಚಾರ ಇರುವಂತೆ ನೋಡಿಕೊಳ್ಳಿ.",
            "ಎಲೆಗಳಿಗೆ ನೀರು ನೇರವಾಗಿ ಬೀಳದಂತೆ ಎಚ್ಚರಿಕೆ ವಹಿಸಿ.",
            "ರೋಗದ ಪ್ರಮಾಣ ಹೆಚ್ಚಾದರೆ ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
        ],
        "dont": [
            "AI result ಮಾತ್ರ ಆಧರಿಸಿ pesticide ಅಥವಾ ಔಷಧಿ ಬಳಸಬೇಡಿ.",
            "Confidence ಕಡಿಮೆ ಇದ್ದಾಗ result ಅನ್ನು ಖಚಿತ diagnosis ಎಂದು ಪರಿಗಣಿಸಬೇಡಿ.",
            "ರೋಗ ಖಚಿತವಾಗದೆ ದುಬಾರಿ treatment ಆರಂಭಿಸಬೇಡಿ."
        ]
    },

    "Tomato___healthy": {
        "crop": "ಟೊಮ್ಯಾಟೊ (Tomato)",
        "disease": "Healthy / ಆರೋಗ್ಯಕರ",
        "summary": "ಚಿತ್ರದ ಆಧಾರದ ಮೇಲೆ ಆರೋಗ್ಯಕರವಾಗಿರುವ ಸಾಧ್ಯತೆ ಇದೆ.",
        "do": [
            "ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
            "ಎಲೆಗಳ ಬಣ್ಣ ಮತ್ತು ಗಿಡದ ಬೆಳವಣಿಗೆಯನ್ನು ಗಮನಿಸಿ.",
            "ಅಸಾಮಾನ್ಯ ಲಕ್ಷಣ ಕಂಡುಬಂದರೆ ಹೊಸ ಚಿತ್ರದಿಂದ ಮತ್ತೆ ಪರಿಶೀಲಿಸಿ."
        ],
        "dont": [
            "AI result ಮಾತ್ರ ನೋಡಿ ಬೆಳೆಗೆ ಅಗತ್ಯವಿಲ್ಲದ ಔಷಧಿ ಬಳಸಬೇಡಿ."
        ]
    },

    "Pepper,_bell___healthy": {
        "crop": "Capsicum / Bell Pepper",
        "disease": "Healthy / ಆರೋಗ್ಯಕರ",
        "summary": "ಚಿತ್ರದ ಆಧಾರದ ಮೇಲೆ ಆರೋಗ್ಯಕರವಾಗಿರುವ ಸಾಧ್ಯತೆ ಇದೆ.",
        "do": [
            "ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
            "ಎಲೆ ಮತ್ತು ಹಣ್ಣುಗಳಲ್ಲಿ ಬದಲಾವಣೆಗಳನ್ನು ಗಮನಿಸಿ.",
            "ಅಸಾಮಾನ್ಯ ಲಕ್ಷಣ ಕಂಡರೆ ಸ್ಪಷ್ಟವಾದ ಚಿತ್ರದಿಂದ ಮತ್ತೆ ಪರಿಶೀಲಿಸಿ."
        ],
        "dont": [
            "ಅಗತ್ಯವಿಲ್ಲದೆ pesticide ಅಥವಾ ಔಷಧಿ ಬಳಸಬೇಡಿ."
        ]
    },

    "Soybean___healthy": {
        "crop": "ಸೋಯಾಬೀನ್ (Soybean)",
        "disease": "Healthy / ಆರೋಗ್ಯಕರ",
        "summary": "ಚಿತ್ರದ ಆಧಾರದ ಮೇಲೆ ಆರೋಗ್ಯಕರವಾಗಿರುವ ಸಾಧ್ಯತೆ ಇದೆ.",
        "do": [
            "ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
            "ಎಲೆಗಳ ಬಣ್ಣ ಮತ್ತು ಕಲೆಗಳನ್ನು ಗಮನಿಸಿ.",
            "ಸಮಸ್ಯೆ ಕಂಡರೆ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
        ],
        "dont": [
            "AI result ಮಾತ್ರ ಆಧರಿಸಿ ಔಷಧಿ ಬಳಸಬೇಡಿ."
        ]
    }
}


def get_disease_info(label: str):
    """Return database information for an AI model label."""

    if not label:
        return None

    # Exact match first.
    if label in DISEASE_DATABASE:
        return DISEASE_DATABASE[label]

    # Fallback: match by normalized text.
    normalized = label.lower().replace(" ", "_")

    for key, value in DISEASE_DATABASE.items():
        if key.lower() == normalized:
            return value

    return None