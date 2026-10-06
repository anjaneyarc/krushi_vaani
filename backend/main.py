from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from disease_database import get_disease_info, DISEASE_DATABASE
from medicine_database import get_all_medicines, get_medicine_by_id


app = FastAPI()


# =========================================================
# AI IMAGE CLASSIFIER
# =========================================================
image_classifier = None

def get_image_classifier():

    global image_classifier

    if image_classifier is None:

        print("Loading AI image classifier...")

        from transformers import pipeline

        image_classifier = pipeline(
            "image-classification",
            model="kimcomehome/plantvillage-vit-leaf-disease"
        )

        print("AI image classifier loaded successfully.")

    return image_classifier
# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return {
        "message": "KrushiVaani Backend is Running!",
        "status": "success"
    }


# =========================================================
# FARMER PROBLEM
# =========================================================

class ProblemRequest(BaseModel):
    problem: str


def classify_problem(problem: str):

    text = problem.lower()

    pest_words = [
        "ಕೀಟ",
        "ಹುಳು",
        "ಹುಳ",
        "ಕೀಟ ಬಂದಿದೆ",
        "ಕೀಟ ಬಂತು",
        "insect",
        "pest"
    ]

    water_words = [
        "ನೀರು",
        "ನೀರಿನ ಕೊರತೆ",
        "ನೀರು ಕಡಿಮೆ",
        "ನೀರು ಸಾಲುತ್ತಿಲ್ಲ",
        "ನೀರಿಲ್ಲ",
        "water"
    ]

    leaf_words = [
        "ಎಲೆ",
        "ಎಲೆಗಳು",
        "ಎಲೆ ಹಳದಿ",
        "ಎಲೆ ಹಳದಿಯಾಗಿದೆ",
        "ಎಲೆಯಲ್ಲಿ ಕಲೆ",
        "ಎಲೆ ಒಣಗುತ್ತಿದೆ",
        "leaf"
    ]

    disease_words = [
        "ರೋಗ",
        "ರೋಗ ಬಂದಿದೆ",
        "ಬೆಳೆ ರೋಗ",
        "ಸೊಂಕು",
        "infection",
        "disease"
    ]

    fertilizer_words = [
        "ಗೊಬ್ಬರ",
        "ರಸಗೊಬ್ಬರ",
        "ಗೊಬ್ಬರ ಯಾವುದು",
        "fertilizer"
    ]

    if any(word in text for word in pest_words):
        return "pest"

    elif any(word in text for word in water_words):
        return "water"

    elif any(word in text for word in leaf_words):
        return "leaf"

    elif any(word in text for word in disease_words):
        return "disease"

    elif any(word in text for word in fertilizer_words):
        return "fertilizer"

    return "unknown"


# =========================================================
# FARMER RESPONSE
# =========================================================

def generate_farmer_response(category):

    if category == "pest":

        return {
            "answer": (
                "ನಿಮ್ಮ ಬೆಳೆಗೆ ಕೀಟ ಅಥವಾ ಹುಳು ಸಮಸ್ಯೆ ಇರುವಂತೆ ಕಾಣುತ್ತಿದೆ. "
                "ಮೊದಲು ಯಾವ ಕೀಟ ಎಂದು ಗುರುತಿಸುವುದು ಮುಖ್ಯ."
            ),

            "main_advice": [
                "ಬೆಳೆಯನ್ನು ಹತ್ತಿರದಿಂದ ಪರಿಶೀಲಿಸಿ.",
                "ಕೀಟವು ಎಲೆ, ಕಾಂಡ ಅಥವಾ ಹಣ್ಣಿನಲ್ಲಿ ಇದೆಯೇ ಎಂದು ಗಮನಿಸಿ.",
                "ಕೀಟದ ಪ್ರಮಾಣ ಹೆಚ್ಚಾಗುತ್ತಿದೆಯೇ ಎಂದು ಗಮನಿಸಿ.",
                "ಸಾಧ್ಯವಾದರೆ ಪೀಡಿತ ಭಾಗದ ಸ್ಪಷ್ಟವಾದ photo upload ಮಾಡಿ."
            ],

            "do": [
                "ಪೀಡಿತ ಎಲೆ ಅಥವಾ ಹಣ್ಣಿನ photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
            ],

            "dont": [
                "ಕೀಟವನ್ನು ಗುರುತಿಸದೆ pesticide ಬಳಸಬೇಡಿ.",
                "AI result ಮಾತ್ರ ನೋಡಿ ತಕ್ಷಣ ಔಷಧಿ ಬಳಸಬೇಡಿ.",
                "ಅಗತ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚು pesticide ಬಳಸಬೇಡಿ.",
                "ಕಡಿಮೆ confidence ಇರುವ AI result ಅನ್ನು ಖಚಿತ diagnosis ಎಂದು ಪರಿಗಣಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ನಿಮ್ಮ ಬೆಳೆಯ ಪೀಡಿತ ಭಾಗದ "
                "ಸ್ಪಷ್ಟವಾದ photo upload ಮಾಡಿ. "
                "KrushiVaani ಆ ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸುತ್ತದೆ."
            )
        }

    elif category == "water":

        return {
            "answer": (
                "ನಿಮ್ಮ ಬೆಳೆಗೆ ನೀರಿನ ಕೊರತೆ ಇರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "ಮಣ್ಣಿನ ತೇವಾಂಶವನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "ಬೆಳೆಯ ಅಗತ್ಯಕ್ಕೆ ಅನುಗುಣವಾಗಿ ನೀರು ನೀಡಿ.",
                "ಅತಿಯಾಗಿ ನೀರು ಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ."
            ],

            "do": [
                "ಮಣ್ಣಿನ ತೇವಾಂಶ ಪರಿಶೀಲಿಸಿ.",
                "ಬೆಳೆಯ ಹಂತಕ್ಕೆ ಅನುಗುಣವಾಗಿ ನೀರು ನೀಡಿ.",
                "ನೀರಿನ ಲಭ್ಯತೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ."
            ],

            "dont": [
                "ಅಗತ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚು ನೀರು ಹಾಕಬೇಡಿ.",
                "ನೀರು ನಿಲ್ಲುವಂತೆ ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ನೀರಿನ ಸಮಸ್ಯೆಯಿರುವ ಭಾಗದ photo upload ಮಾಡಿದರೆ "
                "KrushiVaani ಇನ್ನಷ್ಟು ಮಾಹಿತಿ ನೀಡಲು ಪ್ರಯತ್ನಿಸುತ್ತದೆ."
            )
        }

    elif category == "leaf":

        return {
            "answer": (
                "ನಿಮ್ಮ ಬೆಳೆಯ ಎಲೆಗಳಲ್ಲಿ ಸಮಸ್ಯೆ ಇರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "ಎಲೆಗಳ ಬಣ್ಣವನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "ಎಲೆಗಳಲ್ಲಿ ಕಲೆಗಳಿವೆಯೇ ಎಂದು ಗಮನಿಸಿ.",
                "ಎಲೆ ಒಣಗುತ್ತಿದೆಯೇ ಅಥವಾ ಮಡಚಿಕೊಳ್ಳುತ್ತಿದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ."
            ],

            "do": [
                "ಪೀಡಿತ ಎಲೆಯ ಸ್ಪಷ್ಟವಾದ photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ಸಮಸ್ಯೆ ಹೆಚ್ಚಾಗುತ್ತಿದೆಯೇ ಎಂದು ಗಮನಿಸಿ."
            ],

            "dont": [
                "ಕಾರಣ ತಿಳಿಯದೆ pesticide ಬಳಸಬೇಡಿ.",
                "AI result ಮಾತ್ರ ಆಧರಿಸಿ treatment ಆರಂಭಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಪೀಡಿತ ಎಲೆಯ photo upload ಮಾಡಿ."
            )
        }

    elif category == "disease":

        return {
            "answer": (
                "ನಿಮ್ಮ ಬೆಳೆಯಲ್ಲಿ ರೋಗದ ಲಕ್ಷಣಗಳು ಕಂಡುಬರುತ್ತಿರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "ರೋಗದ ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.",
                "ಪೀಡಿತ ಎಲೆ ಮತ್ತು ಗಿಡವನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "ರೋಗದ ಹರಡುವಿಕೆಯನ್ನು ಗಮನಿಸಿ."
            ],

            "do": [
                "ಪೀಡಿತ ಭಾಗದ photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
            ],

            "dont": [
                "ರೋಗ ಖಚಿತವಾಗದೆ ಔಷಧಿ ಬಳಸಬೇಡಿ.",
                "AI result ಮಾತ್ರ ನೋಡಿ treatment ಆರಂಭಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಪೀಡಿತ ಬೆಳೆಯ photo upload ಮಾಡಿ."
            )
        }

    elif category == "fertilizer":

        return {
            "answer": (
                "ಗೊಬ್ಬರ ಬಳಸುವ ಮೊದಲು ನಿಮ್ಮ ಬೆಳೆಯ ಪ್ರಕಾರ "
                "ಮತ್ತು ಮಣ್ಣಿನ ಸ್ಥಿತಿಯನ್ನು ಪರಿಗಣಿಸಬೇಕು."
            ),

            "main_advice": [
                "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮಾಡಿಸುವುದು ಉತ್ತಮ.",
                "ಬೆಳೆಯ ಪ್ರಕಾರಕ್ಕೆ ಸೂಕ್ತವಾದ ಗೊಬ್ಬರವನ್ನು ಆಯ್ಕೆ ಮಾಡಿ.",
                "ಕೃಷಿ ಅಧಿಕಾರಿಯ ಸಲಹೆ ಪಡೆಯಿರಿ."
            ],

            "do": [
                "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ.",
                "ಬೆಳೆಯ ಪ್ರಕಾರ ತಿಳಿದುಕೊಳ್ಳಿ.",
                "ತಜ್ಞರ ಸಲಹೆಯಂತೆ ಗೊಬ್ಬರ ಬಳಸಿ."
            ],

            "dont": [
                "ಅಗತ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚು ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ.",
                "ಯಾವುದೇ ಗೊಬ್ಬರವನ್ನು ಊಹೆಯಿಂದ ಬಳಸಬೇಡಿ."
            ],

            "next_step": (
                "ನಿಮ್ಮ ಬೆಳೆಯ photo upload ಮಾಡಿದರೆ "
                "ಹೆಚ್ಚಿನ ಮಾಹಿತಿ ಪಡೆಯಬಹುದು."
            )
        }

    else:

        return {
            "answer": (
                "ನಿಮ್ಮ ಸಮಸ್ಯೆಯನ್ನು ಸಂಪೂರ್ಣವಾಗಿ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ."
            ),

            "main_advice": [
                "ದಯವಿಟ್ಟು ಬೆಳೆಯ ಹೆಸರು ತಿಳಿಸಿ.",
                "ಸಮಸ್ಯೆಯ ಲಕ್ಷಣಗಳನ್ನು ವಿವರಿಸಿ."
            ],

            "do": [
                "ಬೆಳೆಯ ಹೆಸರು ತಿಳಿಸಿ.",
                "ಸಮಸ್ಯೆಯ photo upload ಮಾಡಿ."
            ],

            "dont": [
                "ಸಮಸ್ಯೆ ತಿಳಿಯದೆ ಔಷಧಿ ಬಳಸಬೇಡಿ."
            ],

            "next_step": (
                "ಬೆಳೆಯ photo upload ಮಾಡಿ ಅಥವಾ ಸಮಸ್ಯೆಯನ್ನು "
                "ಇನ್ನಷ್ಟು ವಿವರವಾಗಿ ಹೇಳಿ."
            )
        }


# =========================================================
# PROBLEM API
# =========================================================

@app.post("/problem")
def receive_problem(data: ProblemRequest):

    problem = data.problem

    category = classify_problem(problem)

    response = generate_farmer_response(category)

    return {
        "message": "KrushiVaani Response",
        "problem": problem,
        "category": category,
        "answer": response["answer"],
        "main_advice": response["main_advice"],
        "do": response["do"],
        "dont": response["dont"],
        "next_step": response["next_step"]
    }


# =========================================================
# NORMALIZE AI LABEL
# =========================================================

def normalize_label(label: str) -> str:

    if not label:
        return ""

    return (
        label
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# =========================================================
# FIND DISEASE INFORMATION
# =========================================================

def find_disease_information(label: str):

    if not label:
        return None

    # -----------------------------------------------------
    # 1. Exact database match
    # -----------------------------------------------------

    info = get_disease_info(label)

    if info is not None:
        return info

    # -----------------------------------------------------
    # 2. Normalized match
    # -----------------------------------------------------

    normalized_label = normalize_label(label)

    for key, value in DISEASE_DATABASE.items():

        normalized_key = normalize_label(key)

        if normalized_label == normalized_key:
            return value

    # -----------------------------------------------------
    # 3. Partial match
    # -----------------------------------------------------

    for key, value in DISEASE_DATABASE.items():

        normalized_key = normalize_label(key)

        if (
            normalized_label in normalized_key
            or normalized_key in normalized_label
        ):
            return value

    # -----------------------------------------------------
    # 4. Disease-name based matching
    # -----------------------------------------------------

    label_lower = label.lower()

    disease_keywords = {

        "early_blight": [
            "early_blight",
            "early blight"
        ],

        "yellow_leaf_curl": [
            "yellow_leaf_curl",
            "yellow leaf curl",
            "yellow_leaf_curl_virus"
        ],

        "healthy": [
            "healthy"
        ],

        "late_blight": [
            "late_blight",
            "late blight"
        ],

        "bacterial_spot": [
            "bacterial_spot",
            "bacterial spot"
        ],

        "leaf_mold": [
            "leaf_mold",
            "leaf mold"
        ],

        "septoria": [
            "septoria"
        ]
    }

    for disease_key, keywords in disease_keywords.items():

        for keyword in keywords:

            if keyword in label_lower:

                for db_key, value in DISEASE_DATABASE.items():

                    if disease_key in normalize_label(db_key):

                        return value

    return None


# =========================================================
# GENERIC DISEASE INFORMATION
# =========================================================

def create_generic_disease_info(label: str):

    label_text = label.replace("_", " ")

    return {
        "crop": "AI ಮೂಲಕ ಗುರುತಿಸಲಾದ ಬೆಳೆ",
        "disease": label_text,
        "summary": (
            "AI ಚಿತ್ರ ವಿಶ್ಲೇಷಣೆಯ ಆಧಾರದ ಮೇಲೆ ಈ ಸಮಸ್ಯೆ "
            "ಗುರುತಿಸಲಾಗಿದೆ. ಆದರೆ AI result ಅನ್ನು ಖಚಿತ "
            "ರೋಗನಿರ್ಣಯ ಎಂದು ಪರಿಗಣಿಸಬಾರದು."
        ),

        "do": [
            "ಪೀಡಿತ ಬೆಳೆಯ ಸ್ಪಷ್ಟವಾದ photo ತೆಗೆದು ಮತ್ತೆ ಪರಿಶೀಲಿಸಿ.",
            "ಬೆಳೆಯ ಲಕ್ಷಣಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಗಮನಿಸಿ.",
            "ಸಮಸ್ಯೆ ಹೆಚ್ಚಾದರೆ ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
        ],

        "dont": [
            "AI result ಮಾತ್ರ ಆಧರಿಸಿ pesticide ಅಥವಾ ಔಷಧಿ ಬಳಸಬೇಡಿ.",
            "Confidence ಕಡಿಮೆ ಇದ್ದಾಗ result ಅನ್ನು ಖಚಿತ diagnosis ಎಂದು ಪರಿಗಣಿಸಬೇಡಿ.",
            "ರೋಗ ಖಚಿತವಾಗದೆ ದುಬಾರಿ treatment ಆರಂಭಿಸಬೇಡಿ."
        ]
    }


# =========================================================
# IMAGE ANALYSIS
# =========================================================

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...)
):

    from PIL import Image
    import io

    try:

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:

            return {
                "message": "Image file is empty.",
                "filename": file.filename,
                "crop_or_disease": "Unknown",
                "confidence": 0.0,
                "disease_info": create_generic_disease_info(
                    "Unknown"
                ),
                "all_predictions": []
            }

        # -------------------------------------------------
        # OPEN IMAGE
        # -------------------------------------------------

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # -------------------------------------------------
        # AI PREDICTION
        # -------------------------------------------------

        classifier = get_image_classifier()

        predictions = classifier(image)
        if not predictions:

            return {
                "message": "AI could not analyze the image.",
                "filename": file.filename,
                "crop_or_disease": "Unknown",
                "confidence": 0.0,
                "disease_info": create_generic_disease_info(
                    "Unknown"
                ),
                "all_predictions": []
            }

        # -------------------------------------------------
        # BEST RESULT
        # -------------------------------------------------

        best_prediction = predictions[0]

        label = best_prediction.get(
            "label",
            "Unknown"
        )

        confidence = round(
            float(
                best_prediction.get(
                    "score",
                    0.0
                )
            ) * 100,
            2
        )

        # -------------------------------------------------
        # DISEASE DATABASE
        # -------------------------------------------------

        disease_info = find_disease_information(
            label
        )

        # -------------------------------------------------
        # FALLBACK INFORMATION
        # -------------------------------------------------

        if disease_info is None:

            disease_info = create_generic_disease_info(
                label
            )

        # -------------------------------------------------
        # FINAL RESPONSE
        # -------------------------------------------------

        return {
            "message": "Image analyzed successfully!",
            "filename": file.filename,

            "crop_or_disease": label,

            "confidence": confidence,

            "disease_info": disease_info,

            "all_predictions": predictions[:5]
        }

    except Exception as e:

        print(
            f"Image Analysis Error: {e}"
        )

        return {
            "message": "Image analysis failed.",
            "filename": file.filename,

            "crop_or_disease": "Unknown",

            "confidence": 0.0,

            "disease_info": create_generic_disease_info(
                "Unknown"
            ),

            "all_predictions": []
        }
   # =========================================================
# GOVERNMENT SCHEMES
# =========================================================

GOVERNMENT_SCHEMES = [
    {
        "id": 1,
        "title": "PM-KISAN",
        "title_kn": "ಪ್ರಧಾನಮಂತ್ರಿ ಕಿಸಾನ್ ಸಮ್ಮಾನ್ ನಿಧಿ",
        "description": "ಅರ್ಹ ರೈತ ಕುಟುಂಬಗಳಿಗೆ ಕೇಂದ್ರ ಸರ್ಕಾರದ ಆದಾಯ ಸಹಾಯ ಯೋಜನೆ.",
        "benefit": "ಅರ್ಹ ರೈತರಿಗೆ ವರ್ಷಕ್ಕೆ ₹6,000 ನೇರವಾಗಿ ಬ್ಯಾಂಕ್ ಖಾತೆಗೆ.",
        "eligibility": "ಅರ್ಹ ಭೂಮಾಲೀಕ ರೈತ ಕುಟುಂಬಗಳು.",
        "category": "Financial Support",
        "official_url": "https://pmkisan.gov.in/"
    },
    {
        "id": 2,
        "title": "Pradhan Mantri Fasal Bima Yojana",
        "title_kn": "ಪ್ರಧಾನಮಂತ್ರಿ ಫಸಲ್ ಬಿಮಾ ಯೋಜನೆ",
        "description": "ಬೆಳೆ ಹಾನಿಯಿಂದ ರೈತರಿಗೆ ವಿಮಾ ರಕ್ಷಣೆ ನೀಡುವ ಯೋಜನೆ.",
        "benefit": "ಅರ್ಹ ಬೆಳೆ ನಷ್ಟಗಳಿಗೆ ವಿಮಾ ಪರಿಹಾರ ಪಡೆಯಲು ಅವಕಾಶ.",
        "eligibility": "ಯೋಜನೆಯ ನಿಯಮಗಳಿಗೆ ಒಳಪಡುವ ರೈತರು.",
        "category": "Crop Insurance",
        "official_url": "https://pmfby.gov.in/"
    },
    {
        "id": 3,
        "title": "Kisan Credit Card",
        "title_kn": "ಕಿಸಾನ್ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್",
        "description": "ಕೃಷಿ ಚಟುವಟಿಕೆಗಳಿಗೆ ಸಾಲ ಸೌಲಭ್ಯ ಪಡೆಯಲು ಸಹಾಯ ಮಾಡುವ ಯೋಜನೆ.",
        "benefit": "ಕೃಷಿ ಅಗತ್ಯಗಳಿಗೆ ಸಾಲ ಸೌಲಭ್ಯ.",
        "eligibility": "ಅರ್ಹ ರೈತರು ಮತ್ತು ಕೃಷಿ ಚಟುವಟಿಕೆಯಲ್ಲಿ ತೊಡಗಿರುವವರು.",
        "category": "Agricultural Credit",
        "official_url": "https://www.myscheme.gov.in/schemes/kcc"
    }
]


@app.get("/schemes")
def get_government_schemes():

    return {
        "message": "Government schemes loaded successfully",
        "count": len(GOVERNMENT_SCHEMES),
        "schemes": GOVERNMENT_SCHEMES
    }

# =========================================================
# AGRICULTURE MEDICINES
# =========================================================

@app.get("/medicines")
def get_medicines():

    return {
        "message": "Agriculture medicines loaded successfully",
        "count": len(get_all_medicines()),
        "medicines": get_all_medicines()
    }


@app.get("/medicines/{medicine_id}")
def get_medicine(medicine_id: int):

    medicine = get_medicine_by_id(medicine_id)

    if medicine is None:

        return {
            "message": "Medicine information not found.",
            "medicine": None
        }

    return {
        "message": "Medicine information loaded successfully",
        "medicine": medicine
    }

# =========================================================
# FARMER GOVERNMENT NOTIFICATIONS
# =========================================================

GOVERNMENT_NOTIFICATIONS = [
    {
        "id": 1,
        "title": "ಹೊಸ ರೈತ ಯೋಜನೆ ಮಾಹಿತಿ",
        "message": "ರೈತರಿಗೆ ಲಭ್ಯವಿರುವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ ಮಾಹಿತಿಗಾಗಿ KrushiVaani ಪರಿಶೀಲಿಸಿ.",
        "category": "Government Scheme",
        "date": "2026-10-04",
        "is_new": True
    },
    {
        "id": 2,
        "title": "ಬೆಳೆ ವಿಮೆ ಮಾಹಿತಿ",
        "message": "ಬೆಳೆ ಹಾನಿಯಿಂದ ರಕ್ಷಣೆ ಪಡೆಯಲು ಪ್ರಧಾನಮಂತ್ರಿ ಫಸಲ್ ಬಿಮಾ ಯೋಜನೆಯ ವಿವರಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
        "category": "Crop Insurance",
        "date": "2026-10-04",
        "is_new": True
    }
]


@app.get("/notifications")
def get_notifications():
    return {
        "message": "Farmer notifications loaded successfully",
        "count": len(GOVERNMENT_NOTIFICATIONS),
        "notifications": GOVERNMENT_NOTIFICATIONS
    }