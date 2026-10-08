import os
import io
import json
import shutil

import numpy as np
import onnxruntime as ort

from PIL import Image
from huggingface_hub import snapshot_download

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from disease_database import (
    get_disease_info,
    DISEASE_DATABASE
)

from medicine_database import (
    get_all_medicines,
    get_medicine_by_id
)


# =========================================================
# APP
# =========================================================

app = FastAPI()


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
# CROP DISEASE AI - ONNX
# =========================================================

AI_MODEL_REPO = "BiernyVR/crop-disease-classifier"

ai_session = None
ai_input_name = None
ai_labels = None


# =========================================================
# LOAD AI MODEL
# =========================================================

def get_ai_model():

    global ai_session
    global ai_input_name
    global ai_labels

    # If already loaded, reuse it
    if ai_session is not None:

        return (
            ai_session,
            ai_input_name,
            ai_labels
        )

    print(
        "Downloading/loading crop disease AI model..."
    )

    # -----------------------------------------------------
    # Download required files
    # -----------------------------------------------------

    model_dir = snapshot_download(
        repo_id=AI_MODEL_REPO,
        allow_patterns=[
            "efficientnet_v2_s_best.onnx",
            "efficientnet_v2_s_best.onnx.data",
            "classes.json"
        ]
    )

    print(
        "Hugging Face model directory:"
    )

    print(model_dir)

    # -----------------------------------------------------
    # Local model directory
    # -----------------------------------------------------

    local_model_dir = os.path.join(
        os.path.dirname(__file__),
        "ai_model"
    )

    os.makedirs(
        local_model_dir,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Source files
    # -----------------------------------------------------

    source_model = os.path.join(
        model_dir,
        "efficientnet_v2_s_best.onnx"
    )

    source_data = os.path.join(
        model_dir,
        "efficientnet_v2_s_best.onnx.data"
    )

    source_labels = os.path.join(
        model_dir,
        "classes.json"
    )

    # -----------------------------------------------------
    # Local files
    # -----------------------------------------------------

    model_path = os.path.join(
        local_model_dir,
        "efficientnet_v2_s_best.onnx"
    )

    data_path = os.path.join(
        local_model_dir,
        "efficientnet_v2_s_best.onnx.data"
    )

    labels_path = os.path.join(
        local_model_dir,
        "classes.json"
    )

    # -----------------------------------------------------
    # Copy model
    # -----------------------------------------------------

    print(
        "Preparing local AI model files..."
    )

    if not os.path.exists(model_path):

        print(
            "Copying ONNX model..."
        )

        shutil.copy2(
            source_model,
            model_path
        )

    else:

        print(
            "ONNX model already exists."
        )

    # -----------------------------------------------------
    # Copy external data
    # -----------------------------------------------------

    if not os.path.exists(data_path):

        print(
            "Copying ONNX external data..."
        )

        shutil.copy2(
            source_data,
            data_path
        )

    else:

        print(
            "ONNX external data already exists."
        )

    # -----------------------------------------------------
    # Copy labels
    # -----------------------------------------------------

    if not os.path.exists(labels_path):

        print(
            "Copying class labels..."
        )

        shutil.copy2(
            source_labels,
            labels_path
        )

    else:

        print(
            "Class labels already exist."
        )

    # -----------------------------------------------------
    # Load classes
    # -----------------------------------------------------

    print(
        "Loading class labels..."
    )

    with open(
        labels_path,
        "r",
        encoding="utf-8"
    ) as file:

        class_data = json.load(file)

    # classes.json structure:
    #
    # {
    #     "model_name": "...",
    #     "image_size": 224,
    #     "num_classes": 38,
    #     "classes": [...]
    # }

    ai_labels = class_data["classes"]

    print(
        f"Loaded {len(ai_labels)} disease classes."
    )

    # -----------------------------------------------------
    # Load ONNX model
    # -----------------------------------------------------

    print(
        "Loading ONNX model..."
    )

    ai_session = ort.InferenceSession(
        model_path,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    # -----------------------------------------------------
    # Input name
    # -----------------------------------------------------

    ai_input_name = (
        ai_session
        .get_inputs()[0]
        .name
    )

    print(
        "Crop disease AI model loaded successfully."
    )

    print(
        "AI input name:",
        ai_input_name
    )

    return (
        ai_session,
        ai_input_name,
        ai_labels
    )


# =========================================================
# PREDICT CROP DISEASE
# =========================================================

def predict_crop_disease(image):

    session, input_name, labels = (
        get_ai_model()
    )

    # -----------------------------------------------------
    # Convert image to RGB
    # -----------------------------------------------------

    image = image.convert("RGB")

    # -----------------------------------------------------
    # Resize
    # -----------------------------------------------------

    image = image.resize(
        (224, 224)
    )

    # -----------------------------------------------------
    # Convert to NumPy
    # -----------------------------------------------------

    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    # -----------------------------------------------------
    # Normalize 0-255 -> 0-1
    # -----------------------------------------------------

    image_array = (
        image_array / 255.0
    )

    # -----------------------------------------------------
    # ImageNet normalization
    # -----------------------------------------------------

    mean = np.array(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=np.float32
    )

    std = np.array(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=np.float32
    )

    image_array = (
        image_array - mean
    ) / std

    # -----------------------------------------------------
    # HWC -> CHW
    # -----------------------------------------------------

    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    # -----------------------------------------------------
    # Add batch dimension
    # -----------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # -----------------------------------------------------
    # ONNX inference
    # -----------------------------------------------------

    outputs = session.run(
        None,
        {
            input_name: image_array
        }
    )

    scores = outputs[0][0]

    # -----------------------------------------------------
    # Softmax
    # -----------------------------------------------------

    exp_scores = np.exp(
        scores - np.max(scores)
    )

    probabilities = (
        exp_scores /
        exp_scores.sum()
    )

    # -----------------------------------------------------
    # Top 5
    # -----------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[::-1][:5]

    predictions = []

    for index in top_indices:

        index = int(index)

        # Get actual disease name
        if index < len(labels):

            label = labels[index]

        else:

            label = (
                f"Class {index}"
            )

        # Probability as decimal
        score = float(
            probabilities[index]
        )

        predictions.append(
            {
                "label": label,
                "score": round(
                    score,
                    6
                )
            }
        )

    return predictions


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": (
            "KrushiVaani Backend is Running!"
        ),
        "status": "success"
    }


# =========================================================
# FARMER PROBLEM
# =========================================================

class ProblemRequest(BaseModel):

    problem: str


# =========================================================
# CLASSIFY FARMER PROBLEM
# =========================================================

# =========================================================
# CLASSIFY FARMER PROBLEM
# =========================================================

def classify_problem(problem: str):

    text = problem.lower().strip()

    # -----------------------------------------------------
    # TRACTOR SERVICE / REPAIR
    # -----------------------------------------------------
    tractor_service_words = [
        "tractor",
        "ಟ್ರ್ಯಾಕ್ಟರ್",
        "ಟ್ರಾಕ್ಟರ್",
        "tractor service",
        "tractor repair",
        "ಟ್ರ್ಯಾಕ್ಟರ್ ಸರ್ವಿಸ್",
        "ಟ್ರ್ಯಾಕ್ಟರ್ ರಿಪೇರಿ",
        "ಟ್ರ್ಯಾಕ್ಟರ್ ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ",
        "ಟ್ರ್ಯಾಕ್ಟರ್ ಕೆಟ್ಟು",
        "ಟ್ರ್ಯಾಕ್ಟರ್ ಕೆಟ್ಟಿದೆ"
    ]

    # -----------------------------------------------------
    # FARM MACHINERY SERVICE / REPAIR
    # -----------------------------------------------------
    machinery_service_words = [
        "sprayer",
        "ಸ್ಪ್ರೇಯರ್",
        "plough",
        "ploughing machine",
        "ನೇಗಿಲು",
        "ರೋಟಾವೇಟರ್",
        "rotavator",
        "cultivator",
        "ಕಲ್ಟಿವೇಟರ್",
        "harvester",
        "ಹಾರ್ವೆಸ್ಟರ್",
        "farm machine",
        "farm machinery",
        "ಕೃಷಿ ಯಂತ್ರ",
        "ಕೃಷಿ ಯಂತ್ರೋಪಕರಣ",
        "machine repair",
        "ಯಂತ್ರ ರಿಪೇರಿ",
        "ಯಂತ್ರ ಕೆಟ್ಟಿದೆ",
        "ಯಂತ್ರ ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ"
    ]

    # -----------------------------------------------------
    # PUMP / MOTOR SERVICE
    # -----------------------------------------------------
    pump_motor_service_words = [
        "pump",
        "motor",
        "water pump",
        "farm motor",
        "ಪಂಪ್",
        "ಮೋಟಾರ್",
        "ನೀರಿನ ಪಂಪ್",
        "ನೀರಿನ ಮೋಟಾರ್",
        "ಕೃಷಿ ಮೋಟಾರ್",
        "ಮೋಟಾರ್ ಕೆಟ್ಟಿದೆ",
        "ಮೋಟಾರ್ ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ",
        "ಪಂಪ್ ಕೆಟ್ಟಿದೆ",
        "ಪಂಪ್ ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ"
    ]

    # -----------------------------------------------------
    # SEED / SEED SHOP HELP
    # -----------------------------------------------------
    seed_service_words = [
        "seed",
        "seeds",
        "ಬೀಜ",
        "ಬೀಜಗಳು",
        "ಬೀಜ ಬೇಕು",
        "ಬೀಜ ಖರೀದಿ",
        "ಬೀಜ ಅಂಗಡಿ",
        "seed shop",
        "seed service"
    ]

    # -----------------------------------------------------
    # AGRICULTURE SHOP / INPUT SERVICE
    # -----------------------------------------------------
    agri_shop_service_words = [
        "agri shop",
        "agriculture shop",
        "ಕೃಷಿ ಅಂಗಡಿ",
        "ಕೃಷಿ ಮಳಿಗೆ",
        "ಗೊಬ್ಬರ ಅಂಗಡಿ",
        "ಔಷಧಿ ಅಂಗಡಿ",
        "pesticide shop",
        "fertilizer shop",
        "ಕೀಟನಾಶಕ ಅಂಗಡಿ",
        "ಕೃಷಿ ಸಾಮಗ್ರಿ",
        "ಕೃಷಿ ಸಾಮಾನು"
    ]

    # -----------------------------------------------------
    # VETERINARY SERVICE
    # -----------------------------------------------------
    veterinary_service_words = [
        "veterinary",
        "vet",
        "animal doctor",
        "cow doctor",
        "cattle doctor",
        "ಪಶು ವೈದ್ಯ",
        "ಪಶು ವೈದ್ಯರು",
        "ಜಾನುವಾರು ವೈದ್ಯ",
        "ಹಸು ವೈದ್ಯ",
        "ಹಸುಗೆ ಚಿಕಿತ್ಸೆ",
        "ಜಾನುವಾರು ಚಿಕಿತ್ಸೆ",
        "ಕುರಿ ಚಿಕಿತ್ಸೆ",
        "ಮೇಕೆ ಚಿಕಿತ್ಸೆ"
    ]

    # -----------------------------------------------------
    # GENERAL FARM SERVICE
    # -----------------------------------------------------
    general_service_words = [
        "service ಬೇಕು",
        "service ಬೇಕಾಗಿದೆ",
        "ಸರ್ವಿಸ್ ಬೇಕು",
        "ಸರ್ವಿಸ್ ಬೇಕಾಗಿದೆ",
        "repair ಬೇಕು",
        "repair ಮಾಡಬೇಕು",
        "ರಿಪೇರಿ ಬೇಕು",
        "ರಿಪೇರಿ ಮಾಡಬೇಕು",
        "ರಿಪೇರಿ ಮಾಡಿಸಬೇಕು",
        "ಕೆಟ್ಟುಹೋಗಿದೆ",
        "ಕೆಟ್ಟಿದೆ",
        "ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ",
        "work ಮಾಡುತ್ತಿಲ್ಲ",
        "technician ಬೇಕು",
        "ಟೆಕ್ನಿಷಿಯನ್ ಬೇಕು"
    ]

    # -----------------------------------------------------
    # EXISTING FARM PROBLEM WORDS
    # -----------------------------------------------------
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

    # =====================================================
    # SERVICE CLASSIFICATION FIRST
    # =====================================================

    if any(word in text for word in tractor_service_words):
        return "tractor_service"

    elif any(word in text for word in pump_motor_service_words):
        return "pump_motor_service"

    elif any(word in text for word in machinery_service_words):
        return "farm_machinery_service"

    elif any(word in text for word in seed_service_words):
        return "seed_service"

    elif any(word in text for word in agri_shop_service_words):
        return "agri_shop_service"

    elif any(word in text for word in veterinary_service_words):
        return "veterinary_service"

    elif any(word in text for word in general_service_words):
        return "general_service"

    # =====================================================
    # EXISTING FARM PROBLEM CLASSIFICATION
    # =====================================================

    elif any(word in text for word in pest_words):
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

    # =====================================================
    # TRACTOR SERVICE
    # =====================================================

    if category == "tractor_service":

        return {
            "answer": (
                "ನಿಮ್ಮ ಟ್ರ್ಯಾಕ್ಟರ್‌ಗೆ service ಅಥವಾ repair ಅಗತ್ಯವಿರುವಂತೆ ಕಾಣುತ್ತಿದೆ. "
                "ಮೊದಲು ಸಮಸ್ಯೆ ಯಾವ ಭಾಗದಲ್ಲಿದೆ ಎಂದು ಗುರುತಿಸುವುದು ಮುಖ್ಯ."
            ),

            "main_advice": [
                "ಟ್ರ್ಯಾಕ್ಟರ್ start ಆಗುತ್ತಿದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
                "Engine oil ಮತ್ತು coolant level ಪರಿಶೀಲಿಸಿ.",
                "Battery ಮತ್ತು battery connection ಪರಿಶೀಲಿಸಿ.",
                "Tyre pressure ಮತ್ತು tyre condition ಪರಿಶೀಲಿಸಿ.",
                "ಅಸಾಮಾನ್ಯ ಶಬ್ದ ಅಥವಾ smoke ಬರುತ್ತಿದೆಯೇ ಎಂದು ಗಮನಿಸಿ."
            ],

            "do": [
                "ಟ್ರ್ಯಾಕ್ಟರ್‌ನ ಸಮಸ್ಯೆಯನ್ನು ಗಮನಿಸಿ.",
                "ಸಮಸ್ಯೆ ಯಾವಾಗ ಪ್ರಾರಂಭವಾಯಿತು ಎಂದು ನೆನಪಿಡಿ.",
                "Regular service schedule ಅನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "ಅಗತ್ಯವಿದ್ದರೆ authorized mechanic ಅಥವಾ qualified technician ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "dont": [
                "ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ engine ಅಥವಾ electrical parts ತೆರೆಯಬೇಡಿ.",
                "ಅಸಾಮಾನ್ಯ ಶಬ್ದ ಬಂದರೆ tractor ಅನ್ನು ಬಲವಂತವಾಗಿ ಓಡಿಸಬೇಡಿ.",
                "ತಜ್ಞರ ಸಲಹೆ ಇಲ್ಲದೆ engine oil ಅಥವಾ spare parts ಬದಲಾಯಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ನಿಮ್ಮ ಟ್ರ್ಯಾಕ್ಟರ್‌ನಲ್ಲಿ ಏನು ಸಮಸ್ಯೆ ಇದೆ ಎಂದು ತಿಳಿಸಿ. "
                "ಉದಾಹರಣೆಗೆ: start ಆಗುತ್ತಿಲ್ಲ, battery problem, engine problem, "
                "brake problem ಅಥವಾ service ಬೇಕು ಎಂದು ಹೇಳಬಹುದು."
            )
        }

    # =====================================================
    # PUMP / MOTOR SERVICE
    # =====================================================

    elif category == "pump_motor_service":

        return {
            "answer": (
                "ನಿಮ್ಮ ಕೃಷಿ ನೀರಿನ pump ಅಥವಾ motor ನಲ್ಲಿ service/repair ಅಗತ್ಯವಿರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "Motor start ಆಗುತ್ತಿದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
                "Power supply ಇದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
                "Switch, cable ಮತ್ತು connection ಅನ್ನು ಗಮನಿಸಿ.",
                "Motor unusual sound ಮಾಡುತ್ತಿದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
                "Water flow ಸರಿಯಾಗಿ ಬರುತ್ತಿದೆಯೇ ಎಂದು ಗಮನಿಸಿ."
            ],

            "do": [
                "Power supply ಅನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "Motor ಸಮಸ್ಯೆ ಯಾವಾಗ ಪ್ರಾರಂಭವಾಯಿತು ಎಂದು ಗಮನಿಸಿ.",
                "Qualified electrician ಅಥವಾ technician ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "dont": [
                "Power ON ಇರುವಾಗ electrical connection touch ಮಾಡಬೇಡಿ.",
                "ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ motor ತೆರೆಯಬೇಡಿ.",
                "Electrical problem ಇದ್ದಾಗ ಸ್ವತಃ repair ಮಾಡಲು ಪ್ರಯತ್ನಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: motor/pump ನಲ್ಲಿ ಯಾವ ಸಮಸ್ಯೆ ಇದೆ ಎಂದು ತಿಳಿಸಿ. "
                "ಉದಾಹರಣೆಗೆ: motor start ಆಗುತ್ತಿಲ್ಲ, water ಬರುತ್ತಿಲ್ಲ ಅಥವಾ "
                "motor sound ಮಾಡುತ್ತಿದೆ ಎಂದು ಹೇಳಬಹುದು."
            )
        }

    # =====================================================
    # FARM MACHINERY SERVICE
    # =====================================================

    elif category == "farm_machinery_service":

        return {
            "answer": (
                "ನಿಮ್ಮ ಕೃಷಿ ಯಂತ್ರೋಪಕರಣಕ್ಕೆ service ಅಥವಾ repair ಅಗತ್ಯವಿರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "ಯಂತ್ರದ ಸಮಸ್ಯೆ ಯಾವ ಭಾಗದಲ್ಲಿದೆ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
                "ಯಂತ್ರ start ಆಗುತ್ತಿದೆಯೇ ಎಂದು ಗಮನಿಸಿ.",
                "ಯಂತ್ರದಲ್ಲಿ unusual sound ಅಥವಾ vibration ಇದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "Blade, belt ಮತ್ತು moving parts ಗಳ condition ಪರಿಶೀಲಿಸಿ.",
                "Regular maintenance schedule ಅನ್ನು ಗಮನಿಸಿ."
            ],

            "do": [
                "ಯಂತ್ರದ model ಮತ್ತು ಸಮಸ್ಯೆಯನ್ನು note ಮಾಡಿ.",
                "ಯಂತ್ರವನ್ನು ಸ್ವಚ್ಛವಾಗಿ ಇಟ್ಟುಕೊಳ್ಳಿ.",
                "ಅಗತ್ಯವಿದ್ದರೆ qualified technician ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "dont": [
                "ಯಂತ್ರ running ಇರುವಾಗ moving parts touch ಮಾಡಬೇಡಿ.",
                "Safety guard ತೆಗೆದು machine operate ಮಾಡಬೇಡಿ.",
                "ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ machine repair ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ ಯಂತ್ರದಲ್ಲಿ ಸಮಸ್ಯೆ ಇದೆ ಮತ್ತು "
                "ಏನು ಸಮಸ್ಯೆ ಆಗಿದೆ ಎಂದು ತಿಳಿಸಿ. "
                "ಉದಾಹರಣೆಗೆ: sprayer ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ ಅಥವಾ rotavator repair ಬೇಕು."
            )
        }

    # =====================================================
    # SEED SERVICE
    # =====================================================

    elif category == "seed_service":

        return {
            "answer": (
                "ನಿಮಗೆ ಬೀಜ ಖರೀದಿ ಅಥವಾ ಸೂಕ್ತವಾದ ಬೀಜದ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕಾಗಿದೆ."
            ),

            "main_advice": [
                "ಬೆಳೆಯ ಪ್ರಕಾರಕ್ಕೆ ಸೂಕ್ತವಾದ variety ಆಯ್ಕೆ ಮಾಡಿ.",
                "Certified ಅಥವಾ ಉತ್ತಮ ಗುಣಮಟ್ಟದ ಬೀಜವನ್ನು ಆಯ್ಕೆ ಮಾಡಿ.",
                "Seed packet ಮೇಲಿನ expiry ಮತ್ತು certification information ಪರಿಶೀಲಿಸಿ.",
                "ನಿಮ್ಮ ಪ್ರದೇಶದ ಹವಾಮಾನ ಮತ್ತು ಮಣ್ಣಿಗೆ ಸೂಕ್ತವಾದ variety ಆಯ್ಕೆ ಮಾಡುವುದು ಉತ್ತಮ."
            ],

            "do": [
                "ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ನಿಮ್ಮ ಪ್ರದೇಶವನ್ನು ತಿಳಿಸಿ.",
                "ಬೀಜ ಖರೀದಿಸುವ ಮೊದಲು label ಮತ್ತು quality information ಪರಿಶೀಲಿಸಿ.",
                "ಅಧಿಕೃತ ಕೃಷಿ ಮಳಿಗೆ ಅಥವಾ ವಿಶ್ವಾಸಾರ್ಹ seller ಅನ್ನು ಆಯ್ಕೆ ಮಾಡಿ."
            ],

            "dont": [
                "ಗುಣಮಟ್ಟ ತಿಳಿಯದ ಬೀಜವನ್ನು ಖರೀದಿಸಬೇಡಿ.",
                "Expiry ಆಗಿರುವ seed ಬಳಸಬೇಡಿ.",
                "ಬೆಳೆಯ ಪ್ರಕಾರ ತಿಳಿಯದೆ random variety ಆಯ್ಕೆ ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ ಬೆಳೆಗೆ ಬೀಜ ಬೇಕು ಎಂದು ತಿಳಿಸಿ. "
                "ಉದಾಹರಣೆಗೆ: ಟೊಮೆಟೊ ಬೀಜ, ಅಕ್ಕಿ ಬೀಜ ಅಥವಾ ಮೆಣಸಿನಕಾಯಿ ಬೀಜ."
            )
        }

    # =====================================================
    # AGRICULTURE SHOP SERVICE
    # =====================================================

    elif category == "agri_shop_service":

        return {
            "answer": (
                "ನಿಮಗೆ ಕೃಷಿ ಸಾಮಗ್ರಿ ಅಥವಾ ಕೃಷಿ ಮಳಿಗೆಗೆ ಸಂಬಂಧಿಸಿದ ಸಹಾಯ ಬೇಕಾಗಿದೆ."
            ),

            "main_advice": [
                "ನಿಮಗೆ ಬೇಕಾದ ಕೃಷಿ ಸಾಮಗ್ರಿಯನ್ನು ಮೊದಲು ಗುರುತಿಸಿ.",
                "Fertilizer, seed ಅಥವಾ pesticide ಬೇಕೇ ಎಂದು ಸ್ಪಷ್ಟಪಡಿಸಿ.",
                "Product label ಮತ್ತು expiry information ಪರಿಶೀಲಿಸಿ.",
                "ಅಗತ್ಯವಿದ್ದರೆ ಕೃಷಿ ಅಧಿಕಾರಿಯ ಸಲಹೆ ಪಡೆಯಿರಿ."
            ],

            "do": [
                "ಬೇಕಾದ product ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ಅಧಿಕೃತ ಅಥವಾ ವಿಶ್ವಾಸಾರ್ಹ ಕೃಷಿ ಮಳಿಗೆಯಿಂದ ಖರೀದಿಸಿ."
            ],

            "dont": [
                "Label ಇಲ್ಲದ ಕೃಷಿ ಉತ್ಪನ್ನ ಖರೀದಿಸಬೇಡಿ.",
                "ಅಗತ್ಯವಿಲ್ಲದ pesticide ಅಥವಾ fertilizer ಖರೀದಿಸಬೇಡಿ.",
                "ತಜ್ಞರ ಸಲಹೆ ಇಲ್ಲದೆ chemical ಬಳಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ನಿಮಗೆ ಯಾವ ಕೃಷಿ ಸಾಮಗ್ರಿ ಬೇಕು ಎಂದು ತಿಳಿಸಿ. "
                "ಉದಾಹರಣೆಗೆ: ಬೀಜ, ಗೊಬ್ಬರ, pesticide ಅಥವಾ ಕೃಷಿ ಉಪಕರಣ."
            )
        }

    # =====================================================
    # VETERINARY SERVICE
    # =====================================================

    elif category == "veterinary_service":

        return {
            "answer": (
                "ನಿಮ್ಮ ಜಾನುವಾರಿಗೆ veterinary ಅಥವಾ animal health service ಅಗತ್ಯವಿರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "ಜಾನುವಾರದ ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.",
                "ಆಹಾರ ಸೇವನೆ ಮತ್ತು ನೀರು ಕುಡಿಯುವ ಪ್ರಮಾಣವನ್ನು ಗಮನಿಸಿ.",
                "ಜಾನುವಾರಕ್ಕೆ ಗಾಯ ಅಥವಾ ಅಸಾಮಾನ್ಯ ವರ್ತನೆ ಇದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "ಗಂಭೀರ ಲಕ್ಷಣಗಳಿದ್ದರೆ veterinary doctor ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "do": [
                "ಜಾನುವಾರದ ಸಮಸ್ಯೆಯನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ವಿವರಿಸಿ.",
                "ಲಕ್ಷಣಗಳು ಯಾವಾಗ ಪ್ರಾರಂಭವಾದವು ಎಂದು ಗಮನಿಸಿ.",
                "ಅಗತ್ಯವಿದ್ದರೆ veterinary doctor ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "dont": [
                "ವೈದ್ಯರ ಸಲಹೆ ಇಲ್ಲದೆ medicine ನೀಡಬೇಡಿ.",
                "ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ injection ನೀಡಬೇಡಿ.",
                "ಗಂಭೀರ ಸ್ಥಿತಿಯಲ್ಲಿ ಚಿಕಿತ್ಸೆ ವಿಳಂಬ ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ ಜಾನುವಾರಿಗೆ ಏನು ಸಮಸ್ಯೆ ಇದೆ ಎಂದು ತಿಳಿಸಿ. "
                "ಉದಾಹರಣೆಗೆ: ಹಸು ತಿನ್ನುತ್ತಿಲ್ಲ, ಮೇಕೆಗೆ ಗಾಯವಾಗಿದೆ ಅಥವಾ "
                "ಜಾನುವಾರಿಗೆ ಜ್ವರದ ಲಕ್ಷಣಗಳಿವೆ."
            )
        }

    # =====================================================
    # GENERAL FARM SERVICE
    # =====================================================

    elif category == "general_service":

        return {
            "answer": (
                "ನಿಮಗೆ ಕೃಷಿಗೆ ಸಂಬಂಧಿಸಿದ service ಅಥವಾ repair ಸಹಾಯ ಬೇಕಾಗಿದೆ."
            ),

            "main_advice": [
                "ಯಾವ ವಸ್ತು ಅಥವಾ ಯಂತ್ರಕ್ಕೆ service ಬೇಕು ಎಂದು ಮೊದಲು ಗುರುತಿಸಿ.",
                "ಸಮಸ್ಯೆಯ ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.",
                "ಅಗತ್ಯವಿದ್ದರೆ qualified technician ಅಥವಾ ಸಂಬಂಧಿತ service provider ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "do": [
                "ಸಮಸ್ಯೆಯ ವಸ್ತುವಿನ ಹೆಸರು ತಿಳಿಸಿ.",
                "ಸಮಸ್ಯೆ ಏನು ಎಂದು ವಿವರಿಸಿ.",
                "ಸಾಧ್ಯವಾದರೆ ಸಮಸ್ಯೆಯ photo ತೆಗೆದುಕೊಳ್ಳಿ."
            ],

            "dont": [
                "ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ machine ಅಥವಾ electrical equipment ತೆರೆಯಬೇಡಿ.",
                "ಅಗತ್ಯವಿಲ್ಲದೆ parts ಬದಲಾಯಿಸಬೇಡಿ.",
                "ಗಂಭೀರ ಸಮಸ್ಯೆಯನ್ನು ನಿರ್ಲಕ್ಷಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ ವಸ್ತು ಅಥವಾ ಯಂತ್ರಕ್ಕೆ service ಬೇಕು "
                "ಮತ್ತು ಏನು ಸಮಸ್ಯೆ ಇದೆ ಎಂದು ವಿವರವಾಗಿ ಹೇಳಿ."
            )
        }

    # =====================================================
    # EXISTING PEST
    # =====================================================

    elif category == "pest":

        return {
            "answer": (
                "ನಿಮ್ಮ ಬೆಳೆಗೆ ಕೀಟ ಅಥವಾ ಹುಳು ಸಮಸ್ಯೆ "
                "ಇರುವಂತೆ ಕಾಣುತ್ತಿದೆ. ಮೊದಲು ಯಾವ ಕೀಟ "
                "ಎಂದು ಗುರುತಿಸುವುದು ಮುಖ್ಯ."
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

    # =====================================================
    # EXISTING WATER
    # =====================================================

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

    # =====================================================
    # EXISTING LEAF
    # =====================================================

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

    # =====================================================
    # EXISTING DISEASE
    # =====================================================

    elif category == "disease":

        return {
            "answer": (
                "ನಿಮ್ಮ ಬೆಳೆಯಲ್ಲಿ ರೋಗದ ಲಕ್ಷಣಗಳು "
                "ಕಂಡುಬರುತ್ತಿರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
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

    # =====================================================
    # EXISTING FERTILIZER
    # =====================================================

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

    # =====================================================
    # UNKNOWN
    # =====================================================

    else:

        return {
            "answer": (
                "ನಿಮ್ಮ ಸಮಸ್ಯೆಯನ್ನು ಸಂಪೂರ್ಣವಾಗಿ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು "
                "ಸಾಧ್ಯವಾಗಲಿಲ್ಲ."
            ),

            "main_advice": [
                "ದಯವಿಟ್ಟು ಸಮಸ್ಯೆಯನ್ನು ಸ್ವಲ್ಪ ವಿವರವಾಗಿ ತಿಳಿಸಿ.",
                "ಸಾಧ್ಯವಾದರೆ ಬೆಳೆಯ ಅಥವಾ ಸಮಸ್ಯೆಯ ವಸ್ತುವಿನ ಹೆಸರನ್ನು ತಿಳಿಸಿ."
            ],

            "do": [
                "ಸಮಸ್ಯೆಯ ಹೆಸರು ಅಥವಾ ಲಕ್ಷಣಗಳನ್ನು ತಿಳಿಸಿ.",
                "ಸಾಧ್ಯವಾದರೆ photo upload ಮಾಡಿ."
            ],

            "dont": [
                "ಸಮಸ್ಯೆ ತಿಳಿಯದೆ ಔಷಧಿ ಅಥವಾ repair ಪ್ರಯತ್ನ ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಸಮಸ್ಯೆಯನ್ನು ಇನ್ನಷ್ಟು ವಿವರವಾಗಿ ಹೇಳಿ. "
                "KrushiVaani ನಿಮಗೆ ಸೂಕ್ತವಾದ ಸಹಾಯ ನೀಡಲು ಪ್ರಯತ್ನಿಸುತ್ತದೆ."
            )
        }


# =========================================================
# PROBLEM API
# =========================================================

@app.post("/problem")
def receive_problem(
    data: ProblemRequest
):

    problem = data.problem

    category = classify_problem(
        problem
    )

    response = generate_farmer_response(
        category
    )

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
def find_disease_information(
    label: str
):
    if not label:
        return None

    # -----------------------------------------------------
    # 1. Exact database match
    # -----------------------------------------------------

    info = get_disease_info(label)

    if info is not None:
        return info

    # -----------------------------------------------------
    # 2. Normalize label
    # -----------------------------------------------------

    normalized_label = normalize_label(label)

    # -----------------------------------------------------
    # 3. Exact normalized database match
    # -----------------------------------------------------

    for key, value in DISEASE_DATABASE.items():

        normalized_key = normalize_label(key)

        if normalized_label == normalized_key:
            return value

    # -----------------------------------------------------
    # 4. IMPORTANT:
    # Match crop + disease together
    # Do NOT match "healthy" alone.
    # -----------------------------------------------------

    label_parts = normalized_label.split("___")

    if len(label_parts) >= 2:

        crop_part = label_parts[0]
        disease_part = label_parts[1]

        for key, value in DISEASE_DATABASE.items():

            normalized_key = normalize_label(key)

            key_parts = normalized_key.split("___")

            if len(key_parts) >= 2:

                db_crop = key_parts[0]
                db_disease = key_parts[1]

                # Both crop and disease must match
                if (
                    crop_part == db_crop
                    and disease_part == db_disease
                ):
                    return value

    # -----------------------------------------------------
    # 5. Disease-specific matching
    # -----------------------------------------------------

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

    label_lower = label.lower()

    for disease_key, keywords in disease_keywords.items():

        for keyword in keywords:

            if keyword in label_lower:

                for db_key, value in DISEASE_DATABASE.items():

                    normalized_db_key = normalize_label(
                        db_key
                    )

                    if disease_key in normalized_db_key:

                        # If possible, also verify crop
                        label_crop = normalized_label.split("___")[0]
                        db_crop = normalized_db_key.split("___")[0]

                        if label_crop == db_crop:
                            return value

    # -----------------------------------------------------
    # 6. No database match
    # Let generic information handle it
    # -----------------------------------------------------

    return None

# =========================================================
# GENERIC DISEASE INFORMATION
# =========================================================
def create_generic_disease_info(
    label: str
):
    """
    Create user-friendly generic information
    directly from the AI model label.
    """

    if not label:
        label = "Unknown"

    # -----------------------------------------------------
    # Convert AI label into readable text
    # -----------------------------------------------------

    clean_label = label.replace(
        "___",
        " | "
    )

    clean_label = clean_label.replace(
        "_",
        " "
    )

    # -----------------------------------------------------
    # Extract crop and disease
    # -----------------------------------------------------

    parts = clean_label.split("|")

    if len(parts) >= 2:

        crop_name = parts[0].strip()

        disease_name = "|".join(
            parts[1:]
        ).strip()

    else:

        crop_name = "Unknown crop"

        disease_name = clean_label.strip()

    # -----------------------------------------------------
    # Make crop name user friendly
    # -----------------------------------------------------

    crop_display_names = {

        "Corn (maize)":
            "ಮೆಕ್ಕೆಜೋಳ (Corn)",

        "Tomato":
            "ಟೊಮ್ಯಾಟೊ (Tomato)",

        "Potato":
            "ಆಲೂಗಡ್ಡೆ (Potato)",

        "Soybean":
            "ಸೋಯಾಬೀನ್ (Soybean)",

        "Apple":
            "ಸೇಬು (Apple)",

        "Grape":
            "ದ್ರಾಕ್ಷಿ (Grape)",

        "Peach":
            "ಪೀಚ್ (Peach)",

        "Cherry (including sour)":
            "ಚೆರ್ರಿ (Cherry)",

        "Pepper, bell":
            "ಕ್ಯಾಪ್ಸಿಕಂ (Bell Pepper)",

        "Blueberry":
            "ಬ್ಲೂಬೆರ್ರಿ (Blueberry)",

        "Raspberry":
            "ರಾಸ್ಪ್ಬೆರಿ (Raspberry)",

        "Squash":
            "ಸ್ಕ್ವಾಶ್ (Squash)",

        "Strawberry":
            "ಸ್ಟ್ರಾಬೆರಿ (Strawberry)"
    }

    crop_name = crop_display_names.get(
        crop_name,
        crop_name
    )

    # -----------------------------------------------------
    # Healthy result
    # -----------------------------------------------------

    if (
        "healthy"
        in disease_name.lower()
    ):

        disease_display = (
            "ಆರೋಗ್ಯಕರ / Healthy"
        )

        summary = (
            "AI ಚಿತ್ರ ವಿಶ್ಲೇಷಣೆಯ ಪ್ರಕಾರ "
            "ಈ ಬೆಳೆಯಲ್ಲಿ ಪ್ರಮುಖ ರೋಗದ ಲಕ್ಷಣಗಳು "
            "ಕಾಣಿಸದಿರುವ ಸಾಧ್ಯತೆ ಇದೆ."
        )

        do_list = [

            "ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",

            "ಎಲೆಗಳ ಬಣ್ಣ ಮತ್ತು ಬೆಳವಣಿಗೆಯನ್ನು ಗಮನಿಸಿ.",

            "ನೀರಾವರಿ ಮತ್ತು ಗೊಬ್ಬರವನ್ನು "
            "ಅಗತ್ಯಕ್ಕೆ ಅನುಗುಣವಾಗಿ ನೀಡಿ."
        ]

        dont_list = [

            "ಅಗತ್ಯವಿಲ್ಲದೆ pesticide ಅಥವಾ "
            "ಔಷಧಿ ಬಳಸಬೇಡಿ.",

            "AI result ಅನ್ನು ಮಾತ್ರ ಆಧರಿಸಿ "
            "ಖಚಿತ diagnosis ಎಂದು ಪರಿಗಣಿಸಬೇಡಿ."
        ]

    else:

        # -------------------------------------------------
        # Disease result
        # -------------------------------------------------

        disease_display = disease_name

        summary = (
            "AI ಚಿತ್ರ ವಿಶ್ಲೇಷಣೆಯ ಆಧಾರದ ಮೇಲೆ "
            "ಈ ಸಮಸ್ಯೆ ಗುರುತಿಸಲಾಗಿದೆ. "
            "ಆದರೆ AI result ಅನ್ನು ಖಚಿತ "
            "ರೋಗನಿರ್ಣಯ ಎಂದು ಪರಿಗಣಿಸಬಾರದು."
        )

        do_list = [

            "ಪೀಡಿತ ಬೆಳೆಯ ಸ್ಪಷ್ಟವಾದ photo "
            "ತೆಗೆದು ಮತ್ತೆ ಪರಿಶೀಲಿಸಿ.",

            "ಬೆಳೆಯ ಲಕ್ಷಣಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಗಮನಿಸಿ.",

            "ಸಮಸ್ಯೆ ಹೆಚ್ಚಾದರೆ ಸ್ಥಳೀಯ ಕೃಷಿ "
            "ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
        ]

        dont_list = [

            "AI result ಮಾತ್ರ ಆಧರಿಸಿ "
            "pesticide ಅಥವಾ ಔಷಧಿ ಬಳಸಬೇಡಿ.",

            "Confidence ಕಡಿಮೆ ಇದ್ದಾಗ result ಅನ್ನು "
            "ಖಚಿತ diagnosis ಎಂದು ಪರಿಗಣಿಸಬೇಡಿ.",

            "ರೋಗ ಖಚಿತವಾಗದೆ ದುಬಾರಿ treatment "
            "ಆರಂಭಿಸಬೇಡಿ."
        ]

    # -----------------------------------------------------
    # FINAL INFORMATION
    # -----------------------------------------------------

    return {

        "crop": crop_name,

        "disease": disease_display,

        "summary": summary,

        "do": do_list,

        "dont": dont_list
    }
    label_text = label.replace(
        "_",
        " "
    )

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
                "disease_info": (
                    create_generic_disease_info(
                        "Unknown"
                    )
                ),
                "all_predictions": []
            }

        # -------------------------------------------------
        # OPEN IMAGE
        # -------------------------------------------------

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")

        # -------------------------------------------------
        # AI PREDICTION
        # -------------------------------------------------

        predictions = predict_crop_disease(
            image
        )

        if not predictions:

            return {
                "message": (
                    "AI could not analyze the image."
                ),
                "filename": file.filename,
                "crop_or_disease": "Unknown",
                "confidence": 0.0,
                "disease_info": (
                    create_generic_disease_info(
                        "Unknown"
                    )
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

            disease_info = (
                create_generic_disease_info(
                    label
                )
            )

        # -------------------------------------------------
        # FINAL RESPONSE
        # -------------------------------------------------

        return {
            "message": (
                "Image analyzed successfully!"
            ),

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

            "disease_info": (
                create_generic_disease_info(
                    "Unknown"
                )
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

        "title_kn": (
            "ಪ್ರಧಾನಮಂತ್ರಿ ಕಿಸಾನ್ ಸಮ್ಮಾನ್ ನಿಧಿ"
        ),

        "description": (
            "ಅರ್ಹ ರೈತ ಕುಟುಂಬಗಳಿಗೆ ಕೇಂದ್ರ ಸರ್ಕಾರದ "
            "ಆದಾಯ ಸಹಾಯ ಯೋಜನೆ."
        ),

        "benefit": (
            "ಅರ್ಹ ರೈತರಿಗೆ ವರ್ಷಕ್ಕೆ ₹6,000 "
            "ನೇರವಾಗಿ ಬ್ಯಾಂಕ್ ಖಾತೆಗೆ."
        ),

        "eligibility": (
            "ಅರ್ಹ ಭೂಮಾಲೀಕ ರೈತ ಕುಟುಂಬಗಳು."
        ),

        "category": "Financial Support",

        "official_url": (
            "https://pmkisan.gov.in/"
        )
    },

    {
        "id": 2,

        "title": (
            "Pradhan Mantri Fasal Bima Yojana"
        ),

        "title_kn": (
            "ಪ್ರಧಾನಮಂತ್ರಿ ಫಸಲ್ ಬಿಮಾ ಯೋಜನೆ"
        ),

        "description": (
            "ಬೆಳೆ ಹಾನಿಯಿಂದ ರೈತರಿಗೆ ವಿಮಾ ರಕ್ಷಣೆ "
            "ನೀಡುವ ಯೋಜನೆ."
        ),

        "benefit": (
            "ಅರ್ಹ ಬೆಳೆ ನಷ್ಟಗಳಿಗೆ ವಿಮಾ ಪರಿಹಾರ "
            "ಪಡೆಯಲು ಅವಕಾಶ."
        ),

        "eligibility": (
            "ಯೋಜನೆಯ ನಿಯಮಗಳಿಗೆ ಒಳಪಡುವ ರೈತರು."
        ),

        "category": "Crop Insurance",

        "official_url": (
            "https://pmfby.gov.in/"
        )
    },

    {
        "id": 3,

        "title": "Kisan Credit Card",

        "title_kn": (
            "ಕಿಸಾನ್ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್"
        ),

        "description": (
            "ಕೃಷಿ ಚಟುವಟಿಕೆಗಳಿಗೆ ಸಾಲ ಸೌಲಭ್ಯ "
            "ಪಡೆಯಲು ಸಹಾಯ ಮಾಡುವ ಯೋಜನೆ."
        ),

        "benefit": (
            "ಕೃಷಿ ಅಗತ್ಯಗಳಿಗೆ ಸಾಲ ಸೌಲಭ್ಯ."
        ),

        "eligibility": (
            "ಅರ್ಹ ರೈತರು ಮತ್ತು ಕೃಷಿ ಚಟುವಟಿಕೆಯಲ್ಲಿ "
            "ತೊಡಗಿರುವವರು."
        ),

        "category": "Agricultural Credit",

        "official_url": (
            "https://www.myscheme.gov.in/schemes/kcc"
        )
    }
]


@app.get("/schemes")
def get_government_schemes():

    return {
        "message": (
            "Government schemes loaded successfully"
        ),

        "count": len(
            GOVERNMENT_SCHEMES
        ),

        "schemes": GOVERNMENT_SCHEMES
    }


# =========================================================
# AGRICULTURE MEDICINES
# =========================================================

@app.get("/medicines")
def get_medicines():

    medicines = get_all_medicines()

    return {
        "message": (
            "Agriculture medicines loaded successfully"
        ),

        "count": len(
            medicines
        ),

        "medicines": medicines
    }


@app.get("/medicines/{medicine_id}")
def get_medicine(
    medicine_id: int
):

    medicine = get_medicine_by_id(
        medicine_id
    )

    if medicine is None:

        return {
            "message": (
                "Medicine information not found."
            ),

            "medicine": None
        }

    return {
        "message": (
            "Medicine information loaded successfully"
        ),

        "medicine": medicine
    }


# =========================================================
# FARMER GOVERNMENT NOTIFICATIONS
# =========================================================

GOVERNMENT_NOTIFICATIONS = [

    {
        "id": 1,

        "title": (
            "ಹೊಸ ರೈತ ಯೋಜನೆ ಮಾಹಿತಿ"
        ),

        "message": (
            "ರೈತರಿಗೆ ಲಭ್ಯವಿರುವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ "
            "ಮಾಹಿತಿಗಾಗಿ KrushiVaani ಪರಿಶೀಲಿಸಿ."
        ),

        "category": "Government Scheme",

        "date": "2026-10-04",

        "is_new": True
    },

    {
        "id": 2,

        "title": (
            "ಬೆಳೆ ವಿಮೆ ಮಾಹಿತಿ"
        ),

        "message": (
            "ಬೆಳೆ ಹಾನಿಯಿಂದ ರಕ್ಷಣೆ ಪಡೆಯಲು "
            "ಪ್ರಧಾನಮಂತ್ರಿ ಫಸಲ್ ಬಿಮಾ ಯೋಜನೆಯ "
            "ವಿವರಗಳನ್ನು ಪರಿಶೀಲಿಸಿ."
        ),

        "category": "Crop Insurance",

        "date": "2026-10-04",

        "is_new": True
    }
]


@app.get("/notifications")
def get_notifications():

    return {
        "message": (
            "Farmer notifications loaded successfully"
        ),

        "count": len(
            GOVERNMENT_NOTIFICATIONS
        ),

        "notifications": (
            GOVERNMENT_NOTIFICATIONS
        )
    }