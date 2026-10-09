import os
import io
import json
import shutil

from dotenv import load_dotenv
from google import genai

load_dotenv()

import numpy as np
import onnxruntime as ort

from PIL import Image
from huggingface_hub import snapshot_download

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from urllib import request as urllib_request
from urllib import error as urllib_error

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
# GEMINI AI HELPER
# =========================================================

GEMINI_MODEL = "gemini-3.8-flash"


def generate_gemini_farmer_response(problem, smart_context=None, previous_context=None):
    """Generate a Kannada farming answer; return None if Gemini fails."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are KrushiVaani, a careful Kannada-speaking farming assistant for Indian farmers.
Answer primarily in simple, natural Kannada.
Use the farmer's current question and context; handle follow-up questions using previous context.
Do not invent certainty about crop diseases, pesticide doses, prices, schemes, phone numbers, or local services.
If information is insufficient, clearly say what details or photos are needed.
Do not recommend mixing chemicals or using pesticides without confirming the cause.
Keep the response practical and farmer-friendly.

Current question:
{problem}

Detected crop context:
{json.dumps(smart_context or {}, ensure_ascii=False)}

Previous conversation context:
{json.dumps(previous_context or {}, ensure_ascii=False)}

Return ONLY a valid JSON object with exactly these keys:
{{
  "answer": "Short direct answer in Kannada",
  "main_advice": ["Practical step 1", "Practical step 2"],
  "do": ["Helpful action"],
  "dont": ["Action to avoid"],
  "next_step": "One clear next step in Kannada"
}}
All values must be in Kannada where practical. main_advice, do, and dont must be arrays of strings.
"""

        result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )

        parsed = json.loads(result.text or "")
        required = ["answer", "main_advice", "do", "dont", "next_step"]

        if not all(key in parsed for key in required):
            return None
        if not isinstance(parsed["answer"], str):
            return None
        for key in ["main_advice", "do", "dont"]:
            if not isinstance(parsed[key], list) or not all(
                isinstance(item, str) for item in parsed[key]
            ):
                return None
        if not isinstance(parsed["next_step"], str):
            return None

        return parsed

    except Exception as exc:
        print(f"Gemini response unavailable; using existing system: {type(exc).__name__}")
        return None



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
    location: Optional[str] = None

    # Smart Follow-up Conversation
    conversation_id: Optional[str] = None
    previous_context: Optional[dict] = None


# =========================================================
# CLASSIFY FARMER PROBLEM
# =========================================================

# =========================================================
# CLASSIFY FARMER PROBLEM
# =========================================================
# =========================================================
# SMART CLASSIFY FARMER PROBLEM
# =========================================================
def classify_problem(problem: str):
    text = problem.lower().strip()

    # ---------------------------------------------------------
    # TRACTOR START PROBLEM
    # ---------------------------------------------------------
    tractor_start_keywords = [
        "tractor start",
        "tractor won't start",
        "tractor wont start",
        "tractor not start",
        "tractor starting problem",
        "tractor start problem",
        "tractor start ಆಗುತ್ತಿಲ್ಲ",
        "tractor ಸ್ಟಾರ್ಟ್ ಆಗುತ್ತಿಲ್ಲ",
        "ಟ್ರಾಕ್ಟರ್ ಸ್ಟಾರ್ಟ್ ಆಗುತ್ತಿಲ್ಲ",
        "ಟ್ರಾಕ್ಟರ್ ಶುರು ಆಗುತ್ತಿಲ್ಲ",
        "ಟ್ರಾಕ್ಟರ್ ಸ್ಟಾರ್ಟ್ ಸಮಸ್ಯೆ",
    ]

    if any(k in text for k in tractor_start_keywords):
        return "tractor_start_problem"

    # ---------------------------------------------------------
    # TRACTOR BATTERY
    # ---------------------------------------------------------
    tractor_battery_keywords = [
        "tractor battery",
        "battery down",
        "battery dead",
        "battery weak",
        "battery problem",
        "battery issue",
        "tractor ಬ್ಯಾಟರಿ",
        "ಬ್ಯಾಟರಿ ಡೌನ್",
        "ಬ್ಯಾಟರಿ ಸಮಸ್ಯೆ",
        "ಟ್ರಾಕ್ಟರ್ ಬ್ಯಾಟರಿ",
    ]

    if any(k in text for k in tractor_battery_keywords):
        return "tractor_battery_problem"

    # ---------------------------------------------------------
    # TRACTOR ENGINE
    # ---------------------------------------------------------
    tractor_engine_keywords = [
        "tractor engine",
        "engine problem",
        "engine issue",
        "engine overheating",
        "engine smoke",
        "engine heat",
        "tractor overheating",
        "ಟ್ರಾಕ್ಟರ್ ಎಂಜಿನ್",
        "ಎಂಜಿನ್ ಸಮಸ್ಯೆ",
        "ಎಂಜಿನ್ ಬಿಸಿ",
        "ಎಂಜಿನ್ ಹೊಗೆ",
        "ಟ್ರಾಕ್ಟರ್ ಹೊಗೆ",
    ]

    if any(k in text for k in tractor_engine_keywords):
        return "tractor_engine_problem"

    # ---------------------------------------------------------
    # TRACTOR BRAKE
    # ---------------------------------------------------------
    tractor_brake_keywords = [
        "tractor brake",
        "brake problem",
        "brake issue",
        "brake not working",
        "brake weak",
        "ಟ್ರಾಕ್ಟರ್ ಬ್ರೇಕ್",
        "ಬ್ರೇಕ್ ಸಮಸ್ಯೆ",
        "ಬ್ರೇಕ್ ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ",
        "ಬ್ರೇಕ್ ಸರಿಯಾಗಿ ಇಲ್ಲ",
    ]

    if any(k in text for k in tractor_brake_keywords):
        return "tractor_brake_problem"

    # ---------------------------------------------------------
    # GENERAL TRACTOR SERVICE
    # ---------------------------------------------------------
    tractor_service_keywords = [
        "tractor service",
        "tractor repair",
        "tractor mechanic",
        "tractor maintenance",
        "tractor servicing",
        "ಟ್ರಾಕ್ಟರ್ ಸರ್ವಿಸ್",
        "ಟ್ರಾಕ್ಟರ್ ಸರ್ವೀಸ್",
        "ಟ್ರಾಕ್ಟರ್ ರಿಪೇರಿ",
        "ಟ್ರಾಕ್ಟರ್ ರಿಪೇರ್",
        "ಟ್ರಾಕ್ಟರ್ ಮೆಕ್ಯಾನಿಕ್",
        "ಟ್ರಾಕ್ಟರ್ ಮೆಕಾನಿಕ್",
        "ಟ್ರಾಕ್ಟರ್ ನಿರ್ವಹಣೆ",
    ]

    if any(k in text for k in tractor_service_keywords):
        return "tractor_service"

    # ---------------------------------------------------------
    # PUMP / MOTOR
    # ---------------------------------------------------------
    pump_motor_keywords = [
        "pump",
        "motor",
        "water pump",
        "farm pump",
        "pump repair",
        "motor repair",
        "pump service",
        "motor service",
        "ಪಂಪ್",
        "ಮೋಟಾರ್",
        "ನೀರಿನ ಪಂಪ್",
        "ಮೋಟಾರ್ ರಿಪೇರಿ",
        "ಪಂಪ್ ರಿಪೇರಿ",
        "ಮೋಟಾರ್ ಸರ್ವಿಸ್",
        "ಪಂಪ್ ಸರ್ವಿಸ್",
    ]

    if any(k in text for k in pump_motor_keywords):
        return "pump_motor_service"

    # ---------------------------------------------------------
    # FARM MACHINERY
    # ---------------------------------------------------------
    machinery_keywords = [
        "sprayer",
        "rotavator",
        "cultivator",
        "harvester",
        "thresher",
        "farm machine",
        "farm machinery",
        "machine repair",
        "machinery repair",
        "ಸ್ಪ್ರೇಯರ್",
        "ಸ್ಪ್ರೇಯರ್ ರಿಪೇರಿ",
        "ರೋಟಾವೇಟರ್",
        "ಕಲ್ಟಿವೇಟರ್",
        "ಹಾರ್ವೆಸ್ಟರ್",
        "ಥ್ರೆಷರ್",
        "ಕೃಷಿ ಯಂತ್ರ",
        "ಕೃಷಿ ಯಂತ್ರೋಪಕರಣ",
        "ಯಂತ್ರ ರಿಪೇರಿ",
    ]

    if any(k in text for k in machinery_keywords):
        return "farm_machinery_service"

    # ---------------------------------------------------------
    # SEEDS
    # ---------------------------------------------------------
    seed_keywords = [
        "seed",
        "seeds",
        "seed shop",
        "seed dealer",
        "ಬೀಜ",
        "ಬೀಜಗಳು",
        "ಬೀಜ ಅಂಗಡಿ",
        "ಬೀಜ ಮಾರಾಟಗಾರ",
    ]

    if any(k in text for k in seed_keywords):
        return "seed_service"

    # ---------------------------------------------------------
    # AGRICULTURE SHOP
    # ---------------------------------------------------------
    agri_shop_keywords = [
        "agriculture shop",
        "agri shop",
        "fertilizer shop",
        "pesticide shop",
        "krushi shop",
        "ಕೃಷಿ ಅಂಗಡಿ",
        "ಕೃಷಿ ಶಾಪ್",
        "ಅಗ್ರಿ ಶಾಪ್",
        "ಗೊಬ್ಬರ ಅಂಗಡಿ",
        "ಕೀಟನಾಶಕ ಅಂಗಡಿ",
    ]

    if any(k in text for k in agri_shop_keywords):
        return "agri_shop_service"

    # ---------------------------------------------------------
    # VETERINARY / ANIMAL DOCTOR
    # ---------------------------------------------------------
    veterinary_keywords = [
        "veterinary",
        "veterinary doctor",
        "vet doctor",
        "animal doctor",
        "cow doctor",
        "buffalo doctor",
        "goat doctor",
        "sheep doctor",
        "cattle doctor",
        "livestock doctor",
        "cow problem",
        "buffalo problem",
        "goat problem",
        "ಹಸುವಿಗೆ",
        "ಹಸುವಿನ",
        "ಹಸು",
        "ಎಮ್ಮೆ",
        "ಮೇಕೆ",
        "ಕುರಿ",
        "ಜಾನುವಾರು",
        "ಪಶು ವೈದ್ಯ",
        "ಪಶುವೈದ್ಯ",
        "ಪಶು ವೈದ್ಯರು",
        "ಪಶುವೈದ್ಯರು",
        "ಪ್ರಾಣಿ ವೈದ್ಯ",
        "ಪ್ರಾಣಿ ವೈದ್ಯರು",
        "ಹಸುವಿಗೆ ಡಾಕ್ಟರ್",
        "ಹಸುವಿಗೆ ವೈದ್ಯ",
        "ಜಾನುವಾರು ವೈದ್ಯ",
    ]

    if any(k in text for k in veterinary_keywords):
        return "veterinary_service"

    # ---------------------------------------------------------
    # FARM LABOUR / WORKERS
    # ---------------------------------------------------------
    labour_keywords = [
        "farm labour",
        "farm labor",
        "farm worker",
        "farm workers",
        "agriculture labour",
        "agriculture labor",
        "agricultural worker",
        "workers for farm",
        "labour ಬೇಕು",
        "labor ಬೇಕು",
        "worker ಬೇಕು",
        "workers ಬೇಕು",
        "farm labour ಬೇಕು",
        "ಕೃಷಿ ಕಾರ್ಮಿಕ",
        "ಕೃಷಿ ಕಾರ್ಮಿಕರು",
        "ಕೃಷಿ ಕೆಲಸಗಾರ",
        "ಕೃಷಿ ಕೆಲಸಗಾರರು",
        "ಹೊಲ ಕೆಲಸಕ್ಕೆ ಜನ",
        "ಹೊಲಕ್ಕೆ ಕಾರ್ಮಿಕರು",
        "ಹೊಲದ ಕೆಲಸಕ್ಕೆ ಜನ",
        "ಕೆಲಸಗಾರರು ಬೇಕು",
        "ಕಾರ್ಮಿಕರು ಬೇಕು",
        "ಕೃಷಿ ಕೆಲಸಕ್ಕೆ ಜನ ಬೇಕು",
    ]

    if any(k in text for k in labour_keywords):
        return "farm_labour"

    # ---------------------------------------------------------
    # MACHINERY / TRACTOR RENTAL
    # ---------------------------------------------------------
    rental_keywords = [
        "tractor rental",
        "tractor rent",
        "tractor hire",
        "machine rental",
        "machinery rental",
        "farm machine rental",
        "tractor on rent",
        "tractor ಬಾಡಿಗೆ",
        "ಟ್ರಾಕ್ಟರ್ ಬಾಡಿಗೆ",
        "ಟ್ರಾಕ್ಟರ್ ಬಾಡಿಗೆ ಬೇಕು",
        "ಯಂತ್ರ ಬಾಡಿಗೆ",
        "ಕೃಷಿ ಯಂತ್ರ ಬಾಡಿಗೆ",
        "ಕೃಷಿ ಯಂತ್ರೋಪಕರಣ ಬಾಡಿಗೆ",
        "ಮೆಷಿನ್ ಬಾಡಿಗೆ",
        "ಮಷಿನ್ ಬಾಡಿಗೆ",
        "ಬಾಡಿಗೆ ಟ್ರಾಕ್ಟರ್",
    ]

    if any(k in text for k in rental_keywords):
        return "machinery_rental"

    # ---------------------------------------------------------
    # GENERAL FARM SERVICE
    # ---------------------------------------------------------
    general_service_keywords = [
        "farm service",
        "agriculture service",
        "farmer service",
        "service ಬೇಕು",
        "ಸೇವೆ ಬೇಕು",
        "ಕೃಷಿ ಸೇವೆ",
        "ರೈತ ಸೇವೆ",
        "ಕೃಷಿ ಸಹಾಯ",
        "ರೈತರಿಗೆ ಸಹಾಯ",
    ]

    if any(k in text for k in general_service_keywords):
        return "general_service"

    # ---------------------------------------------------------
    # PEST
    # ---------------------------------------------------------
    pest_keywords = [
        "pest",
        "insect",
        "worm",
        "bug",
        "ಕೀಟ",
        "ಹುಳು",
        "ಹುಳ",
        "ಕೀಟ ಬಂದಿದೆ",
        "ಹುಳು ಬಂದಿದೆ",
    ]

    if any(k in text for k in pest_keywords):
        return "pest"

    # ---------------------------------------------------------
    # WATER
    # ---------------------------------------------------------
    water_keywords = [
        "water problem",
        "no water",
        "water shortage",
        "irrigation",
        "ನೀರಿನ ಸಮಸ್ಯೆ",
        "ನೀರು ಇಲ್ಲ",
        "ನೀರಿನ ಕೊರತೆ",
        "ನೀರಾವರಿ",
    ]

    if any(k in text for k in water_keywords):
        return "water"

    # ---------------------------------------------------------
    # LEAF
    # ---------------------------------------------------------
    leaf_keywords = [
        "leaf problem",
        "leaf yellow",
        "yellow leaf",
        "leaves yellow",
        "ಎಲೆ ಸಮಸ್ಯೆ",
        "ಎಲೆ ಹಳದಿ",
        "ಎಲೆಗಳು ಹಳದಿ",
        "ಎಲೆ ಒಣಗುತ್ತಿದೆ",
    ]

    if any(k in text for k in leaf_keywords):
        return "leaf"

    # ---------------------------------------------------------
    # DISEASE
    # ---------------------------------------------------------
    disease_keywords = [
        "disease",
        "plant disease",
        "crop disease",
        "ರೋಗ",
        "ಬೆಳೆ ರೋಗ",
        "ಸಸ್ಯ ರೋಗ",
    ]

    if any(k in text for k in disease_keywords):
        return "disease"

    # ---------------------------------------------------------
    # FERTILIZER
    # ---------------------------------------------------------
    fertilizer_keywords = [
        "fertilizer",
        "fertiliser",
        "urea",
        "manure",
        "ಗೊಬ್ಬರ",
        "ಯೂರಿಯಾ",
        "ರಸಗೊಬ್ಬರ",
        "ಸಾವಯವ ಗೊಬ್ಬರ",
    ]

    if any(k in text for k in fertilizer_keywords):
        return "fertilizer"

    # ---------------------------------------------------------
    # UNKNOWN
    # ---------------------------------------------------------
    return "unknown"

# =========================================================
# SMART CONTEXT LAYER
# =========================================================

def extract_smart_context(problem: str):
    """
    Extract Crop + Problem + Intent from farmer question.
    This works together with the existing classify_problem().
    """

    text = problem.lower().strip()

    
    # =====================================================
    # CROP AGE DETECTION
    # =====================================================

    import re

    crop_age_days = None

    age_patterns = [
        r"\b(\d{1,3})\s*(?:days?|day)\s*(?:old|ago)?\b",
        r"\b(\d{1,3})\s*(?:ದಿನ|ದಿನಗಳು|ದಿನಗಳ|ದಿನದ)\b",
    ]

    for pattern in age_patterns:
        match = re.search(pattern, text)
        if match:
            age = int(match.group(1))
            if 1 <= age <= 3650:
                crop_age_days = age
                break

    # =====================================================
    # CROP DETECTION
    # =====================================================

    crop = None

    crop_keywords = {

        "tomato": [
            "tomato",
            "ಟೊಮ್ಯಾಟೊ",
            "ಟೊಮೇಟೊ",
            "ಟೊಮಾಟೊ",
            "ಟೊಮೆಟೊ"
        ],

        "paddy": [
            "paddy",
            "rice",
            "ಭತ್ತ",
            "ಭತ್ತದ ಬೆಳೆ",
            "ಅಕ್ಕಿ"
        ],

        "coconut": [
            "coconut",
            "ತೆಂಗು",
            "ತೆಂಗಿನ",
            "ತೆಂಗಿನ ಮರ",
            "ತೆಂಗಿನ ಬೆಳೆ"
        ],

        "banana": [
            "banana",
            "ಬಾಳೆ",
            "ಬಾಳೆಹಣ್ಣು",
            "ಬಾಳೆ ಬೆಳೆ"
        ],

        "chilli": [
            "chilli",
            "chili",
            "pepper chilli",
            "ಮೆಣಸಿನಕಾಯಿ",
            "ಮೆಣಸಿನ",
            "ಮೆಣಸಿನ ಬೆಳೆ"
        ],

        "potato": [
            "potato",
            "ಆಲೂಗಡ್ಡೆ",
            "ಆಲೂಗೆಡ್ಡೆ"
        ],

        "onion": [
            "onion",
            "ಈರುಳ್ಳಿ",
            "ಈರುಳ್ಳಿ ಬೆಳೆ"
        ],

        "maize": [
            "maize",
            "corn",
            "ಮೆಕ್ಕೆಜೋಳ",
            "ಜೋಳ"
        ],

        "groundnut": [
            "groundnut",
            "peanut",
            "ಕಡಲೆಕಾಯಿ",
            "ಶೇಂಗಾ",
            "ಶೇಂಗಾ ಬೆಳೆ"
        ],

        "sugarcane": [
            "sugarcane",
            "ಕಬ್ಬು",
            "ಕಬ್ಬಿನ ಬೆಳೆ"
        ],

        "cotton": [
            "cotton",
            "ಹತ್ತಿ",
            "ಹತ್ತಿ ಬೆಳೆ"
        ],

        "arecanut": [
            "arecanut",
            "areca",
            "ಅಡಿಕೆ",
            "ಅಡಿಕೆ ಮರ",
            "ಅಡಿಕೆ ಬೆಳೆ"
        ]
    }

    for crop_name, keywords in crop_keywords.items():

        if any(
            keyword in text
            for keyword in keywords
        ):

            crop = crop_name
            break

    # =====================================================
    # PROBLEM DETECTION
    # =====================================================

    detected_problem = None

    
    # -----------------------------------------------------
    # ADDITIONAL SYMPTOM DETECTION
    # -----------------------------------------------------
    symptom_keywords = {
        "waterlogging": [
            "soil is very wet",
            "soil is wet",
            "waterlogged",
            "water is standing",
            "too much water",
            "ಮಣ್ಣಿನಲ್ಲಿ ನೀರು ನಿಂತಿದೆ",
            "ಮಣ್ಣು ತುಂಬಾ ಒದ್ದೆಯಾಗಿದೆ",
            "ನೀರು ಹೆಚ್ಚಾಗಿದೆ",
        ],
        "dry_soil": [
            "soil is dry",
            "very dry soil",
            "soil is too dry",
            "ಮಣ್ಣು ಒಣಗಿದೆ",
            "ಮಣ್ಣು ತುಂಬಾ ಒಣಗಿದೆ",
            "ತೇವಾಂಶ ಕಡಿಮೆಯಿದೆ",
        ],
        "leaf_spots": [
            "brown spots",
            "black spots",
            "spots on leaves",
            "spots on the leaves",
            "leaf spots",
            "ಎಲೆಗಳ ಮೇಲೆ ಕಲೆ",
            "ಎಲೆಗಳಲ್ಲಿ ಕಲೆ",
            "ಕಂದು ಕಲೆಗಳು",
            "ಕಪ್ಪು ಕಲೆಗಳು",
        ],
        "drying_leaves": [
            "leaves are drying",
            "leaves drying",
            "leaf is drying",
            "leaves are drying up",
            "ಎಲೆಗಳು ಒಣಗುತ್ತಿವೆ",
            "ಎಲೆ ಒಣಗುತ್ತಿದೆ",
        ],
    }

    detected_symptom = None

    for symptom_name, keywords in symptom_keywords.items():
        if any(keyword in text for keyword in keywords):
            detected_symptom = symptom_name
            break

    # -----------------------------------------------------
    # YELLOW LEAF
    # -----------------------------------------------------

    yellow_leaf_keywords = [
        # English variations
        "yellow leaf",
        "yellow leaves",
        "leaf yellow",
        "leaves yellow",
        "leaves are turning yellow",
        "leaves turning yellow",
        "leaf is turning yellow",
        "leaf turning yellow",
        "leaves have turned yellow",
        "leaf has turned yellow",
        "leaves are yellow",
        "leaf is yellow",
        "turning yellow",
        "turned yellow",
        "yellowing",
        "yellow colour leaves",

        # Kannada variations
        "ಎಲೆ ಹಳದಿ",
        "ಎಲೆಗಳು ಹಳದಿ",
        "ಎಲೆ ಹಳದಿಯಾಗಿದೆ",
        "ಎಲೆ ಹಳದಿಯಾಗಿವೆ",
        "ಎಲೆ ಹಳದಿಯಾಗುತ್ತಿದೆ",
        "ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ",
        "ಎಲೆ ಹಳದಿಯಾಗುತ್ತಿವೆ",
        "ಎಲೆ ಹಳದಿ ಆಗಿದೆ",
        "ಎಲೆಗಳು ಹಳದಿ ಆಗಿವೆ",
        "ಎಲೆ ಹಳದಿ ಆಗುತ್ತಿದೆ",
        "ಎಲೆಗಳು ಹಳದಿ ಆಗುತ್ತಿವೆ",
    ]

    if any(keyword in text for keyword in yellow_leaf_keywords):
        detected_problem = "yellow_leaf"
    # -----------------------------------------------------
    # PEST
    # -----------------------------------------------------

    elif any(
        keyword in text
        for keyword in [
            "pest",
            "insect",
            "worm",
            "bug",
            "ಕೀಟ",
            "ಹುಳು",
            "ಹುಳ",
            "ಕೀಟ ಬಂದಿದೆ",
            "ಹುಳು ಬಂದಿದೆ"
        ]
    ):

        detected_problem = "pest"

    # -----------------------------------------------------
    # DISEASE
    # -----------------------------------------------------

    elif any(
        keyword in text
        for keyword in [
            "disease",
            "plant disease",
            "crop disease",
            "ರೋಗ",
            "ಬೆಳೆ ರೋಗ",
            "ಸಸ್ಯ ರೋಗ"
        ]
    ):

        detected_problem = "disease"

    # -----------------------------------------------------
    # WATER
    # -----------------------------------------------------

    elif any(
        keyword in text
        for keyword in [
            "water problem",
            "no water",
            "water shortage",
            "irrigation",
            "watering",
            "ನೀರಿನ ಸಮಸ್ಯೆ",
            "ನೀರು ಇಲ್ಲ",
            "ನೀರಿನ ಕೊರತೆ",
            "ನೀರಾವರಿ",
            "ನೀರು ಹಾಕಬೇಕು",
            "ಬೆಳೆಗೆ ನೀರು"
        ]
    ):

        detected_problem = "water"

    # -----------------------------------------------------
    # FERTILIZER
    # -----------------------------------------------------

    elif any(
        keyword in text
        for keyword in [
            "fertilizer",
            "fertiliser",
            "urea",
            "manure",
            "ಗೊಬ್ಬರ",
            "ಯೂರಿಯಾ",
            "ರಸಗೊಬ್ಬರ",
            "ಸಾವಯವ ಗೊಬ್ಬರ"
        ]
    ):

        detected_problem = "fertilizer"

    # -----------------------------------------------------
    # GROWTH
    # -----------------------------------------------------

    elif any(
        keyword in text
        for keyword in [
            "growth",
            "not growing",
            "slow growth",
            "ಬೆಳವಣಿಗೆ",
            "ಬೆಳೆಯುತ್ತಿಲ್ಲ",
            "ಬೆಳವಣಿಗೆ ಇಲ್ಲ",
            "ಬೆಳೆ ಬೆಳೆಯುತ್ತಿಲ್ಲ"
        ]
    ):

        detected_problem = "growth"

    # -----------------------------------------------------
    # HARVEST
    # -----------------------------------------------------

    elif any(
        keyword in text
        for keyword in [
            "harvest",
            "harvesting",
            "ಕೊಯ್ಲು",
            "ಬೆಳೆ ಕೊಯ್ಲು",
            "ಕೊಯ್ಲು ಯಾವಾಗ"
        ]
    ):

        detected_problem = "harvest"

    # -----------------------------------------------------
    # SOIL
    # -----------------------------------------------------

    elif any(
        keyword in text
        for keyword in [
            "soil",
            "soil test",
            "soil health",
            "ಮಣ್ಣು",
            "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ",
            "ಮಣ್ಣಿನ ಆರೋಗ್ಯ"
        ]
    ):

        detected_problem = "soil"

    # -----------------------------------------------------
    # GENERAL CROP PROBLEM
    # -----------------------------------------------------

    elif crop is not None:

        detected_problem = "general_crop_problem"

    # =====================================================
    # INTENT DETECTION
    # =====================================================

    if detected_problem in [
        "pest",
        "yellow_leaf",
        "disease",
        "water",
        "fertilizer",
        "growth",
        "soil",
        "general_crop_problem"
    ]:

        intent = "crop_problem"

    elif detected_problem == "harvest":

        intent = "harvest"

    elif detected_problem is not None:

        intent = "farmer_question"

    elif crop is not None:

        intent = "crop_question"

    else:

        intent = "general_question"

    # =====================================================
    # RETURN SMART CONTEXT
    # =====================================================
    return {
        "crop": crop,
        "problem": detected_problem,
        "intent": intent,
        "crop_age_days": crop_age_days,
        "symptom": detected_symptom
    }
# =========================================================
# FARMER RESPONSE
# =========================================================
# =========================================================
# SMART FARMER RESPONSE
# =========================================================

def generate_general_farmer_qa(problem: str):
    text = problem.lower().strip()

    # =====================================================
    # TOMATO
    # =====================================================
    if any(k in text for k in [
        "tomato", "tomato crop",
        "ಟೊಮ್ಯಾಟೊ", "ಟೊಮೇಟೊ", "ಟೊಮಾಟೊ"
    ]):
        return {
            "answer": (
                "ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಟೊಮ್ಯಾಟೊ ಬೆಳೆಗೆ ಸಂಬಂಧಿಸಿದೆ ಎಂದು ಅರ್ಥವಾಗಿದೆ. "
                "ಸರಿಯಾದ ಸಲಹೆಗೆ ಬೆಳೆಯ stage, ಮಣ್ಣಿನ ಸ್ಥಿತಿ ಮತ್ತು ಸಮಸ್ಯೆಯ ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಬೇಕು."
            ),
            "main_advice": [
                "ಬೆಳೆಯ stage ಅನ್ನು ಗುರುತಿಸಿ.",
                "ಮಣ್ಣಿನಲ್ಲಿ ಸಾಕಷ್ಟು ತೇವಾಂಶ ಇದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "ಎಲೆ, ಹೂ ಮತ್ತು ಹಣ್ಣಿನಲ್ಲಿ ಯಾವುದೇ ಕಲೆ, ಕೀಟ ಅಥವಾ ಬಣ್ಣ ಬದಲಾವಣೆ ಇದೆಯೇ ಗಮನಿಸಿ.",
                "ಅಗತ್ಯವಿದ್ದರೆ soil test ಮಾಡಿಸಿ."
            ],
            "do": [
                "ಬೆಳೆಯ current condition ಗಮನಿಸಿ.",
                "ಸಮಸ್ಯೆಯ clear photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ನೀರಾವರಿ ಮತ್ತು ಗೊಬ್ಬರದ ಬಳಕೆಯನ್ನು ದಾಖಲಿಸಿಕೊಳ್ಳಿ."
            ],
            "dont": [
                "ಸಮಸ್ಯೆ ಗುರುತಿಸದೆ pesticide ಅಥವಾ fertilizer ಬಳಸಬೇಡಿ.",
                "ಅಗತ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚು ನೀರು ಕೊಡಬೇಡಿ."
            ],
            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಟೊಮ್ಯಾಟೊ ಬೆಳೆ ಎಷ್ಟು ದಿನ/ಯಾವ stage ನಲ್ಲಿ ಇದೆ "
                "ಮತ್ತು ನಿಮಗೆ ಯಾವ ಸಮಸ್ಯೆ ಇದೆ ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # RICE / PADDY
    # =====================================================
    if any(k in text for k in [
        "rice", "paddy",
        "ಅಕ್ಕಿ", "ಭತ್ತ", "ಭತ್ತದ ಬೆಳೆ"
    ]):
        return {
            "answer": (
                "ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಭತ್ತದ ಬೆಳೆಗೆ ಸಂಬಂಧಿಸಿದೆ ಎಂದು ಅರ್ಥವಾಗಿದೆ. "
                "ಭತ್ತದ ನಿರ್ವಹಣೆ ಬೆಳೆಯ stage, ನೀರಿನ ಲಭ್ಯತೆ ಮತ್ತು ಮಣ್ಣಿನ ಸ್ಥಿತಿಯ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿರುತ್ತದೆ."
            ),
            "main_advice": [
                "ಬೆಳೆಯ stage ಅನ್ನು ಗಮನಿಸಿ.",
                "ಹೊಲದಲ್ಲಿ ನೀರಿನ ಮಟ್ಟವನ್ನು ನಿಯಂತ್ರಿಸಿ.",
                "ಮಣ್ಣಿನ ಪೋಷಕಾಂಶ ಸ್ಥಿತಿಯನ್ನು ಪರಿಗಣಿಸಿ.",
                "ಕೀಟ ಅಥವಾ ರೋಗದ ಲಕ್ಷಣಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ."
            ],
            "do": [
                "ನೀರಾವರಿ ಸ್ಥಿತಿಯನ್ನು ಗಮನಿಸಿ.",
                "ಬೆಳೆಯ ಬೆಳವಣಿಗೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "ಅಗತ್ಯವಿದ್ದರೆ soil test ಮಾಡಿಸಿ."
            ],
            "dont": [
                "ಅಗತ್ಯವಿಲ್ಲದೆ ಹೆಚ್ಚಿನ fertilizer ಬಳಸಬೇಡಿ.",
                "ಕೀಟ ಅಥವಾ ರೋಗ ಗುರುತಿಸದೆ chemical ಬಳಸಬೇಡಿ."
            ],
            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಭತ್ತದ ಬೆಳೆಯ ಯಾವ stage ನಲ್ಲಿ ಇದ್ದೀರಿ "
                "ಮತ್ತು ನಿಮಗೆ ಯಾವ ಸಮಸ್ಯೆ ಇದೆ ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # CROP PLANNING
    # =====================================================
    if any(k in text for k in [
        "which crop", "what crop", "crop suggestion",
        "best crop", "crop to grow",
        "ಯಾವ ಬೆಳೆ", "ಯಾವ ಬೆಳೆ ಬೆಳೆಯಬೇಕು",
        "ಯಾವ ಬೆಳೆ ಬೆಳೆಯಲಿ", "ಬೆಳೆ ಆಯ್ಕೆ"
    ]):
        return {
            "answer": (
                "ಯಾವ ಬೆಳೆ ಬೆಳೆಸಬೇಕು ಎಂಬುದನ್ನು ನಿಮ್ಮ ಮಣ್ಣು, ನೀರಿನ ಲಭ್ಯತೆ, "
                "ಹವಾಮಾನ ಮತ್ತು season ಆಧರಿಸಿ ನಿರ್ಧರಿಸುವುದು ಉತ್ತಮ."
            ),
            "main_advice": [
                "ನಿಮ್ಮ ಮಣ್ಣಿನ ಪ್ರಕಾರವನ್ನು ತಿಳಿದುಕೊಳ್ಳಿ.",
                "ನೀರಿನ ಲಭ್ಯತೆಯನ್ನು ಪರಿಗಣಿಸಿ.",
                "ಪ್ರಸ್ತುತ season ಮತ್ತು ಸ್ಥಳೀಯ ಹವಾಮಾನವನ್ನು ಪರಿಗಣಿಸಿ.",
                "ಮಾರುಕಟ್ಟೆ ಬೇಡಿಕೆಯನ್ನೂ ಗಮನಿಸಿ."
            ],
            "do": [
                "Soil test ಮಾಡಿಸಿ.",
                "ನಿಮ್ಮ ಪ್ರದೇಶ ಮತ್ತು season ತಿಳಿಸಿ.",
                "ನೀರಿನ ಲಭ್ಯತೆ ಎಷ್ಟಿದೆ ಎಂದು ತಿಳಿಸಿ."
            ],
            "dont": [
                "ಬೇರೆ ರೈತರ ಬೆಳೆ ನೋಡಿ ಮಾತ್ರ crop ಆಯ್ಕೆ ಮಾಡಬೇಡಿ.",
                "ನೀರಿನ ಲಭ್ಯತೆ ಇಲ್ಲದೆ ಹೆಚ್ಚು ನೀರು ಬೇಕಾಗುವ crop ಆಯ್ಕೆ ಮಾಡಬೇಡಿ."
            ],
            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ನಿಮ್ಮ ಪ್ರದೇಶ, ಮಣ್ಣಿನ ಪ್ರಕಾರ, "
                "ನೀರಿನ ಲಭ್ಯತೆ ಮತ್ತು season ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # SOIL
    # =====================================================
    if any(k in text for k in [
        "soil", "soil test", "soil health",
        "ಮಣ್ಣು", "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ", "ಮಣ್ಣಿನ ಆರೋಗ್ಯ"
    ]):
        return {
            "answer": (
                "ಮಣ್ಣಿನ ಗುಣಮಟ್ಟ ಮತ್ತು ಪೋಷಕಾಂಶಗಳನ್ನು ತಿಳಿದುಕೊಳ್ಳಲು soil testing "
                "ಬಹಳ ಉಪಯುಕ್ತವಾಗಿದೆ."
            ),
            "main_advice": [
                "ಮಣ್ಣಿನ sample ಸರಿಯಾದ ರೀತಿಯಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಿ.",
                "Soil test report ನಲ್ಲಿ pH ಮತ್ತು nutrient levels ಗಮನಿಸಿ.",
                "Report ಆಧರಿಸಿ fertilizer planning ಮಾಡಿ."
            ],
            "do": [
                "ಸಾಧ್ಯವಾದರೆ ಅಧಿಕೃತ soil testing laboratory ಬಳಸಿರಿ.",
                "Soil report ಅನ್ನು ಉಳಿಸಿಕೊಳ್ಳಿ.",
                "ಬೆಳೆ ಆಯ್ಕೆಯಲ್ಲಿ soil condition ಪರಿಗಣಿಸಿ."
            ],
            "dont": [
                "Soil test ಇಲ್ಲದೆ ಹೆಚ್ಚಿನ ಪ್ರಮಾಣದ fertilizer ಬಳಸಬೇಡಿ."
            ],
            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ನಿಮ್ಮ soil report ಇದ್ದರೆ ಅದರ ವಿವರಗಳನ್ನು ತಿಳಿಸಿ "
                "ಅಥವಾ ನಿಮ್ಮ ಮಣ್ಣಿನ ಸಮಸ್ಯೆಯನ್ನು ವಿವರಿಸಿ."
            )
        }

    # =====================================================
    # IRRIGATION
    # =====================================================
    if any(k in text for k in [
        "irrigation", "irrigate", "watering",
        "ನೀರಾವರಿ", "ನೀರು ಹಾಕಬೇಕು", "ನೀರು ಎಷ್ಟು",
        "ನೀರು ಯಾವಾಗ", "ಬೆಳೆಗೆ ನೀರು"
    ]):
        return {
            "answer": (
                "ಬೆಳೆಗೆ ನೀರು ಕೊಡುವ ಪ್ರಮಾಣ ಮತ್ತು ಸಮಯವು crop, soil ಮತ್ತು weather "
                "ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿರುತ್ತದೆ."
            ),
            "main_advice": [
                "ಮಣ್ಣಿನ moisture ಪರಿಶೀಲಿಸಿ.",
                "ಬೆಳೆಯ stage ಗಮನಿಸಿ.",
                "ಹವಾಮಾನ ಮತ್ತು ಮಳೆಯ ಸಾಧ್ಯತೆಯನ್ನು ಪರಿಗಣಿಸಿ.",
                "ನೀರು ನಿಲ್ಲುವ ಪರಿಸ್ಥಿತಿ ತಪ್ಪಿಸಿ."
            ],
            "do": [
                "ಮಣ್ಣಿನ ತೇವಾಂಶವನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "ನೀರಾವರಿ system leakage ಇದೆಯೇ ಪರಿಶೀಲಿಸಿ."
            ],
            "dont": [
                "ಅಗತ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚು ನೀರು ಕೊಡಬೇಡಿ.",
                "ಮಳೆ ಬರುತ್ತಿರುವಾಗ ಅಗತ್ಯವಿಲ್ಲದೆ irrigation ಮಾಡಬೇಡಿ."
            ],
            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ crop ಮತ್ತು ಮಣ್ಣಿನ ಪ್ರಕಾರ ಯಾವುದು "
                "ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # HARVEST
    # =====================================================
    if any(k in text for k in [
        "harvest", "harvesting",
        "ಕೊಯ್ಲು", "ಕೊಯ್ಲು ಯಾವಾಗ", "ಬೆಳೆ ಕೊಯ್ಲು"
    ]):
        return {
            "answer": (
                "ಕೊಯ್ಲಿನ ಸರಿಯಾದ ಸಮಯ crop ಮತ್ತು ಅದರ maturity stage ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿರುತ್ತದೆ."
            ),
            "main_advice": [
                "ಬೆಳೆಯ maturity ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.",
                "ಹವಾಮಾನವನ್ನು ಪರಿಗಣಿಸಿ.",
                "Harvest ನಂತರ storage ಅಥವಾ transportation ವ್ಯವಸ್ಥೆ ಸಿದ್ಧಪಡಿಸಿ."
            ],
            "do": [
                "ಕೊಯ್ಲಿಗೆ ಮುಂಚಿತವಾಗಿ labour ಮತ್ತು machinery ವ್ಯವಸ್ಥೆ ಮಾಡಿ.",
                "ಮಳೆಯ ಸಾಧ್ಯತೆ ಇದ್ದರೆ ಯೋಜನೆಯನ್ನು ಮುಂಚಿತವಾಗಿ ಮಾಡಿ."
            ],
            "dont": [
                "ಅತಿಯಾಗಿ ತೇವವಾಗಿರುವ crop ಅನ್ನು storage ಗೆ ಹಾಕಬೇಡಿ."
            ],
            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ crop ಮತ್ತು ಅದರ ಪ್ರಸ್ತುತ stage ಅನ್ನು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # GENERAL FARM QUESTION
    # =====================================================
    return {
        "answer": (
            "ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಕೃಷಿಗೆ ಸಂಬಂಧಿಸಿದೆ ಎಂದು ಅರ್ಥವಾಗಿದೆ. "
            "ಇನ್ನಷ್ಟು ಸರಿಯಾದ ಸಲಹೆ ನೀಡಲು crop, ಸಮಸ್ಯೆ ಅಥವಾ ಕೆಲಸದ ಬಗ್ಗೆ "
            "ಸ್ವಲ್ಪ ಹೆಚ್ಚಿನ ಮಾಹಿತಿ ಬೇಕಾಗಿದೆ."
        ),
        "main_advice": [
            "ನಿಮ್ಮ crop ಅಥವಾ farming activity ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
            "ನಿಮ್ಮ ಸಮಸ್ಯೆಯ ಮುಖ್ಯ ಲಕ್ಷಣವನ್ನು ವಿವರಿಸಿ.",
            "ಸಾಧ್ಯವಾದರೆ ಸಮಸ್ಯೆಯ photo upload ಮಾಡಿ."
        ],
        "do": [
            "Question ಅನ್ನು ನಿಮ್ಮ ಸಾಮಾನ್ಯ ಮಾತನಾಡುವ ರೀತಿಯಲ್ಲಿ ಕೇಳಬಹುದು.",
            "Crop name ಮತ್ತು ಸಮಸ್ಯೆಯ ವಿವರವನ್ನು ಸೇರಿಸಿ."
        ],
        "dont": [
            "ಸಮಸ್ಯೆ ಗುರುತಿಸದೆ pesticide ಅಥವಾ medicine ಬಳಸಬೇಡಿ.",
            "Mechanical ಅಥವಾ electrical equipment ಅನ್ನು ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ repair ಮಾಡಬೇಡಿ."
        ],
        "next_step": (
            "ಉದಾಹರಣೆಗೆ: 'ಟೊಮ್ಯಾಟೊ ಬೆಳೆಗೆ ಏನು ಮಾಡಬೇಕು?', "
            "'ನನ್ನ ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಹೇಗೆ ಮಾಡುವುದು?', "
            "'ಯಾವ ಬೆಳೆ ಬೆಳೆಯಬಹುದು?' ಎಂದು ಕೇಳಬಹುದು."
        )
    }

def generate_smart_crop_response(smart_context, problem):
    """
    Smart Context ಆಧಾರದ ಮೇಲೆ crop-specific farmer response ನೀಡುತ್ತದೆ.
    Existing tractor/service responses ಅನ್ನು touch ಮಾಡುವುದಿಲ್ಲ.
    """

    crop = smart_context.get("crop")
    detected_problem = smart_context.get("problem")
    crop_age_days = smart_context.get("crop_age_days")
    detected_symptom = smart_context.get("symptom")

    if not crop or not detected_problem:
        return None

    crop_names = {
        "tomato": "ಟೊಮ್ಯಾಟೊ",
        "paddy": "ಭತ್ತ",
        "coconut": "ತೆಂಗು",
        "banana": "ಬಾಳೆ",
        "chilli": "ಮೆಣಸಿನಕಾಯಿ",
        "potato": "ಆಲೂಗಡ್ಡೆ",
        "onion": "ಈರುಳ್ಳಿ",
        "maize": "ಮೆಕ್ಕೆಜೋಳ",
        "groundnut": "ಕಡಲೆಕಾಯಿ",
        "sugarcane": "ಕಬ್ಬು",
        "cotton": "ಹತ್ತಿ",
        "arecanut": "ಅಡಿಕೆ",
    }

    crop_name = crop_names.get(crop, crop)

    # ---------------------------------------------------------
    # 1. YELLOW LEAF - CROP AGE AWARE RESPONSE
    # ---------------------------------------------------------
    if detected_problem == "yellow_leaf":

     
        age_advice = []

        if crop_age_days is not None:
            age_advice.append(
                f"ನೀವು ತಿಳಿಸಿದಂತೆ ನಿಮ್ಮ {crop_name} ಬೆಳೆ "
                f"{crop_age_days} ದಿನಗಳಾಗಿದೆ. ಈ ಮಾಹಿತಿಯನ್ನು "
                "ಬೆಳೆಯ ಹಂತವನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಬಳಸಬಹುದು."
            )

        symptom_advice = {
            "waterlogging": (
                "ಮಣ್ಣಿನಲ್ಲಿ ನೀರು ನಿಂತಿದೆಯೇ ಪರಿಶೀಲಿಸಿ. "
                "ನೀರಿನ ಹೊರಹರಿವು ಸರಿಯಾಗಿದೆಯೇ ಗಮನಿಸಿ."
            ),
            "dry_soil": (
                "ಮಣ್ಣಿನ ತೇವಾಂಶ ಪರಿಶೀಲಿಸಿ. "
                "ಬೆಳೆ ಮತ್ತು ಮಣ್ಣಿನ ಸ್ಥಿತಿಗೆ ತಕ್ಕಂತೆ ನೀರಾವರಿ ನೀಡಿ."
            ),
            "leaf_spots": (
                "ಎಲೆಗಳ ಮೇಲಿನ ಕಲೆಗಳ ಎರಡೂ ಬದಿಗಳ ಸ್ಪಷ್ಟ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ. "
                "ಕಲೆಗಳ ಆಕಾರ ಮತ್ತು ಹರಡುವಿಕೆಯನ್ನು ಗಮನಿಸಿ."
            ),
            "drying_leaves": (
                "ಎಲೆಗಳು ಎಷ್ಟು ಪ್ರಮಾಣದಲ್ಲಿ ಒಣಗುತ್ತಿವೆ ಮತ್ತು "
                "ಮಣ್ಣಿನ ತೇವಾಂಶ ಹೇಗಿದೆ ಎಂದು ಪರಿಶೀಲಿಸಿ."
            )
        }

        extra_symptom_advice = symptom_advice.get(
            detected_symptom
        )

        return {
            "answer": (
                f"ನಿಮ್ಮ {crop_name} ಬೆಳೆಯ ಎಲೆಗಳು ಹಳದಿಯಾಗಿವೆ ಎಂದು ಅರ್ಥವಾಗಿದೆ. "
                + (
                    f"ನೀವು ತಿಳಿಸಿದಂತೆ ಬೆಳೆ {crop_age_days} ದಿನಗಳಾಗಿದೆ. "
                    if crop_age_days is not None else ""
                )
                + "ಇದಕ್ಕೆ ನೀರಿನ ಸಮಸ್ಯೆ, ಪೋಷಕಾಂಶದ ಕೊರತೆ, ಬೇರು ಸಮಸ್ಯೆ "
                "ಅಥವಾ ರೋಗ ಮುಂತಾದ ಹಲವು ಕಾರಣಗಳು ಇರಬಹುದು. "
                "ನಿಖರ ಕಾರಣ ತಿಳಿಯಲು ಹೆಚ್ಚಿನ ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಬೇಕು."
            ),

"main_advice": age_advice + (
    [extra_symptom_advice] if extra_symptom_advice else []
) + [                "ಮಣ್ಣಿನಲ್ಲಿ ನೀರು ಹೆಚ್ಚು ನಿಂತಿದೆಯೇ ಅಥವಾ ತೇವಾಂಶ ಕಡಿಮೆಯಿದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "ಎಲೆಗಳ ಕೆಳಭಾಗದಲ್ಲಿ ಕೀಟಗಳಿವೆಯೇ ಗಮನಿಸಿ.",
                "ಎಲೆಗಳಲ್ಲಿ ಕಲೆ, ಮಚ್ಚೆ ಅಥವಾ ಒಣಗುವ ಲಕ್ಷಣಗಳಿವೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "ಸಾಧ್ಯವಾದರೆ ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ ಪೋಷಕಾಂಶಗಳ ಸ್ಥಿತಿ ತಿಳಿದುಕೊಳ್ಳಿ."
            ],

            "do": [
                "ಹಳದಿಯಾಗಿರುವ ಎಲೆಗಳ ಸ್ಪಷ್ಟ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ಬೆಳೆಯ ವಯಸ್ಸು ಮತ್ತು ಯಾವ ಹಂತದಲ್ಲಿದೆ ಎಂಬುದನ್ನು ತಿಳಿಸಿ.",
                "ನೀರಾವರಿ ಮತ್ತು fertilizer ಬಳಕೆಯನ್ನು ಪರಿಶೀಲಿಸಿ."
            ],

            "dont": [
                "ಕಾರಣ ತಿಳಿಯದೆ pesticide ಅಥವಾ fertilizer ಬಳಸಬೇಡಿ.",
                "ಅಗತ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚು ನೀರು ಕೊಡಬೇಡಿ.",
                "ಒಂದೇ ಸಮಸ್ಯೆಗೆ ಹಲವು chemicals ಅನ್ನು mix ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                f"ಮುಂದಿನ ಹಂತ: {crop_name} ಬೆಳೆಯ ಹಳದಿ ಎಲೆಗಳ photo upload ಮಾಡಿ. "
                "ಎಲೆಗಳ ಮೇಲ್ಭಾಗ ಮತ್ತು ಕೆಳಭಾಗ ಎರಡೂ ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣುವಂತೆ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ."
            )
        }



    # ---------------------------------------------------------
    # 2. PEST
    # ---------------------------------------------------------
    if detected_problem == "pest":
        return {
            "answer": (
                f"ನಿಮ್ಮ {crop_name} ಬೆಳೆಯಲ್ಲಿ ಕೀಟದ ಸಮಸ್ಯೆ ಇರುವ ಸಾಧ್ಯತೆ ಇದೆ. "
                "ಕೀಟದ ಪ್ರಕಾರವನ್ನು ಮೊದಲು ಗುರುತಿಸುವುದು ಮುಖ್ಯ."
            ),

            "main_advice": [
                "ಎಲೆಗಳ ಮೇಲೆ ಮತ್ತು ಎಲೆಗಳ ಕೆಳಭಾಗದಲ್ಲಿ ಕೀಟಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "ಎಲೆಗಳಲ್ಲಿ ರಂಧ್ರ, curling ಅಥವಾ damage ಇದೆಯೇ ಗಮನಿಸಿ.",
                "ಕೀಟದ clear photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ಸಮಸ್ಯೆ ಹೆಚ್ಚು ಇದ್ದರೆ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
            ],

            "do": [
                "ಸಮಸ್ಯೆಯಿರುವ ಎಲೆ ಅಥವಾ ಕೀಟದ clear photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ಯಾವ ಭಾಗದಲ್ಲಿ ಹೆಚ್ಚು ಕೀಟ ಇದೆ ಎಂದು ಗಮನಿಸಿ.",
                "ಬೆಳೆಯ stage ಅನ್ನು ಗಮನಿಸಿ."
            ],

            "dont": [
                "ಕೀಟ ಗುರುತಿಸದೆ pesticide spray ಮಾಡಬೇಡಿ.",
                "ಶಿಫಾರಸ್ಸಿಗಿಂತ ಹೆಚ್ಚು chemical ಬಳಸಬೇಡಿ."
            ],

            "next_step": (
                f"ಮುಂದಿನ ಹಂತ: {crop_name} ಬೆಳೆಯಲ್ಲಿರುವ ಕೀಟದ photo upload ಮಾಡಿ. "
                "KrushiVaani ಅದನ್ನು analyse ಮಾಡಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ."
            )
        }

    # ---------------------------------------------------------
    # 3. DISEASE
    # ---------------------------------------------------------
    if detected_problem == "disease":
        return {
            "answer": (
                f"ನಿಮ್ಮ {crop_name} ಬೆಳೆಗೆ ರೋಗದ ಸಮಸ್ಯೆ ಇರಬಹುದು. "
                "ರೋಗವನ್ನು ಸರಿಯಾಗಿ ಗುರುತಿಸಲು symptoms ಮತ್ತು photo ಬಹಳ ಮುಖ್ಯ."
            ),

            "main_advice": [
                "ಎಲೆ, ಕಾಂಡ ಅಥವಾ ಹಣ್ಣಿನಲ್ಲಿ ಇರುವ ಕಲೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "ಎಲೆಗಳ ಬಣ್ಣ ಬದಲಾವಣೆ ಅಥವಾ ಒಣಗುವ ಲಕ್ಷಣ ಗಮನಿಸಿ.",
                "ರೋಗದ clear photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ಅಗತ್ಯವಿದ್ದರೆ soil ಮತ್ತು crop condition ಪರಿಶೀಲಿಸಿ."
            ],

            "do": [
                "ರೋಗಬಾಧಿತ ಭಾಗದ clear photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ರೋಗ ಯಾವಾಗ ಪ್ರಾರಂಭವಾಯಿತು ಎಂದು ಗಮನಿಸಿ.",
                "ಸಮಸ್ಯೆ ಹರಡುತ್ತಿದೆಯೇ ಪರಿಶೀಲಿಸಿ."
            ],

            "dont": [
                "ರೋಗದ ಹೆಸರು ತಿಳಿಯದೆ chemical ಬಳಸಬೇಡಿ.",
                "ಅಗತ್ಯವಿಲ್ಲದೆ ಹಲವು medicines mix ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                f"ಮುಂದಿನ ಹಂತ: {crop_name} ಬೆಳೆಯ ರೋಗಬಾಧಿತ ಭಾಗದ photo upload ಮಾಡಿ "
                "ಮತ್ತು ಸಮಸ್ಯೆ ಎಷ್ಟು ದಿನಗಳಿಂದ ಇದೆ ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # ---------------------------------------------------------
    # 4. WATER / IRRIGATION
    # ---------------------------------------------------------
    if detected_problem == "water":
        return {
            "answer": (
                f"{crop_name} ಬೆಳೆಗೆ ನೀರಾವರಿ crop stage, soil ಮತ್ತು weather "
                "ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿರುತ್ತದೆ."
            ),

            "main_advice": [
                "ಮಣ್ಣಿನ ತೇವಾಂಶವನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "ಬೆಳೆಯ current stage ಗಮನಿಸಿ.",
                "ಮಳೆಯ ಸಾಧ್ಯತೆಯನ್ನು ಪರಿಗಣಿಸಿ.",
                "ಹೊಲದಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ drainage ಪರಿಶೀಲಿಸಿ."
            ],

            "do": [
                "ಮಣ್ಣಿನ moisture ಅನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "Irrigation system ನಲ್ಲಿ leakage ಇದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "Weather forecast ಗಮನಿಸಿ."
            ],

            "dont": [
                "ಅಗತ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚು ನೀರು ಕೊಡಬೇಡಿ.",
                "ಹೊಲದಲ್ಲಿ ನೀರು ನಿಲ್ಲಲು ಬಿಡಬೇಡಿ."
            ],

            "next_step": (
                f"ಮುಂದಿನ ಹಂತ: {crop_name} ಬೆಳೆಯ ವಯಸ್ಸು, "
                "ಮಣ್ಣಿನ ಪ್ರಕಾರ ಮತ್ತು ನೀರಿನ ಲಭ್ಯತೆ ತಿಳಿಸಿ."
            )
        }

    # ---------------------------------------------------------
    # 5. FERTILIZER
    # ---------------------------------------------------------
    if detected_problem == "fertilizer":
        return {
            "answer": (
                f"{crop_name} ಬೆಳೆಗೆ fertilizer ಅಗತ್ಯವು soil condition ಮತ್ತು "
                "crop stage ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿರುತ್ತದೆ."
            ),

            "main_advice": [
                "ಮೊದಲು soil condition ಪರಿಶೀಲಿಸಿ.",
                "ಸಾಧ್ಯವಾದರೆ soil test ಮಾಡಿಸಿ.",
                "ಬೆಳೆಯ stage ಅನ್ನು ಪರಿಗಣಿಸಿ.",
                "Fertilizer ಅನ್ನು soil test ಅಥವಾ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆಯಂತೆ ಬಳಸಿ."
            ],

            "do": [
                "ಹಿಂದೆ ಬಳಸಿದ fertilizer ವಿವರಗಳನ್ನು ದಾಖಲಿಸಿಕೊಳ್ಳಿ.",
                "Soil test report ಇದ್ದರೆ ಅದನ್ನು ಉಳಿಸಿಕೊಳ್ಳಿ.",
                "ಬೆಳೆಯ ಬೆಳವಣಿಗೆಯನ್ನು ಗಮನಿಸಿ."
            ],

            "dont": [
                "ಅಂದಾಜಿನ ಮೇಲೆ ಹೆಚ್ಚಿನ fertilizer ಬಳಸಬೇಡಿ.",
                "ಎಲ್ಲಾ fertilizer ಗಳನ್ನು ನಿಮ್ಮಷ್ಟಕ್ಕೆ mix ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                f"ಮುಂದಿನ ಹಂತ: {crop_name} ಬೆಳೆಯ age ಮತ್ತು "
                "ನಿಮ್ಮ soil test report ಇದ್ದರೆ ಅದರ ವಿವರಗಳನ್ನು ತಿಳಿಸಿ."
            )
        }

    # ---------------------------------------------------------
    # 6. GROWTH PROBLEM
    # ---------------------------------------------------------
    if detected_problem == "growth":
        return {
            "answer": (
                f"{crop_name} ಬೆಳವಣಿಗೆ ಸರಿಯಾಗಿ ಆಗುತ್ತಿಲ್ಲ ಎಂದು ಅರ್ಥವಾಗಿದೆ. "
                "ಇದಕ್ಕೆ soil, water, nutrients, pests ಅಥವಾ disease ಕಾರಣವಾಗಿರಬಹುದು."
            ),

            "main_advice": [
                "ಮಣ್ಣಿನ ತೇವಾಂಶ ಪರಿಶೀಲಿಸಿ.",
                "ಬೆಳೆಗೆ ಸಾಕಷ್ಟು ಬೆಳಕು ಸಿಗುತ್ತಿದೆಯೇ ಗಮನಿಸಿ.",
                "ಕೀಟ ಅಥವಾ ರೋಗದ ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "Soil nutrient condition ತಿಳಿದುಕೊಳ್ಳಲು soil test ಪರಿಗಣಿಸಿ."
            ],

            "do": [
                "ಬೆಳೆಯ ಬೆಳವಣಿಗೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಗಮನಿಸಿ.",
                "ಸಮಸ್ಯೆಯಿರುವ ಭಾಗದ photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ನೀರಾವರಿ ಮತ್ತು fertilizer history ಪರಿಶೀಲಿಸಿ."
            ],

            "dont": [
                "ಕಾರಣ ತಿಳಿಯದೆ ಹೆಚ್ಚಿನ fertilizer ಬಳಸಬೇಡಿ.",
                "ಅಗತ್ಯವಿಲ್ಲದೆ pesticide ಬಳಸಬೇಡಿ."
            ],

            "next_step": (
                f"ಮುಂದಿನ ಹಂತ: {crop_name} ಬೆಳೆಯ photo, "
                "ಬೆಳೆಯ age ಮತ್ತು ಸಮಸ್ಯೆ ಯಾವಾಗ ಪ್ರಾರಂಭವಾಯಿತು ಎಂಬುದನ್ನು ತಿಳಿಸಿ."
            )
        }

    # ---------------------------------------------------------
    # 7. HARVEST
    # ---------------------------------------------------------
    if detected_problem == "harvest":
        return {
            "answer": (
                f"{crop_name} ಬೆಳೆಯ ಕೊಯ್ಲಿನ ಸರಿಯಾದ ಸಮಯ maturity stage "
                "ಮತ್ತು ಹವಾಮಾನದ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿರುತ್ತದೆ."
            ),

            "main_advice": [
                "ಬೆಳೆಯ maturity ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.",
                "ಹವಾಮಾನ ಮತ್ತು ಮಳೆಯ ಸಾಧ್ಯತೆಯನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "ಕೊಯ್ಲಿಗೆ ಮುಂಚಿತವಾಗಿ labour ಅಥವಾ machinery ವ್ಯವಸ್ಥೆ ಮಾಡಿ.",
                "Harvest ನಂತರ storage ಅಥವಾ transportation ವ್ಯವಸ್ಥೆ ಸಿದ್ಧಪಡಿಸಿ."
            ],

            "do": [
                "ಕೊಯ್ಲಿಗೆ ಬೇಕಾದ labour ಮತ್ತು machinery ಮುಂಚಿತವಾಗಿ arrange ಮಾಡಿ.",
                "Weather forecast ಪರಿಶೀಲಿಸಿ."
            ],

            "dont": [
                "ಅತಿಯಾಗಿ ತೇವವಾಗಿರುವ crop ಅನ್ನು storage ಗೆ ಹಾಕಬೇಡಿ."
            ],

            "next_step": (
                f"ಮುಂದಿನ ಹಂತ: {crop_name} ಬೆಳೆಯ current stage "
                "ಮತ್ತು ಬೆಳೆ ಎಷ್ಟು ದಿನಗಳಾಗಿದೆ ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # ---------------------------------------------------------
    # 8. SOIL
    # ---------------------------------------------------------
    if detected_problem == "soil":
        return {
            "answer": (
                f"{crop_name} ಬೆಳೆಗೆ ಉತ್ತಮ ನಿರ್ವಹಣೆಗೆ ಮಣ್ಣಿನ ಸ್ಥಿತಿ ತಿಳಿದುಕೊಳ್ಳುವುದು "
                "ಬಹಳ ಮುಖ್ಯ."
            ),

            "main_advice": [
                "Soil test ಮಾಡಿಸಿ.",
                "ಮಣ್ಣಿನ pH ಮತ್ತು nutrient levels ಪರಿಶೀಲಿಸಿ.",
                "Soil report ಆಧರಿಸಿ fertilizer planning ಮಾಡಿ."
            ],

            "do": [
                "ಸರಿಯಾದ ರೀತಿಯಲ್ಲಿ soil sample ತೆಗೆದುಕೊಳ್ಳಿ.",
                "Soil test report ಅನ್ನು ಉಳಿಸಿಕೊಳ್ಳಿ."
            ],

            "dont": [
                "Soil test ಇಲ್ಲದೆ ಹೆಚ್ಚಿನ fertilizer ಬಳಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ನಿಮ್ಮ soil test report ಇದ್ದರೆ ಅದರ ವಿವರಗಳನ್ನು "
                "KrushiVaani ಗೆ ತಿಳಿಸಿ."
            )
        }

    # ---------------------------------------------------------
    # 9. GENERAL CROP QUESTION
    # ---------------------------------------------------------
    if detected_problem == "general_crop_problem":
        return {
            "answer": (
                f"ನಿಮ್ಮ ಪ್ರಶ್ನೆ {crop_name} ಬೆಳೆಗೆ ಸಂಬಂಧಿಸಿದೆ ಎಂದು ಅರ್ಥವಾಗಿದೆ. "
                "ಸರಿಯಾದ ಸಲಹೆ ನೀಡಲು crop stage ಮತ್ತು ನಿಮ್ಮ ಮುಖ್ಯ ಸಮಸ್ಯೆ ತಿಳಿದುಕೊಳ್ಳುವುದು ಉತ್ತಮ."
            ),

            "main_advice": [
                f"{crop_name} ಬೆಳೆಯ current stage ಗಮನಿಸಿ.",
                "ಮಣ್ಣಿನ ತೇವಾಂಶ ಪರಿಶೀಲಿಸಿ.",
                "ಕೀಟ ಅಥವಾ ರೋಗದ ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
                "ಅಗತ್ಯವಿದ್ದರೆ soil test ಮಾಡಿಸಿ."
            ],

            "do": [
                "ಬೆಳೆಯ clear photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ಬೆಳೆ ಎಷ್ಟು ದಿನಗಳಾಗಿದೆ ಎಂದು ಗಮನಿಸಿ.",
                "ನೀರಾವರಿ ಮತ್ತು fertilizer ಬಳಕೆಯನ್ನು ದಾಖಲಿಸಿಕೊಳ್ಳಿ."
            ],

            "dont": [
                "ಸಮಸ್ಯೆ ಗುರುತಿಸದೆ pesticide ಅಥವಾ fertilizer ಬಳಸಬೇಡಿ."
            ],

            "next_step": (
                f"ಮುಂದಿನ ಹಂತ: {crop_name} ಬೆಳೆಯ ಯಾವ ಸಮಸ್ಯೆ ಇದೆ "
                "ಎಂದು ನಿಮ್ಮ ಸಾಮಾನ್ಯ ಮಾತಿನಲ್ಲಿ ಕೇಳಿ."
            )
        }

    return None

def generate_farmer_response(category, problem=None):
    # =====================================================
    # TRACTOR START PROBLEM
    # =====================================================

    if category == "tractor_start_problem":

        return {
            "answer": (
                "ನಿಮ್ಮ ಟ್ರ್ಯಾಕ್ಟರ್ start ಆಗುತ್ತಿಲ್ಲ ಎಂದು ಅರ್ಥವಾಗಿದೆ. "
                "ಮೊದಲು battery, fuel ಮತ್ತು starting system ಪರಿಶೀಲಿಸುವುದು ಮುಖ್ಯ."
            ),

            "main_advice": [
                "Battery charge ಮತ್ತು battery terminal connection ಪರಿಶೀಲಿಸಿ.",
                "Fuel ಸಾಕಷ್ಟು ಇದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
                "Key ಮತ್ತು ignition system ಸರಿಯಾಗಿ ಕೆಲಸ ಮಾಡುತ್ತಿದೆಯೇ ಗಮನಿಸಿ.",
                "Starter motor ಕೆಲಸ ಮಾಡುತ್ತಿದೆಯೇ ಗಮನಿಸಿ.",
                "Dashboard ನಲ್ಲಿ warning light ಬರುತ್ತಿದೆಯೇ ಪರಿಶೀಲಿಸಿ."
            ],

            "do": [
                "Battery connection ಅನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "Fuel level ಪರಿಶೀಲಿಸಿ.",
                "Problem ಯಾವಾಗ ಪ್ರಾರಂಭವಾಯಿತು ಎಂದು ಗಮನಿಸಿ.",
                "ಸಾಧ್ಯವಾದರೆ qualified tractor mechanic ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "dont": [
                "Electrical wiring ಅನ್ನು ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ ತೆರೆಯಬೇಡಿ.",
                "Starter ಅನ್ನು ನಿರಂತರವಾಗಿ operate ಮಾಡಬೇಡಿ.",
                "Engine problem ಇದ್ದಾಗ tractor ಅನ್ನು force ಮಾಡಿ start ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: tractor start ಮಾಡುವಾಗ ಏನಾಗುತ್ತದೆ ಎಂದು ಹೇಳಿ. "
                "ಉದಾಹರಣೆಗೆ: battery light ಬರುತ್ತದೆ, click sound ಬರುತ್ತದೆ, "
                "starter ತಿರುಗುತ್ತದೆ ಆದರೆ engine start ಆಗುವುದಿಲ್ಲ."
            )
        }

    # =====================================================
    # TRACTOR BATTERY PROBLEM
    # =====================================================

    elif category == "tractor_battery_problem":

        return {
            "answer": (
                "ನಿಮ್ಮ ಟ್ರ್ಯಾಕ್ಟರ್‌ನಲ್ಲಿ battery ಸಮಸ್ಯೆ ಇರುವ ಸಾಧ್ಯತೆ ಇದೆ."
            ),

            "main_advice": [
                "Battery charge level ಪರಿಶೀಲಿಸಿ.",
                "Battery terminals loose ಅಥವಾ dirty ಆಗಿವೆಯೇ ನೋಡಿ.",
                "Battery cable connections ಪರಿಶೀಲಿಸಿ.",
                "Dashboard battery warning light ಗಮನಿಸಿ.",
                "Battery ಹಳೆಯದಾಗಿದ್ದರೆ technician ಮೂಲಕ test ಮಾಡಿಸಿ."
            ],

            "do": [
                "Battery terminals ಅನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "Battery voltage ಅನ್ನು qualified technician ಮೂಲಕ test ಮಾಡಿಸಿ.",
                "Battery problem ಮತ್ತೆ ಮತ್ತೆ ಬರುತ್ತಿದೆಯೇ ಗಮನಿಸಿ."
            ],

            "dont": [
                "Battery terminals ಅನ್ನು short ಮಾಡಬೇಡಿ.",
                "ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ battery repair ಮಾಡಬೇಡಿ.",
                "Damaged battery ಅನ್ನು ಬಳಸುತ್ತಲೇ ಇರಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: battery completely dead ಆಗಿದೆಯೇ ಅಥವಾ "
                "tractor slow ಆಗಿ start ಆಗುತ್ತಿದೆಯೇ ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # TRACTOR ENGINE PROBLEM
    # =====================================================

    elif category == "tractor_engine_problem":

        return {
            "answer": (
                "ನಿಮ್ಮ ಟ್ರ್ಯಾಕ್ಟರ್ engine ನಲ್ಲಿ ಸಮಸ್ಯೆ ಇರುವ ಸಾಧ್ಯತೆ ಇದೆ. "
                "Engine smoke, overheating ಅಥವಾ unusual sound ಇರುವುದನ್ನು ಗಮನಿಸುವುದು ಮುಖ್ಯ."
            ),

            "main_advice": [
                "Engine oil level ಪರಿಶೀಲಿಸಿ.",
                "Coolant level ಪರಿಶೀಲಿಸಿ.",
                "Engine overheating ಆಗುತ್ತಿದೆಯೇ ಗಮನಿಸಿ.",
                "Smoke ಬಣ್ಣವನ್ನು ಗಮನಿಸಿ.",
                "Engine ನಲ್ಲಿ unusual sound ಅಥವಾ vibration ಇದೆಯೇ ಗಮನಿಸಿ."
            ],

            "do": [
                "Engine problem ಕಂಡುಬಂದರೆ tractor ಅನ್ನು ಸುರಕ್ಷಿತ ಸ್ಥಳದಲ್ಲಿ ನಿಲ್ಲಿಸಿ.",
                "Oil ಮತ್ತು coolant level ಪರಿಶೀಲಿಸಿ.",
                "Smoke ಅಥವಾ sound ಯಾವಾಗ ಬರುತ್ತದೆ ಎಂದು ಗಮನಿಸಿ.",
                "Qualified mechanic ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "dont": [
                "Engine overheating ಆಗುತ್ತಿದ್ದರೆ tractor ಓಡಿಸುತ್ತಲೇ ಇರಬೇಡಿ.",
                "Engine parts ಅನ್ನು ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ ತೆರೆಯಬೇಡಿ.",
                "Smoke ಸಮಸ್ಯೆಯನ್ನು ನಿರ್ಲಕ್ಷಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: engine smoke, overheating, sound ಅಥವಾ "
                "power ಕಡಿಮೆಯಾಗಿರುವುದರಲ್ಲಿ ಯಾವ ಸಮಸ್ಯೆ ಇದೆ ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # TRACTOR BRAKE PROBLEM
    # =====================================================

    elif category == "tractor_brake_problem":

        return {
            "answer": (
                "ನಿಮ್ಮ ಟ್ರ್ಯಾಕ್ಟರ್‌ನಲ್ಲಿ brake ಸಮಸ್ಯೆ ಇದೆ ಎಂದು ಅರ್ಥವಾಗಿದೆ. "
                "ಇದು safety-related problem ಆದ್ದರಿಂದ ಎಚ್ಚರಿಕೆ ಅಗತ್ಯ."
            ),

            "main_advice": [
                "Brake pedal response ಪರಿಶೀಲಿಸಿ.",
                "Brake oil ಅಥವಾ hydraulic system ಅನ್ನು technician ಮೂಲಕ ಪರಿಶೀಲಿಸಿ.",
                "Brake ಹಿಡಿಯುವ distance ಹೆಚ್ಚಾಗಿದೆಯೇ ಗಮನಿಸಿ.",
                "ಎರಡು brake sides ಸಮವಾಗಿ ಕೆಲಸ ಮಾಡುತ್ತಿವೆಯೇ ಪರಿಶೀಲಿಸಿ."
            ],

            "do": [
                "Tractor ಅನ್ನು ಸುರಕ್ಷಿತ ಸ್ಥಳದಲ್ಲಿ ನಿಲ್ಲಿಸಿ.",
                "Brake problem ಇದ್ದರೆ qualified mechanic ಅನ್ನು ಸಂಪರ್ಕಿಸಿ.",
                "Repair ಆದ ನಂತರ controlled area ನಲ್ಲಿ brake test ಮಾಡಿಸಿ."
            ],

            "dont": [
                "Brake ಸರಿಯಾಗಿ ಕೆಲಸ ಮಾಡದ tractor ಅನ್ನು ರಸ್ತೆಯಲ್ಲಿ ಓಡಿಸಬೇಡಿ.",
                "Brake system ಅನ್ನು ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ ತೆರೆಯಬೇಡಿ.",
                "Brake problem ಅನ್ನು ನಿರ್ಲಕ್ಷಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: brake completely ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲವೇ, "
                "ಅಥವಾ brake weak ಆಗಿದೆಯೇ ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # TRACTOR SERVICE
    # =====================================================

    elif category == "tractor_service":

        return {
            "answer": (
                "ನಿಮ್ಮ ಟ್ರ್ಯಾಕ್ಟರ್‌ಗೆ service ಅಥವಾ repair ಅಗತ್ಯವಿರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "Engine oil ಮತ್ತು coolant ಪರಿಶೀಲಿಸಿ.",
                "Battery ಮತ್ತು electrical connections ಪರಿಶೀಲಿಸಿ.",
                "Tyre pressure ಪರಿಶೀಲಿಸಿ.",
                "Brake ಮತ್ತು clutch condition ಗಮನಿಸಿ.",
                "Hydraulic system ಮತ್ತು leakage ಪರಿಶೀಲಿಸಿ.",
                "Regular service schedule ಪರಿಶೀಲಿಸಿ."
            ],

            "do": [
                "Tractor model ಮತ್ತು approximate running hours note ಮಾಡಿ.",
                "ಕೊನೆಯ service ಯಾವಾಗ ಮಾಡಿಸಿದ್ದೀರಿ ಎಂದು ಪರಿಶೀಲಿಸಿ.",
                "ಸಮಸ್ಯೆಯ ಲಕ್ಷಣಗಳನ್ನು note ಮಾಡಿ.",
                "Qualified mechanic ಅಥವಾ authorized service center ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "dont": [
                "ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ engine ಅಥವಾ hydraulic parts ತೆರೆಯಬೇಡಿ.",
                "Major problem ಇದ್ದಾಗ tractor ಅನ್ನು force ಮಾಡಿ ಓಡಿಸಬೇಡಿ.",
                "ಅಗತ್ಯವಿಲ್ಲದೆ spare parts ಬದಲಾಯಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ನಿಮಗೆ ಸಾಮಾನ್ಯ service ಬೇಕೇ ಅಥವಾ "
                "ಯಾವುದಾದರೂ specific problem ಇದೆಯೇ ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # PUMP / MOTOR
    # =====================================================

    elif category == "pump_motor_service":

        return {
            "answer": (
                "ನಿಮ್ಮ farm pump ಅಥವಾ motor ನಲ್ಲಿ ಸಮಸ್ಯೆ ಇರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "Power supply ಇದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "Switch ಮತ್ತು cable condition ಗಮನಿಸಿ.",
                "Motor start ಆಗುತ್ತಿದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "Water flow ಸರಿಯಾಗಿದೆಯೇ ಗಮನಿಸಿ.",
                "Motor unusual sound ಅಥವಾ overheating ಇದೆಯೇ ಗಮನಿಸಿ."
            ],

            "do": [
                "Power OFF ಮಾಡಿ ನಂತರ ಮಾತ್ರ physical connection ಪರಿಶೀಲಿಸಿ.",
                "Problem ಯಾವಾಗ ಪ್ರಾರಂಭವಾಯಿತು ಎಂದು ಗಮನಿಸಿ.",
                "Qualified electrician ಅಥವಾ motor technician ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "dont": [
                "Power ON ಇರುವಾಗ wiring touch ಮಾಡಬೇಡಿ.",
                "Motor ಅನ್ನು ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ ತೆರೆಯಬೇಡಿ.",
                "Electrical fault ಇದ್ದಾಗ ಸ್ವತಃ repair ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: motor start ಆಗುತ್ತಿಲ್ಲವೇ, "
                "water ಬರುತ್ತಿಲ್ಲವೇ ಅಥವಾ sound/heat problem ಇದೆಯೇ ಎಂದು ಹೇಳಿ."
            )
        }

    # =====================================================
    # FARM MACHINERY
    # =====================================================

    elif category == "farm_machinery_service":

        return {
            "answer": (
                "ನಿಮ್ಮ ಕೃಷಿ ಯಂತ್ರೋಪಕರಣದಲ್ಲಿ service ಅಥವಾ repair ಅಗತ್ಯವಿದೆ."
            ),

            "main_advice": [
                "ಯಂತ್ರದ ಹೆಸರು ಮತ್ತು model ಗುರುತಿಸಿ.",
                "ಯಂತ್ರ start ಆಗುತ್ತಿದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "Unusual sound ಅಥವಾ vibration ಗಮನಿಸಿ.",
                "Blade, belt ಮತ್ತು moving parts ಪರಿಶೀಲಿಸಿ.",
                "Fuel ಅಥವಾ electrical system ಅಗತ್ಯವಿದ್ದರೆ technician ಮೂಲಕ ಪರಿಶೀಲಿಸಿ."
            ],

            "do": [
                "ಯಂತ್ರದ ಸಮಸ್ಯೆಯನ್ನು note ಮಾಡಿ.",
                "Machine ಅನ್ನು ಸ್ವಚ್ಛವಾಗಿ ಇಟ್ಟುಕೊಳ್ಳಿ.",
                "Qualified technician ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "dont": [
                "Machine running ಇರುವಾಗ moving parts touch ಮಾಡಬೇಡಿ.",
                "Safety guard ತೆಗೆದು operate ಮಾಡಬೇಡಿ.",
                "ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ machine repair ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ machine ಎಂದು ಹೇಳಿ. "
                "ಉದಾಹರಣೆಗೆ sprayer, rotavator, cultivator ಅಥವಾ harvester."
            )
        }

    # =====================================================
    # SEED
    # =====================================================

    elif category == "seed_service":

        return {
            "answer": (
                "ನಿಮಗೆ ಬೀಜದ ಆಯ್ಕೆ ಅಥವಾ ಬೀಜ ಖರೀದಿ ಬಗ್ಗೆ ಸಹಾಯ ಬೇಕಾಗಿದೆ."
            ),

            "main_advice": [
                "ಯಾವ ಬೆಳೆಗೆ ಬೀಜ ಬೇಕು ಎಂದು ಮೊದಲು ಗುರುತಿಸಿ.",
                "ನಿಮ್ಮ ಪ್ರದೇಶ ಮತ್ತು ಹವಾಮಾನಕ್ಕೆ ಸೂಕ್ತವಾದ variety ಆಯ್ಕೆ ಮಾಡಿ.",
                "Certified ಮತ್ತು ಉತ್ತಮ ಗುಣಮಟ್ಟದ seed ಆಯ್ಕೆ ಮಾಡಿ.",
                "Seed packet ಮೇಲಿನ lot, expiry ಮತ್ತು certification information ಪರಿಶೀಲಿಸಿ."
            ],

            "do": [
                "ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ನಿಮ್ಮ ಪ್ರದೇಶವನ್ನು ತಿಳಿಸಿ.",
                "ನಿಮ್ಮ season ಅನ್ನು ತಿಳಿಸಿ.",
                "ವಿಶ್ವಾಸಾರ್ಹ seed seller ಅನ್ನು ಆಯ್ಕೆ ಮಾಡಿ."
            ],

            "dont": [
                "ಗುಣಮಟ್ಟ ತಿಳಿಯದ seed ಖರೀದಿಸಬೇಡಿ.",
                "Expiry ಆಗಿರುವ seed ಬಳಸಬೇಡಿ.",
                "ಕೇವಲ ಕಡಿಮೆ ಬೆಲೆ ನೋಡಿ seed ಆಯ್ಕೆ ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ ಬೆಳೆಗೆ seed ಬೇಕು ಮತ್ತು "
                "ಯಾವ season ನಲ್ಲಿ ಬೆಳೆಯಲು ಬಯಸುತ್ತೀರಿ ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # AGRICULTURE SHOP
    # =====================================================

    elif category == "agri_shop_service":

        return {
            "answer": (
                "ನಿಮಗೆ ಕೃಷಿ ಮಳಿಗೆ ಅಥವಾ ಕೃಷಿ ಸಾಮಗ್ರಿಗಳ ಬಗ್ಗೆ ಸಹಾಯ ಬೇಕಾಗಿದೆ."
            ),

            "main_advice": [
                "ಬೇಕಾದ product ಅನ್ನು ಮೊದಲು ಗುರುತಿಸಿ.",
                "Seed, fertilizer, pesticide ಅಥವಾ equipment ಯಾವುದೆಂದು ತಿಳಿಸಿ.",
                "Product label ಮತ್ತು expiry ಪರಿಶೀಲಿಸಿ.",
                "ಅಧಿಕೃತ ಅಥವಾ ವಿಶ್ವಾಸಾರ್ಹ seller ಆಯ್ಕೆ ಮಾಡಿ."
            ],

            "do": [
                "Product ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ಅಗತ್ಯವಿದ್ದರೆ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
            ],

            "dont": [
                "Label ಇಲ್ಲದ product ಖರೀದಿಸಬೇಡಿ.",
                "ಅಗತ್ಯವಿಲ್ಲದೆ pesticide ಬಳಸಬೇಡಿ.",
                "ತಜ್ಞರ ಸಲಹೆ ಇಲ್ಲದೆ chemical mix ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ನಿಮಗೆ ಯಾವ product ಬೇಕು ಎಂದು ಹೇಳಿ."
            )
        }

    # =====================================================
    # VETERINARY
    # =====================================================

    elif category == "veterinary_service":

        return {
            "answer": (
                "ನಿಮ್ಮ ಜಾನುವಾರಿಗೆ veterinary help ಅಗತ್ಯವಿರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "ಯಾವ ಜಾನುವಾರಿಗೆ ಸಮಸ್ಯೆ ಇದೆ ಎಂದು ಗುರುತಿಸಿ.",
                "ಜಾನುವಾರು ಆಹಾರ ತಿನ್ನುತ್ತಿದೆಯೇ ಗಮನಿಸಿ.",
                "ನೀರು ಕುಡಿಯುತ್ತಿದೆಯೇ ಗಮನಿಸಿ.",
                "ಜ್ವರ, ಗಾಯ, swelling ಅಥವಾ unusual behaviour ಇದೆಯೇ ಪರಿಶೀಲಿಸಿ."
            ],

            "do": [
                "ಲಕ್ಷಣಗಳು ಯಾವಾಗ ಪ್ರಾರಂಭವಾದವು ಎಂದು ಗಮನಿಸಿ.",
                "ಜಾನುವಾರದ ಸ್ಥಿತಿಯನ್ನು ಗಮನಿಸಿ.",
                "ಗಂಭೀರ ಲಕ್ಷಣಗಳಿದ್ದರೆ veterinary doctor ಅನ್ನು ಸಂಪರ್ಕಿಸಿ."
            ],

            "dont": [
                "ವೈದ್ಯರ ಸಲಹೆ ಇಲ್ಲದೆ medicine ಅಥವಾ injection ನೀಡಬೇಡಿ.",
                "ಗಂಭೀರ ಸ್ಥಿತಿಯಲ್ಲಿ ಚಿಕಿತ್ಸೆ ವಿಳಂಬ ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ animal ಮತ್ತು ಏನು problem ಎಂದು ಹೇಳಿ. "
                "ಉದಾಹರಣೆಗೆ: ಹಸು ತಿನ್ನುತ್ತಿಲ್ಲ, ಮೇಕೆಗೆ ಗಾಯವಾಗಿದೆ."
            )
        }

    # =====================================================
    # FARM LABOUR
    # =====================================================

    elif category == "farm_labour":

        return {
            "answer": (
                "ನಿಮಗೆ ಕೃಷಿ ಕೆಲಸಕ್ಕೆ ಕಾರ್ಮಿಕರು ಅಥವಾ farm workers ಬೇಕಾಗಿದ್ದಾರೆ ಎಂದು ಅರ್ಥವಾಗಿದೆ."
            ),

            "main_advice": [
                "ಯಾವ ಕೃಷಿ ಕೆಲಸಕ್ಕೆ workers ಬೇಕು ಎಂದು ನಿರ್ಧರಿಸಿ.",
                "ಎಷ್ಟು ಜನ workers ಬೇಕು ಎಂದು ನಿರ್ಧರಿಸಿ.",
                "ಕೆಲಸದ ದಿನಗಳು ಮತ್ತು ಸಮಯವನ್ನು ನಿಗದಿಪಡಿಸಿ.",
                "ಸ್ಥಳೀಯವಾಗಿ ವಿಶ್ವಾಸಾರ್ಹ workers ಅಥವಾ farmer groups ಮೂಲಕ ಹುಡುಕಿ."
            ],

            "do": [
                "ಕೆಲಸದ ಪ್ರಕಾರವನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ತಿಳಿಸಿ.",
                "ಕೆಲಸದ ಸ್ಥಳ ಮತ್ತು ದಿನಾಂಕವನ್ನು ತಿಳಿಸಿ.",
                "ಕೆಲಸದ ಅವಧಿ ಮತ್ತು payment ಅನ್ನು ಮುಂಚಿತವಾಗಿ ಸ್ಪಷ್ಟಪಡಿಸಿ."
            ],

            "dont": [
                "ಕೆಲಸದ ಷರತ್ತುಗಳನ್ನು ಸ್ಪಷ್ಟಪಡಿಸದೆ worker ಕರೆಸಿಕೊಳ್ಳಬೇಡಿ.",
                "ಅಸುರಕ್ಷಿತ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ ಕೆಲಸ ಮಾಡಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ ಕೆಲಸಕ್ಕೆ workers ಬೇಕು, "
                "ಎಷ್ಟು ಜನ ಮತ್ತು ಯಾವ ಪ್ರದೇಶದಲ್ಲಿ ಬೇಕು ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # MACHINERY RENTAL
    # =====================================================

    elif category == "machinery_rental":

        return {
            "answer": (
                "ನಿಮಗೆ tractor ಅಥವಾ ಕೃಷಿ ಯಂತ್ರವನ್ನು ಬಾಡಿಗೆಗೆ ಪಡೆಯುವ ಸಹಾಯ ಬೇಕಾಗಿದೆ."
            ),

            "main_advice": [
                "ಯಾವ machine ಬೇಕು ಎಂದು ಮೊದಲು ನಿರ್ಧರಿಸಿ.",
                "ಎಷ್ಟು ಗಂಟೆ ಅಥವಾ ಎಷ್ಟು ದಿನ ಬೇಕು ಎಂದು ನಿರ್ಧರಿಸಿ.",
                "Machine condition ಪರಿಶೀಲಿಸಿ.",
                "Rental charge ಮತ್ತು fuel terms ಮುಂಚಿತವಾಗಿ ತಿಳಿದುಕೊಳ್ಳಿ."
            ],

            "do": [
                "Machine owner ಅಥವಾ rental provider ಜೊತೆ rate confirm ಮಾಡಿ.",
                "Machine ಕೆಲಸಕ್ಕೆ ಸೂಕ್ತವಾಗಿದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "Payment ಮತ್ತು rental duration ಸ್ಪಷ್ಟಪಡಿಸಿ."
            ],

            "dont": [
                "Machine condition ಪರಿಶೀಲಿಸದೆ rent ತೆಗೆದುಕೊಳ್ಳಬೇಡಿ.",
                "Payment terms ತಿಳಿಯದೆ ಹಣ ನೀಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ machine ಬೇಕು ಮತ್ತು "
                "ಎಷ್ಟು ದಿನ/ಗಂಟೆ ಬೇಕು ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # GENERAL SERVICE
    # =====================================================

    elif category == "general_service":

        return {
            "answer": (
                "ನಿಮಗೆ ಒಂದು service ಅಥವಾ repair ಸಹಾಯ ಬೇಕಾಗಿದೆ. "
                "ಸಮಸ್ಯೆಯ ವಸ್ತು ಮತ್ತು ಸಮಸ್ಯೆಯ ಲಕ್ಷಣ ತಿಳಿಸಿದರೆ ಇನ್ನಷ್ಟು ಸೂಕ್ತವಾಗಿ ಸಹಾಯ ಮಾಡಬಹುದು."
            ),

            "main_advice": [
                "ಯಾವ ವಸ್ತು ಅಥವಾ machine ಸಮಸ್ಯೆ ಎಂದು ಗುರುತಿಸಿ.",
                "ಸಮಸ್ಯೆಯ ಮುಖ್ಯ ಲಕ್ಷಣವನ್ನು ಗಮನಿಸಿ.",
                "Problem ಯಾವಾಗ ಪ್ರಾರಂಭವಾಯಿತು ಎಂದು ಗಮನಿಸಿ."
            ],

            "do": [
                "ವಸ್ತುವಿನ ಹೆಸರು ತಿಳಿಸಿ.",
                "Problem ಅನ್ನು ವಿವರಿಸಿ.",
                "ಸಾಧ್ಯವಾದರೆ photo upload ಮಾಡಿ."
            ],

            "dont": [
                "ಸಮಸ್ಯೆ ತಿಳಿಯದೆ repair ಪ್ರಯತ್ನ ಮಾಡಬೇಡಿ.",
                "Electrical ಅಥವಾ mechanical parts ಅನ್ನು ತಿಳುವಳಿಕೆ ಇಲ್ಲದೆ ತೆರೆಯಬೇಡಿ."
            ],

            "next_step": (
                "ಉದಾಹರಣೆಗೆ: 'ನನ್ನ pump start ಆಗುತ್ತಿಲ್ಲ', "
                "'tractor brake problem ಇದೆ' ಅಥವಾ "
                "'sprayer repair ಬೇಕು' ಎಂದು ಹೇಳಬಹುದು."
            )
        }

    # =====================================================
    # PEST
    # =====================================================

    elif category == "pest":

        return {
            "answer": (
                "ನಿಮ್ಮ ಬೆಳೆಗೆ ಕೀಟ ಅಥವಾ ಹುಳು ಸಮಸ್ಯೆ ಇರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "ಯಾವ ಬೆಳೆಯಲ್ಲಿ ಕೀಟ ಬಂದಿದೆ ಎಂದು ಗುರುತಿಸಿ.",
                "ಕೀಟವು ಎಲೆ, ಕಾಂಡ, ಹೂ ಅಥವಾ ಹಣ್ಣಿನಲ್ಲಿ ಇದೆಯೇ ಗಮನಿಸಿ.",
                "ಕೀಟದ ಪ್ರಮಾಣ ಹೆಚ್ಚಾಗುತ್ತಿದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "ಸಾಧ್ಯವಾದರೆ clear photo upload ಮಾಡಿ."
            ],

            "do": [
                "ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ಕೀಟದ clear photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ಸಮಸ್ಯೆ ಹೆಚ್ಚಾಗುತ್ತಿದೆಯೇ ಗಮನಿಸಿ.",
                "ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
            ],

            "dont": [
                "ಕೀಟವನ್ನು ಗುರುತಿಸದೆ pesticide ಬಳಸಬೇಡಿ.",
                "AI result ಮಾತ್ರ ನೋಡಿ treatment ಆರಂಭಿಸಬೇಡಿ.",
                "ಅಗತ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚು pesticide ಬಳಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಬೆಳೆಯ ಹೆಸರು ಮತ್ತು ಕೀಟದ photo upload ಮಾಡಿ."
            )
        }

    # =====================================================
    # WATER
    # =====================================================

    elif category == "water":

        return {
            "answer": (
                "ನಿಮ್ಮ ಬೆಳೆಗೆ ನೀರಿನ ಸಮಸ್ಯೆ ಇರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "ಮಣ್ಣಿನ ತೇವಾಂಶ ಪರಿಶೀಲಿಸಿ.",
                "ನೀರಾವರಿ ವ್ಯವಸ್ಥೆ ಸರಿಯಾಗಿ ಕೆಲಸ ಮಾಡುತ್ತಿದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "ಬೆಳೆಯ ಹಂತಕ್ಕೆ ಅನುಗುಣವಾಗಿ ನೀರು ನೀಡಿ.",
                "ನೀರು ನಿಲ್ಲುವ ಸಮಸ್ಯೆಯೂ ಇದೆಯೇ ಗಮನಿಸಿ."
            ],

            "do": [
                "ಮಣ್ಣಿನ moisture ಪರಿಶೀಲಿಸಿ.",
                "Pump ಮತ್ತು irrigation system ಪರಿಶೀಲಿಸಿ.",
                "ನೀರಿನ ಲಭ್ಯತೆಯನ್ನು ಗಮನಿಸಿ."
            ],

            "dont": [
                "ಅಗತ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚು ನೀರು ಹಾಕಬೇಡಿ.",
                "ನೀರು ನಿಲ್ಲುವಂತೆ ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ನೀರು ಕಡಿಮೆ ಇದೆಯೇ, "
                "pump problem ಇದೆಯೇ ಅಥವಾ irrigation problem ಇದೆಯೇ ಎಂದು ಹೇಳಿ."
            )
        }

    # =====================================================
    # LEAF
    # =====================================================

    elif category == "leaf":

        return {
            "answer": (
                "ನಿಮ್ಮ ಬೆಳೆಯ ಎಲೆಗಳಲ್ಲಿ ಸಮಸ್ಯೆ ಇರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "ಎಲೆಯ ಬಣ್ಣವನ್ನು ಗಮನಿಸಿ.",
                "ಕಲೆಗಳು, holes ಅಥವಾ curling ಇದೆಯೇ ಪರಿಶೀಲಿಸಿ.",
                "ಎಲೆ ಒಣಗುತ್ತಿದೆಯೇ ಗಮನಿಸಿ.",
                "ಸಮಸ್ಯೆ ಯಾವ ಭಾಗದಲ್ಲಿ ಹೆಚ್ಚು ಇದೆ ಎಂದು ಪರಿಶೀಲಿಸಿ."
            ],

            "do": [
                "ಪೀಡಿತ ಎಲೆಯ clear photo ತೆಗೆದುಕೊಳ್ಳಿ.",
                "ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ಸಮಸ್ಯೆ ಹೆಚ್ಚಾಗುತ್ತಿದೆಯೇ ಗಮನಿಸಿ."
            ],

            "dont": [
                "ಕಾರಣ ತಿಳಿಯದೆ pesticide ಬಳಸಬೇಡಿ.",
                "AI result ಮಾತ್ರ ಆಧರಿಸಿ treatment ಆರಂಭಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಬೆಳೆಯ ಹೆಸರು ಮತ್ತು ಪೀಡಿತ ಎಲೆಯ photo upload ಮಾಡಿ."
            )
        }

    # =====================================================
    # DISEASE
    # =====================================================

    elif category == "disease":

        return {
            "answer": (
                "ನಿಮ್ಮ ಬೆಳೆಯಲ್ಲಿ ರೋಗದ ಲಕ್ಷಣಗಳು ಕಂಡುಬರುತ್ತಿರುವಂತೆ ಕಾಣುತ್ತಿದೆ."
            ),

            "main_advice": [
                "ರೋಗದ ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.",
                "ಪೀಡಿತ ಎಲೆ, ಕಾಂಡ ಮತ್ತು ಹಣ್ಣು ಪರಿಶೀಲಿಸಿ.",
                "ಸಮಸ್ಯೆ ಎಷ್ಟು ವೇಗವಾಗಿ ಹರಡುತ್ತಿದೆ ಎಂದು ಗಮನಿಸಿ."
            ],

            "do": [
                "ಪೀಡಿತ ಭಾಗದ clear photo upload ಮಾಡಿ.",
                "ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ಅಗತ್ಯವಿದ್ದರೆ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
            ],

            "dont": [
                "ರೋಗ ಖಚಿತವಾಗದೆ chemical ಬಳಸಬೇಡಿ.",
                "AI result ಮಾತ್ರ ನೋಡಿ treatment ಆರಂಭಿಸಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಬೆಳೆಯ ಹೆಸರು ಮತ್ತು ಪೀಡಿತ ಭಾಗದ photo upload ಮಾಡಿ."
            )
        }

    # =====================================================
    # FERTILIZER
    # =====================================================

    elif category == "fertilizer":

        return {
            "answer": (
                "ನಿಮ್ಮ ಬೆಳೆಗೆ ಯಾವ ಗೊಬ್ಬರ ಸೂಕ್ತ ಎಂಬುದನ್ನು "
                "ಬೆಳೆ ಮತ್ತು ಮಣ್ಣಿನ ಸ್ಥಿತಿಯನ್ನು ಆಧರಿಸಿ ನಿರ್ಧರಿಸಬೇಕು."
            ),

            "main_advice": [
                "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮಾಡಿಸುವುದು ಉತ್ತಮ.",
                "ಬೆಳೆಯ ಹಂತವನ್ನು ಗಮನಿಸಿ.",
                "Nitrogen, phosphorus ಮತ್ತು potassium ಅಗತ್ಯವನ್ನು ಪರಿಗಣಿಸಿ.",
                "ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
            ],

            "do": [
                "ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                "ಬೆಳೆಯ stage ತಿಳಿಸಿ.",
                "ಸಾಧ್ಯವಾದರೆ soil test ಮಾಡಿಸಿ."
            ],

            "dont": [
                "ಊಹೆಯಿಂದ ಹೆಚ್ಚಿನ ಪ್ರಮಾಣದ fertilizer ಹಾಕಬೇಡಿ.",
                "ಅಗತ್ಯವಿಲ್ಲದೆ ಹಲವಾರು fertilizers mix ಮಾಡಬೇಡಿ."
            ],

            "next_step": (
                "ಮುಂದಿನ ಹಂತ: ಯಾವ ಬೆಳೆಗೆ ಗೊಬ್ಬರ ಬೇಕು ಮತ್ತು "
                "ಬೆಳೆ ಯಾವ stage ನಲ್ಲಿ ಇದೆ ಎಂದು ತಿಳಿಸಿ."
            )
        }

    # =====================================================
    # UNKNOWN / GENERAL QUESTION
    # =====================================================
    else:
        return generate_general_farmer_qa(problem)


# =========================================================
# PROBLEM API
# =========================================================

@app.post("/problem")
def receive_problem(data: ProblemRequest):
    problem = data.problem.strip()

    # 1. Existing category system
    category = classify_problem(problem)

    # 2. Detect context from the current question
    current_context = extract_smart_context(problem)

    # 3. Read previous conversation context
    previous_context = data.previous_context or {}

    previous_smart_context = previous_context.get(
        "smart_context", {}
    )

    if not isinstance(previous_smart_context, dict):
        previous_smart_context = {}

    previous_problem = previous_context.get(
        "last_problem", ""
    )

    if not isinstance(previous_problem, str):
        previous_problem = ""

    # 4. Check whether this is a follow-up answer
    
    is_follow_up = (
        bool(previous_context)
        and bool(previous_smart_context)
        and not current_context.get("crop")
        and (
            current_context.get("intent") in [
                "farmer_question",
                "general_question",
                "crop_question"
            ]
            or current_context.get("symptom") is not None
        )
    )

    if is_follow_up:
        # Continue the previous crop conversation
        smart_context = {
            "crop": (
                current_context.get("crop")
                or previous_smart_context.get("crop")
            ),
            "problem": (
                previous_smart_context.get("problem")
                if (
                    current_context.get("symptom") is not None
                    or current_context.get("problem")
                    in [None, "general_crop_problem", "soil"]
                )
                else current_context.get("problem")
            ),
            "intent": previous_smart_context.get(
                "intent",
                current_context.get("intent")
            ),
            "crop_age_days": (
                current_context.get("crop_age_days")
                if current_context.get("crop_age_days") is not None
                
                else previous_smart_context.get("crop_age_days")
            ),

            "symptom": (
                current_context.get("symptom")
                if current_context.get("symptom") is not None
                else previous_smart_context.get("symptom")
            )
        }


        combined_problem = (
            f"{previous_problem}\n"
            f"Farmer's additional information: {problem}"
        )

    else:
        # Start a new question or conversation topic
        smart_context = current_context
        combined_problem = problem

    # 5. Generate the best available response
    # Keep existing smart crop diagnosis as the first choice.
    smart_response = generate_smart_crop_response(
        smart_context,
        combined_problem
    )

    if (
        smart_response is not None
        and smart_context.get("intent") in ["crop_problem", "harvest"]
    ):
        response = smart_response
        response_source = "smart_context"
    else:
        # Ask Gemini for general questions and follow-ups.
        response = generate_gemini_farmer_response(
            problem=combined_problem if is_follow_up else problem,
            smart_context=smart_context,
            previous_context=previous_context
        )

        if response is not None:
            response_source = "gemini"
        else:
            # Safe fallback: preserve the existing response system.
            response = generate_farmer_response(
                category,
                combined_problem if is_follow_up else problem
            )
            response_source = "existing_system"

    # 6. Save context for the next conversation turn
    next_context = {
        "last_problem": combined_problem,
        "smart_context": smart_context
    }

    # 7. Return the response and conversation context
    return {
        "message": "KrushiVaani Response",
        "conversation_id": data.conversation_id,
        "is_follow_up": is_follow_up,
        "problem": problem,
        "category": category,
        "smart_context": smart_context,
        "crop": smart_context.get("crop"),
        "detected_problem": smart_context.get("problem"),
        "intent": smart_context.get("intent"),
        "response_source": response_source,
        "answer": response["answer"],
        "main_advice": response["main_advice"],
        "do": response["do"],
        "dont": response["dont"],
        "next_step": response["next_step"],
        "conversation_context": next_context
    }



    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {

        "message":
            "KrushiVaani Response",

        "problem":
            problem,

        "category":
            category,

        # NEW SMART CONTEXT
        "smart_context":
            smart_context,

        "crop":
            smart_context["crop"],

        "detected_problem":
            smart_context["problem"],

        "intent":
            smart_context["intent"],

        # EXISTING RESPONSE
        "answer":
            response["answer"],

        "main_advice":
            response["main_advice"],

        "do":
            response["do"],

        "dont":
            response["dont"],

        "next_step":
            response["next_step"]
    }
# =========================================================
# NEARBY TRACTOR SHOWROOMS
# =========================================================

class ShowroomRequest(BaseModel):

    latitude: float
    longitude: float
    radius_meters: float = 25000
    brand: Optional[str] = None
    max_results: int = 10


TRACTOR_BRANDS = [
    "Mahindra",
    "John Deere",
    "Swaraj",
    "Sonalika",
    "New Holland",
    "Massey Ferguson",
    "Kubota",
    "Farmtrac",
    "Eicher"
]


def search_google_places(
    query: str,
    latitude: float,
    longitude: float,
    radius_meters: float,
    max_results: int
):

    api_key = os.getenv(
        "GOOGLE_PLACES_API_KEY"
    )

    if not api_key:

        return {
            "success": False,
            "error": "Google Places API key is not configured."
        }

    url = (
        "https://places.googleapis.com/v1/places:searchText"
    )

    payload = {

        "textQuery": query,

        "maxResultCount": min(
            max_results,
            20
        ),

        "rankPreference": "DISTANCE",

        "locationBias": {

            "circle": {

                "center": {

                    "latitude": latitude,

                    "longitude": longitude
                },

                "radius": min(
                    radius_meters,
                    50000
                )
            }
        },

        "languageCode": "en",

        "regionCode": "IN"
    }

    headers = {

        "Content-Type":
            "application/json",

        "X-Goog-Api-Key":
            api_key,

        "X-Goog-FieldMask":
            ",".join([

                "places.id",

                "places.displayName",

                "places.formattedAddress",

                "places.location",

                "places.nationalPhoneNumber",

                "places.internationalPhoneNumber",

                "places.googleMapsUri",

                "places.businessStatus",

                "places.primaryType"

            ])
    }

    try:

        req = urllib_request.Request(

            url,

            data=json.dumps(
                payload
            ).encode("utf-8"),

            headers=headers,

            method="POST"
        )

        with urllib_request.urlopen(
            req,
            timeout=15
        ) as response:

            response_data = (
                response
                .read()
                .decode("utf-8")
            )

            return {

                "success": True,

                "data": json.loads(
                    response_data
                )
            }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)
        }


@app.post(
    "/nearby-tractor-showrooms"
)
def nearby_tractor_showrooms(
    data: ShowroomRequest
):

    if not (
        -90 <= data.latitude <= 90
    ):

        return {

            "message":
                "Invalid latitude.",

            "showrooms": []
        }


    if not (
        -180 <= data.longitude <= 180
    ):

        return {

            "message":
                "Invalid longitude.",

            "showrooms": []
        }


    if data.brand:

        search_query = (
            data.brand
            + " tractor showroom"
        )

    else:

        search_query = (
            "tractor showroom"
        )


    result = search_google_places(

        query=search_query,

        latitude=data.latitude,

        longitude=data.longitude,

        radius_meters=data.radius_meters,

        max_results=data.max_results
    )


    if not result["success"]:

        return {

            "message":
                "Showroom search unavailable.",

            "showrooms": [],

            "error":
                result["error"]
        }


    places = result[
        "data"
    ].get(
        "places",
        []
    )


    showrooms = []


    for place in places:

        name = (
            place
            .get(
                "displayName",
                {}
            )
            .get(
                "text",
                "Tractor Showroom"
            )
        )


        location = place.get(
            "location",
            {}
        )


        phone = (

            place.get(
                "nationalPhoneNumber"
            )

            or

            place.get(
                "internationalPhoneNumber"
            )

        )


        showrooms.append({

            "place_id":
                place.get("id"),

            "name":
                name,

            "address":
                place.get(
                    "formattedAddress"
                ),

            "latitude":
                location.get(
                    "latitude"
                ),

            "longitude":
                location.get(
                    "longitude"
                ),

            "phone":
                phone,

            "maps_url":
                place.get(
                    "googleMapsUri"
                ),

            "business_status":
                place.get(
                    "businessStatus"
                ),

            "primary_type":
                place.get(
                    "primaryType"
                )
        })


    return {

        "message":
            "Nearby tractor showrooms found.",

        "count":
            len(showrooms),

        "showrooms":
            showrooms
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