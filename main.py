
import json
import os
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

MODEL_PATH = ARTIFACT_DIR / "ticket_classifier.keras"
VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
CLASSES_PATH = ARTIFACT_DIR / "classes.json"

# Initial heuristic; tune this using validation data.
CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "0.70")
)


# --------------------------------------------------
# Support queue -> team mapping
# Adjust these teams to match your organization.
# --------------------------------------------------

QUEUE_TO_TEAM = {
    "Billing and Payments": "Billing Team",
    "Customer Service": "Customer Service Team",
    "General Inquiry": "General Support Team",
    "Human Resources": "Human Resources Team",
    "IT Support": "IT Support Team",
    "Product Support": "Product Support Team",
    "Returns and Exchanges": "Returns Team",
    "Sales and Pre-Sales": "Sales Team",
    "Service Outages and Maintenance": "Infrastructure Team",
    "Technical Support": "Technical Support Team",
}


# --------------------------------------------------
# Load artifacts once
# --------------------------------------------------

for path in [MODEL_PATH, VECTORIZER_PATH, CLASSES_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

model = tf.keras.models.load_model(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    classes = json.load(file)

if model.output_shape[-1] != len(classes):
    raise ValueError(
        "Model output count does not match classes.json"
    )


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Support Ticket Classification and Routing API",
    description=(
        "Classifies support tickets using a trained TF-IDF "
        "and Deep MLP model, then recommends a support team."
    ),
    version="1.0.0",
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class TicketRequest(BaseModel):
    subject: str = Field(
        min_length=1,
        max_length=300,
        description="Ticket subject"
    )
    body: str = Field(
        min_length=1,
        max_length=10000,
        description="Ticket description"
    )


@app.get("/")
def root():
    return {
        "message": "Support Ticket Classification API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "number_of_queues": len(classes),
    }


@app.post("/predict")
def predict_ticket(ticket: TicketRequest):
    subject = ticket.subject.strip()
    body = ticket.body.strip()

    if not subject or not body:
        raise HTTPException(
            status_code=422,
            detail="Subject and body must not be blank."
        )

    # Keep preprocessing consistent with training:
    # subject + a space + body.
    text = f"{subject} {body}"

    # Transform with the already-fitted vectorizer.
    features = vectorizer.transform([text]).toarray()
    features = features.astype(np.float32)

    # Predict probabilities for all queues.
    probabilities = model.predict(features, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_queue = classes[predicted_index]
    confidence = float(probabilities[predicted_index])

    # Return the highest-scoring alternatives for review.
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        {
            "queue": classes[int(index)],
            "confidence": round(float(probabilities[index]), 4),
        }
        for index in top_indices
    ]

    review_required = confidence < CONFIDENCE_THRESHOLD
    login_keywords = [
        "login",
        "log in",
        "sign in",
        "password",
        "account access",
        "unable to access",
    ]

    login_related = any(
        keyword in text.lower()
        for keyword in login_keywords
    )

    needs_manual_review = (
        confidence < CONFIDENCE_THRESHOLD
        or (
            login_related
            and predicted_queue == "Billing and Payments"
        )
    )

    return {
    "subject": subject,
    "predicted_queue": predicted_queue,
    "confidence": round(confidence, 4),
    "assigned_team": (
        "Manual Review"
        if needs_manual_review
        else QUEUE_TO_TEAM.get(
            predicted_queue,
            "General Support Team"
        )
    ),
    "review_required": needs_manual_review,
    "top_predictions": top_predictions,
}