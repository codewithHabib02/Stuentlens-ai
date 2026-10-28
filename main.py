import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# =========================================================
# MODEL FEATURES
# Keep the exact feature names and order used during training.
# =========================================================

FEATURES = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Tuition fees up to date",
    "Scholarship holder",
    "Debtor",
    "Age at enrollment",
    "Admission grade",
    "Gender",
    "Application mode",
    "Course",
]

NUMERIC_FEATURES = FEATURES[:5] + [
    FEATURES[8],
    FEATURES[9],
]

BINARY_FEATURES = FEATURES[5:8] + [
    FEATURES[10],
]


# Course is intentionally excluded from this dictionary.
# Any positive integer Course code passes this API's validation.
CATEGORIES = {
    "Tuition fees up to date": [0, 1],
    "Scholarship holder": [0, 1],
    "Debtor": [0, 1],
    "Gender": [0, 1],
    "Application mode": [
        1, 2, 5, 7, 10, 15, 16, 17, 18,
        26, 27, 39, 42, 43, 44, 51, 53, 57,
    ],
}


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "student_outcome_model.pkl"),
    )
)

TURSO_URL = os.getenv(
    "TURSO_DATABASE_URL",
    "",
).strip()

TURSO_TOKEN = os.getenv(
    "TURSO_AUTH_TOKEN",
    "",
).strip()

FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5173",
).strip()

EXTRA_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

ALLOWED_ORIGINS = list(
    dict.fromkeys(
        [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            FRONTEND_ORIGIN,
            *EXTRA_ORIGINS,
        ]
    )
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="StudentLens AI API",
    version="1.0.0",
    description=(
        "Student academic outcome predictions "
        "using the supplied saved machine-learning pipeline."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# =========================================================
# MODEL STATUS
# =========================================================

model = None
model_error = None
model_classes: list[Any] = []
model_has_probabilities = False
db_error = None


# =========================================================
# PREDICTION INPUT
# =========================================================

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    curricular_credited: float = Field(
        alias="Curricular units 1st sem (credited)",
        ge=0,
    )

    curricular_enrolled: float = Field(
        alias="Curricular units 1st sem (enrolled)",
        ge=0,
    )

    curricular_evaluations: float = Field(
        alias="Curricular units 1st sem (evaluations)",
        ge=0,
    )

    curricular_approved: float = Field(
        alias="Curricular units 1st sem (approved)",
        ge=0,
    )

    curricular_grade: float = Field(
        alias="Curricular units 1st sem (grade)",
        ge=0,
    )

    tuition_fees_up_to_date: int = Field(
        alias="Tuition fees up to date",
        ge=0,
        le=1,
    )

    scholarship_holder: int = Field(
        alias="Scholarship holder",
        ge=0,
        le=1,
    )

    debtor: int = Field(
        alias="Debtor",
        ge=0,
        le=1,
    )

    age_at_enrollment: int = Field(
        alias="Age at enrollment",
        ge=16,
        le=100,
    )

    admission_grade: float = Field(
        alias="Admission grade",
        ge=0,
        le=200,
    )

    gender: int = Field(
        alias="Gender",
        ge=0,
        le=1,
    )

    application_mode: int = Field(
        alias="Application mode",
    )

    # Accept any positive integer Course code.
    # Do not restrict this field to a predefined category list.
    course: int = Field(
        alias="Course",
        gt=0,
    )

    def as_features(self) -> dict[str, Any]:
        values = [
            self.curricular_credited,
            self.curricular_enrolled,
            self.curricular_evaluations,
            self.curricular_approved,
            self.curricular_grade,
            self.tuition_fees_up_to_date,
            self.scholarship_holder,
            self.debtor,
            self.age_at_enrollment,
            self.admission_grade,
            self.gender,
            self.application_mode,
            self.course,
        ]

        return dict(zip(FEATURES, values))


# =========================================================
# TURSO DATABASE CONNECTION
# =========================================================

def connect_db():
    if not TURSO_URL or not TURSO_TOKEN:
        raise RuntimeError(
            "Turso is not configured. Set TURSO_DATABASE_URL "
            "and TURSO_AUTH_TOKEN in the backend environment."
        )

    try:
        import libsql

        return libsql.connect(
            TURSO_URL,
            auth_token=TURSO_TOKEN,
        )

    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the Turso client. "
            "Check the installed libsql package and Turso configuration."
        ) from exc


# =========================================================
# CREATE DATABASE TABLE
# =========================================================

def init_db():
    global db_error

    conn = None

    try:
        conn = connect_db()

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS student_predictions (
                id TEXT PRIMARY KEY,
                curricular_credited REAL NOT NULL,
                curricular_enrolled REAL NOT NULL,
                curricular_evaluations REAL NOT NULL,
                curricular_approved REAL NOT NULL,
                curricular_grade REAL NOT NULL,
                tuition_fees_up_to_date INTEGER NOT NULL,
                scholarship_holder INTEGER NOT NULL,
                debtor INTEGER NOT NULL,
                age_at_enrollment INTEGER NOT NULL,
                admission_grade REAL NOT NULL,
                gender INTEGER NOT NULL,
                application_mode INTEGER NOT NULL,
                course INTEGER NOT NULL,
                predicted_outcome TEXT NOT NULL,
                class_probabilities TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.commit()
        db_error = None

        return True

    except Exception as exc:
        db_error = str(exc)
        return False

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# =========================================================
# LOAD THE EXISTING TRAINED MODEL
# No retraining or hyperparameter changes.
# =========================================================

@app.on_event("startup")
def startup():
    global model
    global model_error
    global model_classes
    global model_has_probabilities

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}"
            )

        model = joblib.load(MODEL_PATH)

        model_classes = [
            value.item() if hasattr(value, "item") else value
            for value in getattr(model, "classes_", [])
        ]

        model_has_probabilities = callable(
            getattr(model, "predict_proba", None)
        )

        model_error = None

    except Exception as exc:
        model = None

        model_error = (
            f"Model could not be loaded ({type(exc).__name__}). "
            "Check the model file and dependency versions."
        )

    init_db()


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():
    return {
        "name": "StudentLens AI API",
        "status": "running",
        "docs": "/docs",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    connected = init_db()

    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "database_connected": connected,
        "model_error": model_error,
        "database_error": (
            None
            if connected
            else "Turso connection unavailable or not configured."
        ),
        "class_labels": model_classes,
        "probabilities_supported": model_has_probabilities,
    }


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.get("/model-info")
def model_info():
    model_type = None

    if model is not None:
        model_type = type(
            model.named_steps.get("model", model)
            if hasattr(model, "named_steps")
            else model
        ).__name__

    return {
        "model_type": model_type,
        "model_loaded": model is not None,
        "class_labels": model_classes,
        "probabilities_supported": model_has_probabilities,
        "features": FEATURES,
        "feature_categories": CATEGORIES,
        "performance_metrics": None,
        "target_label_mapping_verified": False,
    }


# =========================================================
# DATABASE HELPER FUNCTIONS
# =========================================================

def serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    probabilities = row.get("class_probabilities")

    if probabilities:
        try:
            row["class_probabilities"] = json.loads(probabilities)
        except (json.JSONDecodeError, TypeError):
            pass

    return row


def fetch_all(conn, sql, params=()):
    cursor = conn.execute(sql, params)

    names = (
        [column[0] for column in cursor.description]
        if cursor.description
        else []
    )

    return [
        dict(zip(names, row))
        for row in cursor.fetchall()
    ]


# =========================================================
# MAKE A PREDICTION
# =========================================================

@app.post("/predict")
def predict(payload: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The prediction model is unavailable. "
                "Check backend model configuration."
            ),
        )

    features = payload.as_features()

    # Validate Application mode against the supported training categories.
    if (
        features["Application mode"]
        not in CATEGORIES["Application mode"]
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported value for Application mode. "
                "Choose a category supported by the trained model."
            ),
        )

    # Course is intentionally NOT checked against a predefined list.
    # PredictionInput already requires Course to be a positive integer.

    # Preserve the original training feature order.
    frame = pd.DataFrame(
        [[features[name] for name in FEATURES]],
        columns=FEATURES,
    )

    # Run the existing saved model.
    try:
        predicted = model.predict(frame)[0]

        if hasattr(predicted, "item"):
            predicted = predicted.item()

        probabilities = None

        if model_has_probabilities:
            values = model.predict_proba(frame)[0]

            probabilities = {
                str(
                    class_value.item()
                    if hasattr(class_value, "item")
                    else class_value
                ): float(probability)
                for class_value, probability in zip(
                    model_classes,
                    values,
                )
            }

    except Exception:
        raise HTTPException(
            status_code=422,
            detail=(
                "The supplied values could not be processed by the "
                "saved model. The Course code may be unsupported by "
                "the model's preprocessing. Verify model compatibility "
                "and the input values."
            ),
        ) from None

    # =====================================================
    # SAVE THE PREDICTION TO TURSO
    # =====================================================

    prediction_id = str(uuid.uuid4())

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    conn = None

    try:
        conn = connect_db()

        conn.execute(
            """
            INSERT INTO student_predictions (
                id,
                curricular_credited,
                curricular_enrolled,
                curricular_evaluations,
                curricular_approved,
                curricular_grade,
                tuition_fees_up_to_date,
                scholarship_holder,
                debtor,
                age_at_enrollment,
                admission_grade,
                gender,
                application_mode,
                course,
                predicted_outcome,
                class_probabilities,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                features[FEATURES[0]],
                features[FEATURES[1]],
                features[FEATURES[2]],
                features[FEATURES[3]],
                features[FEATURES[4]],
                features[FEATURES[5]],
                features[FEATURES[6]],
                features[FEATURES[7]],
                features[FEATURES[8]],
                features[FEATURES[9]],
                features[FEATURES[10]],
                features[FEATURES[11]],
                features[FEATURES[12]],
                str(predicted),
                (
                    json.dumps(probabilities)
                    if probabilities is not None
                    else None
                ),
                created_at,
            ),
        )

        conn.commit()

    except Exception:
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Prediction was calculated, but it could not "
                    "be saved. Check Turso connectivity."
                ),
                "prediction": str(predicted),
                "probabilities": probabilities,
                "saved": False,
            },
        ) from None

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    return {
        "success": True,
        "prediction_id": prediction_id,
        "predicted_outcome": str(predicted),
        "class_probabilities": probabilities,
        "probabilities_supported": probabilities is not None,
        "saved": True,
        "created_at": created_at,
        "disclaimer": (
            "This is a model estimate, not a guaranteed academic outcome."
        ),
    }


# =========================================================
# GET PREDICTION HISTORY
# =========================================================

@app.get("/predictions")
def predictions(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    conn = None

    try:
        conn = connect_db()

        rows = fetch_all(
            conn,
            """
            SELECT *
            FROM student_predictions
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        count_rows = fetch_all(
            conn,
            """
            SELECT COUNT(*) AS total
            FROM student_predictions
            """,
        )

        total = count_rows[0]["total"]

        return {
            "items": [
                serialize_row(row)
                for row in rows
            ],
            "limit": limit,
            "offset": offset,
            "total": total,
        }

    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction history is temporarily unavailable. "
                "Check Turso connectivity."
            ),
        ) from None

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# =========================================================
# GET ONE PREDICTION
# =========================================================

@app.get("/predictions/{prediction_id}")
def prediction_detail(prediction_id: str):
    conn = None

    try:
        conn = connect_db()

        rows = fetch_all(
            conn,
            """
            SELECT *
            FROM student_predictions
            WHERE id = ?
            """,
            (prediction_id,),
        )

        if not rows:
            raise HTTPException(
                status_code=404,
                detail="Prediction record not found.",
            )

        return serialize_row(rows[0])

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction details are temporarily unavailable."
            ),
        ) from None

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# =========================================================
# ANALYTICS
# =========================================================

@app.get("/analytics")
def analytics():
    conn = None

    try:
        conn = connect_db()

        total_rows = fetch_all(
            conn,
            """
            SELECT COUNT(*) AS total
            FROM student_predictions
            """,
        )

        total = total_rows[0]["total"]

        groups = fetch_all(
            conn,
            """
            SELECT
                predicted_outcome,
                COUNT(*) AS count
            FROM student_predictions
            GROUP BY predicted_outcome
            ORDER BY count DESC
            """,
        )

        daily = fetch_all(
            conn,
            """
            SELECT
                substr(created_at, 1, 10) AS date,
                COUNT(*) AS count
            FROM student_predictions
            GROUP BY substr(created_at, 1, 10)
            ORDER BY date ASC
            """,
        )

        return {
            "total_predictions": total,
            "outcome_counts": groups,
            "outcome_distribution": groups,
            "daily_trend": daily,
        }

    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                "Dashboard analytics are temporarily unavailable. "
                "Check Turso connectivity."
            ),
        ) from None

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass