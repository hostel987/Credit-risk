from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from scipy.special import expit

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles


# =========================================================
# GLOBAL MODEL STORAGE
# =========================================================

ml_model = {}


# =========================================================
# PORTABLE CALIBRATED XGBOOST MODEL
# =========================================================

class PortableCalibratedXGB:

    def __init__(self, artifact):

        # -------------------------------------------------
        # Check model format
        # -------------------------------------------------

        if artifact.get("format") != (
            "portable_calibrated_pipeline_xgb_v1"
        ):
            raise ValueError(
                "Invalid credit_risk_model.pkl format."
            )

        if artifact.get("method") != "sigmoid":
            raise ValueError(
                "This loader expects sigmoid calibration."
            )

        # -------------------------------------------------
        # Classes
        # -------------------------------------------------

        self.classes_ = np.asarray(
            artifact["classes"]
        )

        # -------------------------------------------------
        # Load all 5 folds
        # -------------------------------------------------

        self.folds = []

        for fold_number, fold in enumerate(
            artifact["folds"]
        ):

            print(
                f"Loading model fold {fold_number + 1}..."
            )

            # ---------------------------------------------
            # Create empty XGBoost Booster
            # ---------------------------------------------

            booster = xgb.Booster()

            # ---------------------------------------------
            # Load native XGBoost JSON bytes
            # ---------------------------------------------

            booster.load_model(
                bytearray(
                    fold["model_bytes"]
                )
            )

            # ---------------------------------------------
            # Save everything needed for prediction
            # ---------------------------------------------

            self.folds.append({

                "preprocessor":
                    fold["preprocessor"],

                "booster":
                    booster,

                "a":
                    float(fold["a"]),

                "b":
                    float(fold["b"]),

                "feature_names":
                    fold.get(
                        "feature_names"
                    )
            })

        # -------------------------------------------------
        # Safety check
        # -------------------------------------------------

        if len(self.folds) == 0:
            raise ValueError(
                "No model folds found."
            )

        print(
            f"Loaded {len(self.folds)} calibrated folds."
        )


    # =====================================================
    # PREDICT ONE FOLD
    # =====================================================

    def _predict_fold(
        self,
        fold,
        X
    ):

        # -------------------------------------------------
        # STEP 1:
        # Apply the SAME preprocessing used during training
        # -------------------------------------------------

        preprocessor = (
            fold["preprocessor"]
        )

        if preprocessor is not None:

            X_transformed = (
                preprocessor.transform(X)
            )

        else:

            X_transformed = X


        # -------------------------------------------------
        # STEP 2:
        # Get XGBoost model
        # -------------------------------------------------

        booster = fold["booster"]


        # -------------------------------------------------
        # STEP 3:
        # Check number of features
        # -------------------------------------------------

        expected_features = (
            booster.num_features()
        )

        actual_features = (
            X_transformed.shape[1]
        )

        if expected_features != actual_features:

            raise ValueError(
                "Feature count mismatch.\n"
                f"XGBoost expects "
                f"{expected_features} features, "
                f"but preprocessing produced "
                f"{actual_features} features."
            )


        # -------------------------------------------------
        # STEP 4:
        # Create DMatrix
        #
        # IMPORTANT:
        # Do NOT pass the original feature names here.
        # ColumnTransformer may have changed/expanded them.
        # -------------------------------------------------

        dmatrix = xgb.DMatrix(
            X_transformed
        )


        # -------------------------------------------------
        # STEP 5:
        # Get raw XGBoost probability
        # -------------------------------------------------

        probability = np.asarray(
            booster.predict(
                dmatrix
            )
        ).reshape(-1)

        # STEP 6: Keep probability inside (0, 1)
        eps = np.finfo(np.float64).eps
        probability = np.clip(probability, eps, 1.0 - eps)

        # STEP 7: Apply sigmoid calibration directly on the probability
        # (sklearn fits the sigmoid on predict_proba output for XGBoost,
        #  not on logits)
        calibrated_probability = expit(
            -(fold["a"] * probability + fold["b"])
        )

        return calibrated_probability


    # =====================================================
    # PREDICT PROBABILITY
    # =====================================================

    def predict_proba(self, X):

        fold_probabilities = []


        # -------------------------------------------------
        # Predict using every calibrated fold
        # -------------------------------------------------

        for fold in self.folds:

            positive_probability = (
                self._predict_fold(
                    fold,
                    X
                )
            )


            # Binary classification:
            #
            # column 0 = probability of class 0
            # column 1 = probability of class 1

            probabilities = np.column_stack([
                1.0 - positive_probability,
                positive_probability
            ])


            fold_probabilities.append(
                probabilities
            )


        # -------------------------------------------------
        # CalibratedClassifierCV averages the probabilities
        # from the 5 fitted calibrated models.
        # -------------------------------------------------

        return np.mean(
            fold_probabilities,
            axis=0
        )


    # =====================================================
    # PREDICT CLASS
    # =====================================================

    def predict(self, X):

        probabilities = (
            self.predict_proba(X)
        )


        class_indices = np.argmax(
            probabilities,
            axis=1
        )


        return self.classes_[
            class_indices
        ]


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = (
    Path(__file__).resolve().parent
)

MODEL_PATH = (
    BASE_DIR /
    "credit_risk_model.pkl"
)

THRESHOLD_PATH = (
    BASE_DIR /
    "new_best_threshold.pkl"
)


# =========================================================
# FASTAPI LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print()
    print("========================================")
    print("Starting Credit Risk API")
    print("========================================")


    # -----------------------------------------------------
    # Check files
    # -----------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():

        raise FileNotFoundError(
            f"Threshold file not found:\n{THRESHOLD_PATH}"
        )


    # -----------------------------------------------------
    # Load portable model artifact
    # -----------------------------------------------------

    print("Loading credit risk model...")

    artifact = joblib.load(
        MODEL_PATH
    )


    # -----------------------------------------------------
    # Reconstruct model
    # -----------------------------------------------------

    ml_model["model"] = (
        PortableCalibratedXGB(
            artifact
        )
    )


    # -----------------------------------------------------
    # Load threshold
    # -----------------------------------------------------

    print("Loading threshold...")

    threshold = joblib.load(
        THRESHOLD_PATH
    )

    threshold = float(
        threshold
    )


    # -----------------------------------------------------
    # Validate threshold
    # -----------------------------------------------------

    if not 0.0 <= threshold <= 1.0:

        raise ValueError(
            f"Invalid threshold: {threshold}. "
            "Threshold must be between 0 and 1."
        )


    ml_model["threshold"] = (
        threshold
    )


    # -----------------------------------------------------
    # Startup successful
    # -----------------------------------------------------

    print()
    print("========================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("========================================")
    print(
        f"Threshold: {threshold}"
    )
    print(
        f"Number of folds: "
        f"{len(ml_model['model'].folds)}"
    )
    print("========================================")
    print()


    yield


    # -----------------------------------------------------
    # Shutdown
    # -----------------------------------------------------

    ml_model.clear()

    print(
        "Credit risk model unloaded."
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0",
    lifespan=lifespan
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# INPUT MODEL
# =========================================================

class LoanApplication(BaseModel):

    person_age: int

    person_income: float

    person_home_ownership: str

    person_emp_length: float

    loan_intent: str

    loan_grade: str

    loan_amnt: float

    loan_int_rate: float

    loan_percent_income: float

    cb_person_default_on_file: str

    cb_person_cred_hist_length: int


# =========================================================
# PREDICT ENDPOINT
# =========================================================

@app.post("/predict")
def predict(
    data: LoanApplication
):

    try:

        # -------------------------------------------------
        # Convert Pydantic object -> dictionary
        # -------------------------------------------------

        if hasattr(
            data,
            "model_dump"
        ):

            input_data = (
                data.model_dump()
            )

        else:

            input_data = (
                data.dict()
            )


        # -------------------------------------------------
        # Create DataFrame
        # -------------------------------------------------

        input_df = pd.DataFrame([
            input_data
        ])


        print()
        print("INPUT:")
        print(input_df)


        # -------------------------------------------------
        # Get calibrated probability
        # -------------------------------------------------

        probabilities = (
            ml_model["model"]
            .predict_proba(
                input_df
            )
        )


        probability = float(
            probabilities[0][1]
        )


        # -------------------------------------------------
        # Get threshold
        # -------------------------------------------------

        threshold = float(
            ml_model["threshold"]
        )


        # -------------------------------------------------
        # Make final prediction
        # -------------------------------------------------

        prediction = int(
            probability >= threshold
        )


        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        result = {

            "default_probability":
                round(
                    probability,
                    6
                ),

            "default_prediction":
                prediction,

            "threshold":
                round(
                    threshold,
                    6
                ),

            "Result":
                (
                    "High Risk"
                    if prediction == 1
                    else "Low Risk"
                )
        }


        print()
        print("RESULT:")
        print(result)
        print()


        return result


    except Exception as e:

        # -------------------------------------------------
        # Print the REAL error in terminal
        # -------------------------------------------------

        import traceback

        print()
        print(
            "========================================"
        )
        print(
            "PREDICTION ERROR"
        )
        print(
            "========================================"
        )

        traceback.print_exc()

        print(
            "========================================"
        )
        print()


        raise


# =========================================================
# HOME PAGE
# =========================================================

STATIC_DIR = BASE_DIR / "static"

if not (STATIC_DIR / "index.html").exists():
    raise RuntimeError(f"index.html not found in {STATIC_DIR}")

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


# =========================================================
# OPTIONAL STATIC FRONTEND
# =========================================================

# Uncomment ONLY if you have:
#
# static/
#    index.html
#
# app.mount(
#     "/",
#     StaticFiles(
#         directory="static",
#         html=True
#     ),
#     name="static"
# )