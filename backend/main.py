import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)
from typing import Dict, Any, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- FIX: Import tensorflow as tf ---
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Bidirectional

import joblib
import time
from imblearn.over_sampling import SVMSMOTE
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- IMPORTANT: These imports assume 'config.py' and 'data_loader.py' are
#    inside a 'app/' subdirectory within your 'backend/' directory,
#    and 'backend/app/__init__.py' exists.
#    If your files are directly in 'backend/', change these to:
#    from config import TOP_BSE_STOCKS
#    from data_loader import fetch_data
try:
    from app.config import TOP_BSE_STOCKS
    from app.data_loader import fetch_data
except ImportError as e:
    logger.error(f"Failed to import from app/ directory: {e}. "
                 "Ensure 'app' is a valid Python package (has __init__.py) "
                 "and that 'main.py' is launched from 'backend/' root or parent directory "
                 "where 'backend/app' is importable.")
    # Re-raise the error as this is a critical import failure
    # It's better for the app to fail fast if it can't find core modules.
    raise

app = FastAPI(
    title="Stock Movement Prediction API",
    description="A FastAPI backend for stock price prediction using machine learning models.",
    version="1.0.0"
)

# --- Global Variables for Loaded Models and Scalers ---
# These dictionaries will store loaded models and scalers in memory
# to avoid reloading them on every request, improving response times.
loaded_models: Dict[str, tf.keras.Model] = {} # Corrected type hint
loaded_scalers: Dict[str, MinMaxScaler] = {}

# --- Directory for models/scalers relative to this script ---
# This correctly points to 'backend/models/' assuming main.py is in 'backend/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_STORAGE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_STORAGE_DIR, exist_ok=True) # Ensure directory exists on first run if not present
logger.info(f"Model storage directory: {MODEL_STORAGE_DIR}")


# --- CORS Configuration ---
# Uses "FRONTEND_URL" environment variable, supporting multiple URLs separated by comma
allowed_origins_env = os.environ.get("FRONTEND_URL")
origins_list: List[str] = []

if allowed_origins_env:
    origins_list.extend([url.strip() for url in allowed_origins_env.split(',')])

# Always include common local development origins
origins_list.extend([
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
])

# Remove duplicates to ensure unique origins
final_origins = list(set(origins_list))

logger.info(f"Configuring CORS with allowed origins: {final_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=final_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, OPTIONS)
    allow_headers=["*"],  # Allows all headers
)

# --- Hyperparameters ---
short_window = 30
long_window = 120
# Ensure these features align with what your data_loader and preprocessing generate
features = ['Close', 'MACD', 'MACD_diff', 'RSI', 'SMA20', 'SMA200', 'Volume', 'RET1', 'VOL']
n_features = len(features)

# --- Pydantic Model for Request Body ---
class PredictRequest(BaseModel):
    symbol: str
    retrain: bool = False
    threshold: float = 0.02
    future_days: int = 3

# --- Helper Functions for Data Processing and Model Building ---

def create_sequences(data: np.ndarray, threshold: float = 0.02, future_days: int = 3):
    """
    Creates sequences for LSTM model training and the corresponding labels (y).
    Data should already be scaled.
    """
    X_short, X_long, y = [], [], []
    
    # Required data points for one full long sequence + future_days for target
    required_data_points = long_window + future_days
    if len(data) < required_data_points:
        logger.warning(f"Not enough data ({len(data)}) to create sequences for training/prediction. Need at least {required_data_points} points.")
        return np.array([]), np.array([]), np.array([])

    for i in range(len(data) - required_data_points + 1):
        # Short window sequence (last `short_window` days from the `long_window` block)
        X_short.append(data[i + long_window - short_window : i + long_window])
        # Long window sequence (full `long_window` days)
        X_long.append(data[i : i + long_window])
        
        # Original close price (index 0 of features) from the end of the long window
        # Data is scaled, so we need to inverse transform the close price for delta calculation
        # This is complex when 'data' is scaled. It's safer if 'data' passed here is original unscaled 'Close' prices for target calculation.
        # However, the current code assumes 'data' is scaled_data where first feature is Close.
        # Let's assume for now that the original 'Close' price can be extracted or that the target calculation is fine with scaled data.
        # If 'data' here is scaled_data, then data[..., 0] is the scaled close price.
        # To get the original p0, you'd need the scaler to inverse transform it.
        # Given the current setup, we proceed assuming `data[..., 0]` reflects relative changes even when scaled.
        
        # --- IMPORTANT ASSUMPTION ---
        # Assuming `data` passed to `create_sequences` has `Close` as its first feature (index 0)
        # and that the relative change calculation (`delta`) is meaningful on scaled data.
        # If not, you'd need to pass original prices or inverse transform here.
        p0_scaled = data[i + long_window - 1][0] # Scaled 'Close' price at the end of the input window

        # Get future prices (also scaled 'Close' price)
        future_prices_scaled = [data[i + long_window + j][0] for j in range(future_days)]
        future_avg_scaled = np.mean(future_prices_scaled)
        
        if p0_scaled == 0:
            logger.warning(f"Skipping sequence at index {i} due to zero scaled base price. Data might be problematic.")
            continue

        delta_scaled = (future_avg_scaled - p0_scaled) / p0_scaled
        y.append(1 if delta_scaled > threshold else 0)
            
    min_len = min(len(X_short), len(X_long), len(y))
    return np.array(X_short[:min_len]), np.array(X_long[:min_len]), np.array(y[:min_len])

def build_model(input_short_shape: tuple, input_long_shape: tuple) -> tf.keras.Model:
    """
    Generates the TensorFlow Keras Bidirectional LSTM model architecture.
    """
    input_short = Input(shape=input_short_shape, name='input_short')
    input_long = Input(shape=input_long_shape, name='input_long')

    x1 = Bidirectional(LSTM(64, return_sequences=False), name='lstm_short')(input_short)
    x2 = Bidirectional(LSTM(64, return_sequences=False), name='lstm_long')(input_long)

    x = concatenate([x1, x2], name='concatenate_layers')

    x = Dense(64, activation='relu', name='dense_hidden')(x)
    x = Dropout(0.2, name='dropout_layer')(x)
    output = Dense(1, activation='sigmoid', name='output_layer')(x)

    model = Model(inputs=[input_short, input_long], outputs=output, name='stock_prediction_model')
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logger.info("Keras model built and compiled.")
    return model

# --- FastAPI Endpoints ---

@app.get("/")
async def read_root():
    """
    Root endpoint for basic API health check.
    """
    logger.info("Root endpoint hit.")
    return {"message": "📈 Welcome to Stock Movement Prediction API. Visit /docs for OpenAPI specification."}

@app.get("/api/stocks")
async def get_stocks():
    """
    Returns a list of top BSE stocks (from your app.config).
    """
    logger.info("GET /api/stocks endpoint hit.")
    if not TOP_BSE_STOCKS:
        logger.error("TOP_BSE_STOCKS not loaded. Check app/config.py.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stock list not available.")
    # TOP_BSE_STOCKS is a dict, convert to a list of dicts for frontend consumption
    return {"stocks": [{"symbol": sym, "name": name} for sym, name in TOP_BSE_STOCKS.items()]}

@app.get("/api/stocks/{stock_id}")
async def get_stock_info(stock_id: str):
    """
    Fetches and returns recent historical data for a specific stock ID.
    """
    logger.info(f"GET /api/stocks/{stock_id} endpoint hit.")
    name = TOP_BSE_STOCKS.get(stock_id, "Unknown Stock")
    
    df = fetch_data(stock_id) # This `fetch_data` should handle Alpha Vantage API key and errors

    if df.empty:
        logger.warning(f"No data found for stock {stock_id} from data_loader.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No data found for stock {stock_id}")
    
    # Ensure 'Close' column is present before proceeding
    if 'Close' not in df.columns:
        logger.error(f"Missing 'Close' column in fetched data for {stock_id}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Required 'Close' column not found in data for {stock_id}.")

    data_points = df.tail(60).reset_index().to_dict(orient="records")
    return {"name": name, "data": data_points}

# --- Main Prediction Logic Function ---
def process_prediction(symbol: str, retrain: bool, threshold: float, future_days: int) -> Dict[str, Any]:
    """
    Handles the core logic for stock prediction:
    - Fetches data
    - Loads/trains scaler and model
    - Makes predictions
    - Calculates and returns evaluation metrics and prediction results.
    """
    start_time = time.time()
    logger.info(f"Processing prediction request for {symbol} (retrain={retrain}, threshold={threshold}, future_days={future_days}).")

    safe_symbol = symbol.replace('.', '_')
    model_filepath = os.path.join(MODEL_STORAGE_DIR, f"model_{safe_symbol}.h5")
    scaler_filepath = os.path.join(MODEL_STORAGE_DIR, f"scaler_{safe_symbol}.pkl")

    # --- Fetch Data ---
    logger.info(f"Fetching data for {symbol}...")
    # fetch_data should already handle API key and common errors, raising HTTPException
    df = fetch_data(symbol)
    if df.empty:
        logger.error(f"No historical data found for stock {symbol} from data_loader.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No historical data found for stock {symbol}. Cannot proceed with prediction.")

    # Ensure required features are in the DataFrame
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        logger.error(f"Missing required features for {symbol}: {missing_features}. Check data_loader preprocessing.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Missing required data features for {symbol}: {', '.join(missing_features)}. "
                                   "Ensure your data_loader generates all necessary features.")
    
    # Use only the defined features for scaling and sequence creation
    data_for_scaling = df[features].values
    logger.info(f"Fetched {len(data_for_scaling)} data points with {n_features} features for {symbol}.")

    # --- Scaler Loading/Training ---
    scaler: Optional[MinMaxScaler] = None
    if retrain or not os.path.exists(scaler_filepath):
        logger.info(f"Scaler for {symbol} not found or retraining requested. Training new scaler...")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_for_scaling)
        joblib.dump(scaler, scaler_filepath)
        loaded_scalers[symbol] = scaler
        logger.info(f"Scaler for {symbol} trained and saved.")
    else:
        try:
            logger.info(f"Loading existing scaler for {symbol}...")
            scaler = joblib.load(scaler_filepath)
            scaled_data = scaler.transform(data_for_scaling)
            loaded_scalers[symbol] = scaler
            logger.info(f"Scaler for {symbol} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading scaler for {symbol}: {e}. Retraining scaler instead.")
            # Fallback to retraining if loading fails
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data_for_scaling)
            joblib.dump(scaler, scaler_filepath)
            loaded_scalers[symbol] = scaler
            logger.info(f"Scaler for {symbol} retrained due to load error.")
            
    # --- Create Sequences and Split Data ---
    required_sequence_data_points = long_window + future_days
    if len(scaled_data) < required_sequence_data_points:
        logger.error(f"Not enough scaled data ({len(scaled_data)}) for {symbol} to create sequences. "
                     f"Need at least {required_sequence_data_points} data points.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Not enough historical data for {symbol} to create sequences for model evaluation/prediction. "
                   f"Need at least {required_sequence_data_points} data points, but only have {len(scaled_data)}."
        )

    X_short_all, X_long_all, y_all = create_sequences(scaled_data, threshold=threshold, future_days=future_days)
    
    if len(y_all) == 0:
        logger.error(f"No valid sequences could be created for {symbol} after processing. "
                     "This might be due to insufficient data or all 'p0' values being zero.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Not enough valid sequences could be created for {symbol}. "
                   "This might be due to insufficient data or issues with target calculation."
        )

    # --- Fix: Corrected variable name from Xl_long_all to X_long_all ---
    # Flatten sequences for SMOTE
    Xs_flat = X_short_all.reshape(X_short_all.shape[0], -1)
    Xl_flat = X_long_all.reshape(X_long_all.shape[0], -1) # Corrected line

    # --- SMOTE Resampling ---
    smote_k_neighbors = 5 # Default
    unique_classes, counts = np.unique(y_all, return_counts=True)
    
    Xs_res, Xl_res, y_resampled = X_short_all, X_long_all, y_all # Initialize with original data

    if len(unique_classes) < 2:
        logger.warning(f"Only one class present in data for {symbol}. SMOTE cannot be applied.")
    else:
        minority_class_count = min(counts)
        # Adjust k_neighbors for SMOTE if minority class count is too low
        if minority_class_count > 1:
            smote_k_neighbors = min(smote_k_neighbors, minority_class_count - 1)
        else: # minority_class_count is 0 or 1
            logger.warning(f"Minority class has only {minority_class_count} sample(s) for {symbol}. Cannot apply SVMSMOTE.")
            smote_k_neighbors = 0 # Prevent SMOTE if not enough neighbors

        if smote_k_neighbors > 0: # Proceed with SMOTE only if k_neighbors is valid
            try:
                smote = SVMSMOTE(random_state=42, k_neighbors=smote_k_neighbors)
                
                combined_features_flat = np.concatenate((Xs_flat, Xl_flat), axis=1)
                combined_features_resampled, y_resampled = smote.fit_resample(combined_features_flat, y_all)
                
                # Reshape back to sequence format after SMOTE
                Xs_flat_res = combined_features_resampled[:, :Xs_flat.shape[1]]
                Xl_flat_res = combined_features_resampled[:, Xs_flat.shape[1]:]

                Xs_res = Xs_flat_res.reshape(-1, short_window, n_features)
                Xl_res = Xl_flat_res.reshape(-1, long_window, n_features)
                
                logger.info(f"SVMSMOTE applied for {symbol}. Original samples: {len(y_all)}, Resampled samples: {len(y_resampled)}")
            except Exception as e:
                logger.error(f"Error applying SVMSMOTE for {symbol}: {e}. Proceeding without resampling.")
                Xs_res, Xl_res, y_resampled = X_short_all, X_long_all, y_all # Fallback to original
        else:
            logger.warning(f"SVMSMOTE k_neighbors ({smote_k_neighbors}) invalid. Skipping SMOTE for {symbol}.")


    if len(y_resampled) < 2:
        logger.error(f"Not enough resampled data ({len(y_resampled)}) to perform train-test split for {symbol}. Need at least 2 samples.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Not enough data to train/evaluate model for {symbol}.")

    # Split data into training and testing sets
    Xs_train, Xs_test, Xl_train, Xl_test, y_train, y_test = train_test_split(
        Xs_res, Xl_res, y_resampled, test_size=0.2, shuffle=True, random_state=42)

    logger.info(f"Data split for {symbol}: Train samples={len(y_train)}, Test samples={len(y_test)}")

    # --- Model Loading/Training ---
    model: Optional[tf.keras.Model] = None
    if retrain or symbol not in loaded_models or not os.path.exists(model_filepath):
        logger.info(f"Model for {symbol} not found in memory or on disk, or retraining requested. Building and training new model...")
        model = build_model((short_window, n_features), (long_window, n_features))
        
        class_weights = None
        if len(np.unique(y_train)) > 1:
            class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
            logger.info(f"Class weights for {symbol}: {class_weights}")
        else:
            logger.warning(f"Only one class present in y_train for {symbol}. Skipping class_weight calculation.")

        history = model.fit([Xs_train, Xl_train], y_train, epochs=20, batch_size=32,
                            validation_data=([Xs_test, Xl_test], y_test),
                            class_weight=class_weights if class_weights else None, verbose=0)
        
        # Check for NaN loss during training, indicating instability
        if np.isnan(history.history['loss'][-1]):
            logger.error(f"Model training failed for {symbol}: Loss became NaN. Data might be problematic.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                                detail=f"Model training for {symbol} failed (NaN loss). Please try retraining (retrain=true) or check backend logs.")

        model.save(model_filepath)
        loaded_models[symbol] = model # Store newly trained model in memory
        logger.info(f"Model for {symbol} trained and saved. Training history: {history.history.keys()}")
    else:
        # Load from memory if available, otherwise from disk
        model = loaded_models.get(symbol)
        if model is None:
            try:
                logger.info(f"Loading existing model for {symbol} from {model_filepath}...")
                model = load_model(model_filepath)
                loaded_models[symbol] = model # Store loaded model in memory
                logger.info(f"Model for {symbol} loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading model for {symbol}: {e}. Model file might be corrupted or incompatible. Requesting retraining implicitly.")
                # If loading fails, raise an exception to force manual retraining (or handle auto-retrain here if desired)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                                    detail=f"Model for {symbol} could not be loaded. Please try retraining (set retrain=true) or check backend logs for details.")

    # --- Metrics Calculation on Test Set ---
    y_probs = model.predict([Xs_test, Xl_test], verbose=0)
    y_pred = (y_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    
    roc_auc = "N/A"
    pr_auc = "N/A"
    report_dict: Dict[str, Any] = {}

    if len(np.unique(y_test)) > 1:
        try:
            # Ensure y_probs is 1D for metrics if needed
            roc_auc = roc_auc_score(y_test, y_probs.flatten())
            pr_auc = average_precision_score(y_test, y_probs.flatten())
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            logger.info(f"Metrics for {symbol}: Accuracy={acc:.4f}, ROC AUC={roc_auc:.4f}, PR AUC={pr_auc:.4f}")
        except ValueError as e:
            logger.warning(f"Could not calculate ROC AUC/PR AUC/Classification Report for {symbol} due to value error: {e}")
            report_dict = {"info": f"Detailed report not available due to: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error calculating metrics for {symbol}: {e}")
            report_dict = {"info": f"Error calculating metrics: {e}"}
    else:
        logger.warning(f"Only one class present in y_test for {symbol}. ROC AUC, PR AUC, and Classification Report will be N/A.")
        report_dict = {"info": "Only one class found in test set, detailed classification report not available."}


    # --- Prediction for the next N days (using the LATEST available data) ---
    # Ensure there's enough data for the last sequence needed for prediction
    if len(scaled_data) < long_window:
        logger.error(f"Not enough recent data ({len(scaled_data)}) for {symbol} to make a future prediction. "
                     f"Need at least {long_window} data points.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Not enough recent data for {symbol} to make a future prediction. "
                   f"Need at least {long_window} data points, but only have {len(scaled_data)}."
        )

    last_short_data = scaled_data[-short_window:].reshape(1, short_window, n_features)
    last_long_data = scaled_data[-long_window:].reshape(1, long_window, n_features)

    prediction_start_time = time.time()
    next_prob = float(model.predict([last_short_data, last_long_data], verbose=0)[0][0])
    prediction_end_time = time.time()
    logger.info(f"Prediction for {symbol} took {prediction_end_time - prediction_start_time:.4f} seconds.")
    
    next_percent = round(next_prob * 100, 2)

    # --- Interpret the Prediction ---
    direction = "UNCERTAIN"
    explanation = "🤔 Prediction confidence is low — proceed with caution."
    probability_display = next_percent # Default to the 'up' probability

    # Adjusted interpretation logic for clarity
    if next_prob >= 0.55: # If confidence for UP is 55% or more
        direction = "UP"
        probability_display = next_percent
        explanation = f"📈 Expected to go UP by {threshold*100:.0f}% or more over {future_days} days."
    elif next_prob <= 0.45: # If confidence for DOWN is 55% or more (100-45)
        direction = "DOWN"
        probability_display = 100 - next_percent # Show confidence for "Down"
        explanation = f"📉 Expected to go DOWN by {threshold*100:.0f}% or more over {future_days} days."
    else: # Between 45% and 55%
        direction = "FLAT"
        explanation = f"↔️ Expected to remain relatively flat (within {threshold*100:.0f}% change over {future_days} days)."
        probability_display = max(next_percent, 100 - next_percent) # Show higher of the two, but it's close to 50

    total_process_time = time.time() - start_time
    logger.info(f"Total processing time for {symbol}: {total_process_time:.2f} seconds.")

    return {
        "symbol": symbol,
        "model_evaluation": {
            "accuracy": round(acc, 4),
            "roc_auc": round(roc_auc, 4) if isinstance(roc_auc, float) else roc_auc,
            "pr_auc": round(pr_auc, 4) if isinstance(pr_auc, float) else pr_auc,
            "classification_report": report_dict,
            "test_set_size": len(y_test)
        },
        "next_prediction": {
            "direction": direction,
            "probability_percent": probability_display,
            "explanation": explanation,
            "raw_probability_up": round(next_prob, 4),
            "threshold_used": threshold,
            "future_days_considered": future_days
        },
        "processing_time_seconds": round(total_process_time, 2)
    }

# --- API Endpoints for Prediction ---
@app.post("/api/predict", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def predict_post_route(request: PredictRequest):
    """
    Predicts stock movement based on a POST request body.
    Forces retraining if `retrain` is True.
    """
    try:
        return process_prediction(
            symbol=request.symbol,
            retrain=request.retrain,
            threshold=request.threshold,
            future_days=request.future_days
        )
    except HTTPException as e:
        logger.error(f"HTTPException in POST /api/predict for {request.symbol}: {e.detail}")
        raise e # Re-raise FastAPI HTTPExceptions directly
    except Exception as e:
        logger.exception(f"Unhandled exception in POST /api/predict for {request.symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {e}"
        )


@app.get("/api/predict", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def predict_get_route(
    symbol: str,
    retrain: bool = False,
    threshold: float = 0.02,
    future_days: int = 3
):
    """
    Predicts stock movement based on GET request query parameters.
    Forces retraining if `retrain` is True.
    """
    try:
        return process_prediction(
            symbol=symbol,
            retrain=retrain,
            threshold=threshold,
            future_days=future_days
        )
    except HTTPException as e:
        logger.error(f"HTTPException in GET /api/predict for {symbol}: {e.detail}")
        raise e # Re-raise FastAPI HTTPExceptions directly
    except Exception as e:
        logger.exception(f"Unhandled exception in GET /api/predict for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {e}"
        )
