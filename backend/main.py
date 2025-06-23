# backend/main.py

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
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Bidirectional
import joblib
import time
from imblearn.over_sampling import SVMSMOTE
import logging

# --- Logging Configuration ---
# Set up basic logging for visibility in Railway logs
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
    # Exit or raise error if critical imports fail
    raise

app = FastAPI(
    title="Stock Movement Prediction API",
    description="A FastAPI backend for stock price prediction using machine learning models.",
    version="1.0.0"
)

# --- Global Variables for Loaded Models and Scalers ---
# These dictionaries will store loaded models and scalers in memory
# to avoid reloading them on every request, improving response times.
loaded_models: Dict[str, tf.keras.Model] = {}
loaded_scalers: Dict[str, MinMaxScaler] = {}

# --- Directory for models/scalers relative to this script ---
# This correctly points to 'backend/models/' assuming main.py is in 'backend/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_STORAGE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_STORAGE_DIR, exist_ok=True) # Ensure directory exists on first run if not present
logger.info(f"Model storage directory: {MODEL_STORAGE_DIR}")


# --- CORS Configuration ---
# Retrieve allowed origins from environment variable.
# On Railway, you will set FRONTEND_URL in your backend service's environment variables.
# If you have multiple allowed frontend URLs, separate them with commas (e.g., "url1,url2").
allowed_origins_env = os.environ.get("FRONTEND_URL")
origins_list: List[str] = []

if allowed_origins_env:
    # Split by comma and strip whitespace for multiple URLs
    origins_list.extend([url.strip() for url in allowed_origins_env.split(',')])

# Add common local development origins explicitly for local testing.
# These will not apply in production unless explicitly included in FRONTEND_URL.
origins_list.extend([
    "http://localhost:5173",  # Vite default dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",  # Common React dev server
    "http://127.0.0.1:3000",
])

# Remove any duplicate URLs from the list and ensure https for production URLs
final_origins = list(set(origins_list))

logger.info(f"Configuring CORS with allowed origins: {final_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=final_origins,
    allow_credentials=True, # Set to True if your frontend sends cookies, authorization headers, etc.
    allow_methods=["*"],    # Allow all common HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Allow all common HTTP headers
)

# --- Hyperparameters ---
# Define your model's hyperparameters and features here
short_window = 30
long_window = 120
features = ['Close', 'MACD', 'MACD_diff', 'RSI', 'SMA20', 'SMA200', 'Volume', 'RET1', 'VOL']
n_features = len(features) # Number of features used in your model

# --- Pydantic Model for Request Body ---
# Defines the expected structure for a POST request to /api/predict
class PredictRequest(BaseModel):
    symbol: str
    retrain: bool = False # Whether to force retraining the model for this symbol
    threshold: float = 0.02 # Prediction threshold for "up" movement (e.g., 0.02 for 2%)
    future_days: int = 3 # Number of future days to predict over (e.g., predict movement over next 3 days)

# --- Helper Functions for Data Processing and Model Building ---

def create_sequences(data: np.ndarray, threshold: float = 0.02, future_days: int = 3):
    """
    Creates sequences for LSTM model training and the corresponding labels (y).

    Args:
        data (np.ndarray): Scaled historical stock data (2D array, e.g., (timesteps, features)).
        threshold (float): Percentage change threshold for classifying 'up' movement.
        future_days (int): Number of future days to consider for the 'up' movement calculation.

    Returns:
        tuple: (X_short, X_long, y) - NumPy arrays of short sequences, long sequences, and labels.
    """
    X_short, X_long, y = [], [], []
    
    # Ensure there's enough data for at least one long window + future days for y calculation
    # The loop needs to access data up to `i + long_window + future_days - 1`
    required_data_points = long_window + future_days
    if len(data) < required_data_points:
        logger.warning(f"Not enough data ({len(data)}) to create sequences for training/prediction. Need at least {required_data_points} points.")
        return np.array([]), np.array([]), np.array([]) # Return empty arrays if not enough data

    # Iterate through the data to create sequences
    # The loop stops early enough to ensure 'future_days' data points are available for 'y'
    for i in range(len(data) - required_data_points + 1):
        # Short window sequence (e.g., last 30 days of the long window)
        X_short.append(data[i + long_window - short_window : i + long_window])
        # Long window sequence (e.g., last 120 days)
        X_long.append(data[i : i + long_window])
        
        # Assuming 'Close' price is the first feature (index 0)
        p0 = data[i + long_window - 1][0] # Close price at the end of the long window
        
        # Calculate the average price for the 'future_days'
        future_prices = [data[i + long_window + j][0] for j in range(future_days)]
        future_avg = np.mean(future_prices)
        
        if p0 == 0:
            # Skip samples where the base price is zero to avoid division by zero
            # This can happen with very sparse or erroneous data.
            logger.warning(f"Skipping sequence due to zero base price at index {i + long_window - 1}.")
            # To keep X_short, X_long, and y synchronized, we either need to remove the last appended items
            # or continue before appending them. Continuing before appending means lengths won't match
            # if this happens mid-loop without careful handling.
            # A more robust approach if data quality is a concern is to filter the original data
            # before calling create_sequences or handle the `continue` more explicitly by not appending.
            # For now, if we continue, the lists will be out of sync.
            # It's better to ensure `p0` is non-zero as a prerequisite for `fetch_data`.
            continue # This will cause mismatch in lengths of X_short, X_long, y if not handled.
                    # It's generally better to ensure valid `p0` values in the input `data`.

        delta = (future_avg - p0) / p0 # Percentage change
        y.append(1 if delta > threshold else 0) # Label: 1 if price goes up by threshold, 0 otherwise
        
    # After the loop, ensure all lists have the same length due to potential `continue` statements
    # This addresses the issue where `continue` would skip appending to y but not to X_short/X_long.
    # It's safer to build lists conditionally or to prune afterwards.
    min_len = min(len(X_short), len(X_long), len(y))
    return np.array(X_short[:min_len]), np.array(X_long[:min_len]), np.array(y[:min_len])

def build_model(input_short_shape: tuple, input_long_shape: tuple) -> tf.keras.Model:
    """
    Builds and compiles the TensorFlow Keras Bidirectional LSTM model.

    Args:
        input_short_shape (tuple): Shape of the short sequence input (timesteps, features).
        input_long_shape (tuple): Shape of the long sequence input (timesteps, features).

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    # Input layers for short and long sequences
    input_short = Input(shape=input_short_shape, name='input_short')
    input_long = Input(shape=input_long_shape, name='input_long')

    # Bidirectional LSTM layers for each input
    # Using LSTMs with 64 units and return_sequences=False as they lead to a single output for concatenation.
    x1 = Bidirectional(LSTM(64, return_sequences=False), name='lstm_short')(input_short)
    x2 = Bidirectional(LSTM(64, return_sequences=False), name='lstm_long')(input_long)

    # Concatenate the outputs of the two LSTM branches
    x = concatenate([x1, x2], name='concatenate_layers')

    # Dense layers for classification
    x = Dense(64, activation='relu', name='dense_hidden')(x)
    x = Dropout(0.2, name='dropout_layer')(x)
    output = Dense(1, activation='sigmoid', name='output_layer')(x) # Sigmoid for binary classification

    # Create the model with two inputs and one output
    model = Model(inputs=[input_short, input_long], outputs=output, name='stock_prediction_model')
    
    # Compile the model
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
    return {"stocks": TOP_BSE_STOCKS}

@app.get("/api/stocks/{stock_id}")
async def get_stock_info(stock_id: str):
    """
    Fetches and returns recent historical data for a specific stock ID.
    """
    logger.info(f"GET /api/stocks/{stock_id} endpoint hit.")
    name = TOP_BSE_STOCKS.get(stock_id, "Unknown Stock") # Get friendly name
    
    # Fetch data using your data_loader
    df = fetch_data(stock_id)
    
    if df.empty:
        logger.warning(f"No data found for stock {stock_id} from data_loader.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No data found for stock {stock_id}")
    
    # Return last 60 data points as a list of dictionaries, including Date
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

    Args:
        symbol (str): The stock symbol (e.g., 'AAPL').
        retrain (bool): If True, forces retraining of the model and scaler.
        threshold (float): The percentage change threshold for binary classification.
        future_days (int): The number of future days for which to predict the movement.

    Returns:
        Dict[str, Any]: A dictionary containing prediction results and evaluation metrics.
    """
    start_time = time.time()
    logger.info(f"Processing prediction request for {symbol} (retrain={retrain}, threshold={threshold}, future_days={future_days}).")

    # Define file paths for symbol-specific model and scaler
    # Replace '.' in symbol with '_' for valid filenames (e.g., 'NSEI.NS' -> 'NSEI_NS')
    safe_symbol = symbol.replace('.', '_')
    model_filepath = os.path.join(MODEL_STORAGE_DIR, f"model_{safe_symbol}.h5")
    scaler_filepath = os.path.join(MODEL_STORAGE_DIR, f"scaler_{safe_symbol}.pkl")

    # --- Fetch Data ---
    logger.info(f"Fetching data for {symbol}...")
    df = fetch_data(symbol)
    if df.empty:
        logger.error(f"No historical data found for stock {symbol} from data_loader.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No historical data found for stock {symbol}. Cannot proceed with prediction.")

    # Convert DataFrame to NumPy array for processing, using only specified features
    data = df[features].values
    logger.info(f"Fetched {len(data)} data points for {symbol}.")

    # --- Scaler Loading/Training ---
    scaler: Optional[MinMaxScaler] = None
    if retrain or not os.path.exists(scaler_filepath):
        logger.info(f"Scaler for {symbol} not found or retraining requested. Training new scaler...")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data) # Fit and transform on all available data
        joblib.dump(scaler, scaler_filepath) # Save the trained scaler
        loaded_scalers[symbol] = scaler # Store in global cache
        logger.info(f"Scaler for {symbol} trained and saved.")
    else:
        try:
            logger.info(f"Loading existing scaler for {symbol}...")
            scaler = joblib.load(scaler_filepath) # Load existing scaler
            scaled_data = scaler.transform(data) # Transform data using loaded scaler
            loaded_scalers[symbol] = scaler # Store in global cache
            logger.info(f"Scaler for {symbol} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading scaler for {symbol}: {e}. Retraining scaler instead.")
            # Fallback: if loading fails, train a new one
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            joblib.dump(scaler, scaler_filepath)
            loaded_scalers[symbol] = scaler
            logger.info(f"Scaler for {symbol} retrained due to load error.")
    
    # --- Create Sequences and Split Data ---
    # Ensure enough data to create sequences for both training and the final prediction.
    # The minimum data required is `long_window + future_days` for the sequences `X_short`, `X_long`, and labels `y`.
    required_sequence_data = long_window + future_days
    if len(scaled_data) < required_sequence_data:
        logger.error(f"Not enough scaled data ({len(scaled_data)}) for {symbol} to create sequences. "
                     f"Need at least {required_sequence_data} data points.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Not enough historical data for {symbol} to create sequences for model evaluation/prediction. "
                   f"Need at least {required_sequence_data} data points, but only have {len(scaled_data)}."
        )

    # Generate sequences and labels based on the scaled data
    X_short_all, X_long_all, y_all = create_sequences(scaled_data, threshold=threshold, future_days=future_days)
    
    if len(y_all) == 0:
        logger.error(f"No valid sequences could be created for {symbol} after processing.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Not enough valid sequences could be created for {symbol}. "
                   "This might be due to insufficient data or all 'p0' values being zero."
        )

    # Flatten X_short_all and X_long_all for SMOTE processing
    # Reshape (samples, timesteps, features) to (samples, timesteps * features)
    Xs_flat = X_short_all.reshape(X_short_all.shape[0], -1)
    Xl_flat = X_long_all.reshape(X_long_all.shape[0], -1)

    smote_k_neighbors = 5 # Default for SVMSMOTE

    unique_classes, counts = np.unique(y_all, return_counts=True)
    
    Xs_res, Xl_res, y_resampled = X_short_all, X_long_all, y_all # Default to original data
    
    if len(unique_classes) < 2:
        logger.warning(f"Only one class present in data for {symbol}. SMOTE cannot be applied.")
    else:
        # Determine the minority class count to set k_neighbors for SVMSMOTE
        minority_class_count = min(counts)
        # k_neighbors must be <= (minority_class_count - 1)
        # If minority_class_count is very small, SVMSMOTE might fail.
        if minority_class_count > 1: # SVMSMOTE needs at least 2 samples in the minority class to create neighbors
            smote_k_neighbors = min(smote_k_neighbors, minority_class_count - 1)
        else:
            logger.warning(f"Minority class has only {minority_class_count} sample(s) for {symbol}. Cannot apply SVMSMOTE.")
            # If SMOTE isn't applicable, continue with original data and log
            Xs_res, Xl_res, y_resampled = X_short_all, X_long_all, y_all

        if minority_class_count > 1 and smote_k_neighbors > 0: # Check if k_neighbors is valid
            try:
                smote = SVMSMOTE(random_state=42, k_neighbors=smote_k_neighbors)
                # For multi-input models, a common heuristic for SMOTE is to apply it to a concatenated
                # representation of the features, then split them back.
                # However, this can be complex. For simplicity, we apply to one and replicate.
                # A more robust solution might involve custom oversampling methods for sequences.
                
                # Concatenate features for SMOTE, then split back
                # This is a more robust way to handle multi-input SMOTE
                combined_features_flat = np.concatenate((Xs_flat, Xl_flat), axis=1)
                combined_features_resampled, y_resampled = smote.fit_resample(combined_features_flat, y_all)
                
                # Split back into short and long features
                Xs_flat_res = combined_features_resampled[:, :Xs_flat.shape[1]]
                Xl_flat_res = combined_features_resampled[:, Xs_flat.shape[1]:]

                # Reshape back to 3D sequences
                Xs_res = Xs_flat_res.reshape(-1, short_window, n_features)
                Xl_res = Xl_flat_res.reshape(-1, long_window, n_features)
                
                logger.info(f"SVMSMOTE applied for {symbol}. Original samples: {len(y_all)}, Resampled samples: {len(y_resampled)}")
            except Exception as e:
                logger.error(f"Error applying SVMSMOTE for {symbol}: {e}. Proceeding without resampling.")
                Xs_res, Xl_res, y_resampled = X_short_all, X_long_all, y_all
        else:
            logger.warning(f"SVMSMOTE k_neighbors ({smote_k_neighbors}) invalid for minority class count {minority_class_count}. Skipping SMOTE.")
            Xs_res, Xl_res, y_resampled = X_short_all, X_long_all, y_all # Fallback if k_neighbors is 0 or less


    # Perform train-test split on the (possibly resampled) data
    if len(y_resampled) < 2: # Need at least 2 samples for train_test_split (1 train, 1 test)
        logger.error(f"Not enough resampled data ({len(y_resampled)}) to perform train-test split for {symbol}. Need at least 2 samples.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Not enough data to train/evaluate model for {symbol}.")

    Xs_train, Xs_test, Xl_train, Xl_test, y_train, y_test = train_test_split(
        Xs_res, Xl_res, y_resampled, test_size=0.2, shuffle=True, random_state=42)

    logger.info(f"Data split for {symbol}: Train samples={len(y_train)}, Test samples={len(y_test)}")

    # --- Model Loading/Training ---
    model: Optional[tf.keras.Model] = None
    if retrain or not os.path.exists(model_filepath):
        logger.info(f"Model for {symbol} not found or retraining requested. Building and training new model...")
        model = build_model((short_window, n_features), (long_window, n_features))
        
        class_weights = None
        # Compute class weights to handle class imbalance during training
        if len(np.unique(y_train)) > 1:
            class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
            logger.info(f"Class weights for {symbol}: {class_weights}")
        else:
            logger.warning(f"Only one class present in y_train for {symbol}. Skipping class_weight calculation.")

        # Train the model
        history = model.fit([Xs_train, Xl_train], y_train, epochs=20, batch_size=32,
                            validation_data=([Xs_test, Xl_test], y_test),
                            class_weight=class_weights if class_weights else None, verbose=0)
        
        # Check if training was successful (e.g., no NaN losses)
        if np.isnan(history.history['loss'][-1]):
            logger.error(f"Model training failed for {symbol}: Loss became NaN. Data might be problematic.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                                detail=f"Model training for {symbol} failed (NaN loss). Please check data or try retraining.")

        model.save(model_filepath) # Save the trained model
        loaded_models[symbol] = model # Store in global cache
        logger.info(f"Model for {symbol} trained and saved. Training history: {history.history.keys()}")
    else:
        try:
            logger.info(f"Loading existing model for {symbol} from {model_filepath}...")
            model = load_model(model_filepath) # Load existing model
            loaded_models[symbol] = model # Store in global cache
            logger.info(f"Model for {symbol} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}. Model file might be corrupted or incompatible. Requesting retraining.")
            # If loading fails, prompt frontend to retry with retraining
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                                detail=f"Model for {symbol} could not be loaded. Please try retraining (set retrain=true) or check backend logs.")

    # --- API Key for Alpha Vantage (check before data fetching if not already done in data_loader) ---
    # This check should ideally be in data_loader or a global config validation
    ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not ALPHA_VANTAGE_API_KEY:
        logger.warning("ALPHA_VANTAGE_API_KEY environment variable not set. Data fetching might fail if data_loader relies on it.")
        # Consider making this a hard error if data_loader *requires* it.

    # --- Metrics Calculation on Test Set ---
    y_probs = model.predict([Xs_test, Xl_test], verbose=0)
    y_pred = (y_probs > 0.5).astype(int) # Convert probabilities to binary predictions

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # Initialize other metrics as "N/A" in case calculation fails
    roc_auc = "N/A"
    pr_auc = "N/A"
    report_dict: Dict[str, Any] = {} # Type hint for classification_report output

    # Calculate ROC AUC, PR AUC, and Classification Report only if both classes are present in y_test
    if len(np.unique(y_test)) > 1:
        try:
            roc_auc = roc_auc_score(y_test, y_probs)
            pr_auc = average_precision_score(y_test, y_probs)
            # classification_report returns a string by default; output_dict=True for a dict
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            logger.info(f"Metrics for {symbol}: Accuracy={acc:.4f}, ROC AUC={roc_auc:.4f}, PR AUC={pr_auc:.4f}")
        except ValueError as e:
            logger.warning(f"Could not calculate ROC AUC/PR AUC/Classification Report for {symbol} due to value error (e.g., single class in y_test after split or other data issues): {e}")
            report_dict = {"info": f"Detailed report not available due to: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error calculating metrics for {symbol}: {e}")
            report_dict = {"info": f"Error calculating metrics: {e}"}
    else:
        logger.warning(f"Only one class present in y_test for {symbol}. ROC AUC, PR AUC, and Classification Report will be N/A.")
        report_dict = {"info": "Only one class found in test set, detailed classification report not available."}


    # --- Prediction for the next N days (using the LATEST available data) ---
    # We need enough recent data to form the last short and long sequences for prediction
    if len(scaled_data) < long_window: # Prediction only needs `long_window` data points
        logger.error(f"Not enough recent data ({len(scaled_data)}) for {symbol} to make a future prediction. "
                     f"Need at least {long_window} data points.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Not enough recent data for {symbol} to make a future prediction. "
                   f"Need at least {long_window} data points, but only have {len(scaled_data)}."
        )

    # Extract the very last sequences from the entire scaled dataset
    # These are used for the actual 'next day' prediction
    last_short_data = scaled_data[-short_window:].reshape(1, short_window, n_features)
    last_long_data = scaled_data[-long_window:].reshape(1, long_window, n_features)

    # Make the prediction (probability of going "up")
    prediction_start_time = time.time()
    next_prob = float(model.predict([last_short_data, last_long_data], verbose=0)[0][0])
    prediction_end_time = time.time()
    logger.info(f"Prediction for {symbol} took {prediction_end_time - prediction_start_time:.4f} seconds.")
    
    next_percent = round(next_prob * 100, 2) # Convert to percentage

    # --- Interpret the Prediction ---
    direction = "UNCERTAIN"
    explanation = "🤔 Prediction confidence is low — proceed with caution."
    probability_display = next_percent # Default probability shown is for "up" movement

    # Custom thresholds for interpreting the prediction with more confidence
    # You can adjust these thresholds (e.g., 0.55 and 0.45)
    if next_prob >= 0.55: # If probability of "up" is 55% or more
        direction = "UP"
        probability_display = next_percent
        explanation = f"📈 Expected to go UP by {threshold*100:.0f}% or more over {future_days} days."
    elif next_prob <= 0.45: # If probability of "up" is 45% or less (implies 55% or more chance of not going up)
        direction = "DOWN"
        probability_display = 100 - next_percent # Show probability of going down
        explanation = f"📉 Expected to go DOWN by {threshold*100:.0f}% or more over {future_days} days."
    else: # If probability is between 45% and 55%
        direction = "FLAT"
        # For 'FLAT', showing the raw 'up' probability or the probability of staying within the threshold.
        explanation = f"↔️ Expected to remain relatively flat (within {threshold*100:.0f}% change over {future_days} days)."
    
    # Special case for very close to 50/50 for extra uncertainty
    if abs(next_prob - 0.5) < 0.02: # If prediction is very close to 50% (e.g., between 48% and 52%)
        direction = "UNCERTAIN"
        explanation = "🤔 Very close to 50/50 probability, prediction is highly uncertain."
        probability_display = next_percent # Still show the raw 'up' probability value

    total_process_time = time.time() - start_time
    logger.info(f"Total processing time for {symbol}: {total_process_time:.2f} seconds.")

    return {
        "symbol": symbol,
        "model_evaluation": {
            "accuracy": round(acc, 4), # Accuracy on the test set
            "roc_auc": round(roc_auc, 4) if isinstance(roc_auc, float) else roc_auc, # ROC AUC on the test set
            "pr_auc": round(pr_auc, 4) if isinstance(pr_auc, float) else pr_auc, # PR AUC on the test set
            "classification_report": report_dict, # Detailed classification report
            "test_set_size": len(y_test)
        },
        "next_prediction": {
            "direction": direction,
            "probability_percent": probability_display,
            "explanation": explanation,
            "raw_probability_up": round(next_prob, 4), # Raw probability of 'up' for debugging/info
            "threshold_used": threshold,
            "future_days_considered": future_days
        },
        "processing_time_seconds": round(total_process_time, 2)
    }

# --- API Endpoints for Prediction ---
# Using @app.post and @app.get for flexibility, expecting /api/predict
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
        raise e
    except Exception as e:
        logger.exception(f"Unhandled exception in POST /api/predict for {request.symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {e}"
        )


@app.get("/api/predict", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def predict_get_route(
    symbol: str,
    retrain: bool = False, # Query parameter for retraining
    threshold: float = 0.02, # Query parameter for threshold
    future_days: int = 3 # Query parameter for future days
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
        raise e
    except Exception as e:
        logger.exception(f"Unhandled exception in GET /api/predict for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {e}"
        )
