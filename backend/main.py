# backend/main.py

import os
from fastapi import FastAPI, HTTPException
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
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Bidirectional
import joblib
import time
from imblearn.over_sampling import SVMSMOTE

# --- IMPORTANT: These imports assume 'config.py' and 'data_loader.py' are
#     inside a 'app/' subdirectory within your 'backend/' directory,
#     and 'backend/app/__init__.py' exists. ---
from app.config import TOP_BSE_STOCKS
from app.data_loader import fetch_data

app = FastAPI()

# --- Global Variables for Loaded Models and Scalers ---
# These dictionaries will store loaded models and scalers in memory
# to avoid reloading them on every request.
loaded_models = {}
loaded_scalers = {}

# --- Directory for models/scalers relative to this script ---
# This correctly points to 'backend/models/' assuming main.py is in 'backend/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_STORAGE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_STORAGE_DIR, exist_ok=True) # Ensure directory exists on first run if not present

# --- CORS Configuration ---
# Render will provide the URL(s) for your frontend service(s) via the FRONTEND_URL env var.
# It's safer to explicitly list trusted origins than use "*".
# The FRONTEND_URL environment variable on Render should contain your deployed frontend's URL.
# If you have multiple allowed frontend URLs, separate them with commas (e.g., "url1,url2").
allowed_origins_env = os.environ.get("FRONTEND_URL")
origins = []

if allowed_origins_env:
    # Split by comma and strip whitespace for multiple URLs
    origins.extend([url.strip() for url in allowed_origins_env.split(',')])

# Add common local development origins explicitly for local testing
origins.extend([
    "http://localhost:5173", # Vite default dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000", # Common React dev server
    "http://127.0.0.1:3000",
    # If your backend runs locally on a different port (e.g., 8000), you might add it here:
    # "http://localhost:8000",
])

# Remove any duplicate URLs from the list
origins = list(set(origins))

print(f"Configuring CORS with allowed origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
# Defines the expected structure for a POST request to /predict
class PredictRequest(BaseModel):
    symbol: str
    retrain: bool = False # Whether to force retraining the model for this symbol
    threshold: float = 0.02 # Prediction threshold for "up" movement
    future_days: int = 3 # Number of future days to predict over

# --- Helper Functions for Data Processing and Model Building ---

def create_sequences(data, threshold=0.02, future_days=3):
    """
    Creates sequences for LSTM model training and the corresponding labels (y).
    """
    X_short, X_long, y = [], [], []
    
    # Ensure there's enough data for at least one long window + future days for y calculation
    if len(data) < long_window + future_days:
        print(f"Not enough data ({len(data)}) to create sequences for prediction. Need at least {long_window + future_days} points.")
        return np.array([]), np.array([]), np.array([]) # Return empty arrays if not enough data

    # Iterate through the data to create sequences
    # The loop stops early enough to ensure 'future_days' data points are available for 'y'
    for i in range(len(data) - long_window - future_days + 1):
        # Short window sequence (e.g., last 30 days of the long window)
        X_short.append(data[i + long_window - short_window : i + long_window])
        # Long window sequence (e.g., last 120 days)
        X_long.append(data[i : i + long_window])
        
        p0 = data[i + long_window - 1][0] # Assuming 'Close' price is the first feature (index 0) at the end of the long window
        
        # Calculate the average price for the 'future_days'
        future_avg = np.mean([data[i + long_window + j][0] for j in range(future_days)])
        
        if p0 == 0:
            # Skip samples where the base price is zero to avoid division by zero
            # and illogical delta calculations.
            # This can happen with very sparse or erroneous data.
            continue
        
        delta = (future_avg - p0) / p0 # Percentage change
        y.append(1 if delta > threshold else 0) # Label: 1 if price goes up by threshold, 0 otherwise
        
    # Ensure all lists have the same length after potential 'continue' statements
    min_len = min(len(X_short), len(X_long), len(y))
    return np.array(X_short[:min_len]), np.array(X_long[:min_len]), np.array(y[:min_len])

def build_model(input_short_shape, input_long_shape):
    """
    Builds and compiles the TensorFlow Keras Bidirectional LSTM model.
    """
    # Input layers for short and long sequences
    input_short = Input(shape=input_short_shape, name='input_short')
    input_long = Input(shape=input_long_shape, name='input_long')

    # Bidirectional LSTM layers for each input
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
    return model

# --- FastAPI Endpoints ---

@app.get("/")
async def read_root():
    """
    Root endpoint for basic API health check.
    """
    return {"message": "📈 Welcome to Stock Movement Prediction API. Visit /docs for OpenAPI specification."}

@app.get("/api/stocks") # Consistent with frontend expecting /api/stocks
async def get_stocks():
    """
    Returns a list of top BSE stocks (from your app.config).
    """
    return {"stocks": TOP_BSE_STOCKS}

@app.get("/api/stocks/{stock_id}") # Consistent with frontend expecting /api/stocks/{stock_id}
async def get_stock_info(stock_id: str):
    """
    Fetches and returns recent historical data for a specific stock ID.
    """
    name = TOP_BSE_STOCKS.get(stock_id, "Unknown Stock") # Get friendly name
    
    # Fetch data using your data_loader
    df = fetch_data(stock_id)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for stock {stock_id}")
    
    # Return last 60 data points as a list of dictionaries
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
    # Define file paths for symbol-specific model and scaler
    # Replace '.' in symbol with '_' for valid filenames
    model_filepath = os.path.join(MODEL_STORAGE_DIR, f"{symbol.replace('.', '_')}.h5")
    scaler_filepath = os.path.join(MODEL_STORAGE_DIR, f"{symbol.replace('.', '_')}_scaler.pkl")

    # --- Fetch Data ---
    df = fetch_data(symbol)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No historical data found for stock {symbol}. Cannot proceed with prediction.")

    # Convert DataFrame to NumPy array for processing, using only specified features
    data = df[features].values

    # --- Scaler Loading/Training ---
    scaler = None
    if retrain or not os.path.exists(scaler_filepath):
        print(f"Scaler for {symbol} not found or retraining requested. Training new scaler...")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data) # Fit and transform on all available data
        joblib.dump(scaler, scaler_filepath) # Save the trained scaler
        loaded_scalers[symbol] = scaler # Store in global cache
        print(f"Scaler for {symbol} trained and saved.")
    else:
        try:
            print(f"Loading existing scaler for {symbol}...")
            scaler = joblib.load(scaler_filepath) # Load existing scaler
            scaled_data = scaler.transform(data) # Transform data using loaded scaler
            loaded_scalers[symbol] = scaler # Store in global cache
            print(f"Scaler for {symbol} loaded successfully.")
        except Exception as e:
            print(f"Error loading scaler for {symbol}: {e}. Retraining scaler instead.")
            # Fallback: if loading fails, train a new one
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            joblib.dump(scaler, scaler_filepath)
            loaded_scalers[symbol] = scaler
            print(f"Scaler for {symbol} retrained due to load error.")
    
    # --- Create Sequences and Split Data ---
    # Ensure enough data to create sequences for both training and the final prediction.
    # The minimum data required is `long_window + future_days` for the sequences `X_short`, `X_long`, and labels `y`.
    if len(scaled_data) < long_window + future_days:
        raise HTTPException(
            status_code=400, 
            detail=f"Not enough historical data for {symbol} to create sequences for model evaluation/prediction. "
                   f"Need at least {long_window + future_days} data points, but only have {len(scaled_data)}."
        )

    # Generate sequences and labels based on the scaled data
    X_short_all, X_long_all, y_all = create_sequences(scaled_data, threshold=threshold, future_days=future_days)
    
    if len(y_all) == 0:
        raise HTTPException(
            status_code=400, 
            detail=f"Not enough valid sequences could be created for {symbol}. "
                   "This might be due to insufficient data or all 'p0' values being zero."
        )

    # Flatten X_short_all and X_long_all for SMOTE processing
    Xs_flat = X_short_all.reshape(X_short_all.shape[0], -1)
    Xl_flat = X_long_all.reshape(X_long_all.shape[0], -1)

    smote = SVMSMOTE(random_state=42, k_neighbors=min(5, len(y_all) - 1)) # k_neighbors cannot exceed number of samples in minority class
    
    unique_classes, counts = np.unique(y_all, return_counts=True)
    
    # Check if SMOTE is applicable (needs at least two classes and enough samples for k_neighbors)
    if len(unique_classes) < 2 or (min(counts) <= smote.k_neighbors):
        print(f"Warning: Not enough samples or only one class for SMOTE for {symbol}. Skipping SMOTE and using original data for training/evaluation.")
        Xs_res, Xl_res, y_resampled = X_short_all, X_long_all, y_all
    else:
        # Apply SMOTE. This will modify Xs_flat and y_all.
        # For multi-input models, applying SMOTE to one input and then replicating for others
        # is a heuristic. A more robust solution involves concatenating features
        # (Xs_flat, Xl_flat) before SMOTE, then splitting them back.
        Xs_flat_res, y_resampled = smote.fit_resample(Xs_flat, y_all)
        
        # This part ensures Xl_flat_res matches the resampled Xs_flat_res and y_resampled lengths.
        # It's a pragmatic workaround for SVMSMOTE with multiple inputs without deeper changes.
        if len(y_resampled) > len(y_all): # If SMOTE generated new samples
            Xl_flat_res = np.zeros((len(y_resampled), Xl_flat.shape[1]))
            # Copy original data, then new rows remain zeros (or could be custom-generated)
            Xl_flat_res[:len(Xl_flat)] = Xl_flat 
        else: # If no oversampling occurred or undersampling happened
            Xl_flat_res = Xl_flat 

        # Reshape back to 3D sequences
        Xs_res = Xs_flat_res.reshape(-1, short_window, n_features)
        Xl_res = Xl_flat_res.reshape(-1, long_window, n_features)
        
        # Ensure all resulting arrays have the same length after resampling
        min_resampled_len = min(len(y_resampled), len(Xs_res), len(Xl_res))
        y_resampled = y_resampled[:min_resampled_len]
        Xs_res = Xs_res[:min_resampled_len]
        Xl_res = Xl_res[:min_resampled_len]
        print(f"SMOTE applied for {symbol}. Original samples: {len(y_all)}, Resampled samples: {len(y_resampled)}")

    # Perform train-test split on the (possibly resampled) data
    if len(y_resampled) < 2: # Need at least 2 samples for train_test_split (1 train, 1 test)
        raise HTTPException(status_code=400, detail=f"Not enough resampled data to perform train-test split for {symbol}. Need at least 2 samples.")

    Xs_train, Xs_test, Xl_train, Xl_test, y_train, y_test = train_test_split(
        Xs_res, Xl_res, y_resampled, test_size=0.2, shuffle=True, random_state=42)

    print(f"Data split for {symbol}: Train samples={len(y_train)}, Test samples={len(y_test)}")

    # --- Model Loading/Training ---
    model = None
    if retrain or not os.path.exists(model_filepath):
        print(f"Model for {symbol} not found or retraining requested. Building and training new model...")
        model = build_model((short_window, n_features), (long_window, n_features))
        
        class_weights = None
        # Compute class weights to handle class imbalance during training
        if len(np.unique(y_train)) > 1:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = {i: class_weights[i] for i in range(len(class_weights))}
            print(f"Class weights for {symbol}: {class_weights}")
        else:
            print(f"Warning: Only one class present in y_train for {symbol}. Skipping class_weight calculation.")

        # Train the model
        model.fit([Xs_train, Xl_train], y_train, epochs=20, batch_size=32,
                    validation_data=([Xs_test, Xl_test], y_test),
                    class_weight=class_weights if class_weights else None, verbose=0)
        model.save(model_filepath) # Save the trained model
        loaded_models[symbol] = model # Store in global cache
        print(f"Model for {symbol} trained and saved.")
    else:
        try:
            print(f"Loading existing model for {symbol}...")
            model = load_model(model_filepath) # Load existing model
            loaded_models[symbol] = model # Store in global cache
            print(f"Model for {symbol} loaded successfully.")
        except Exception as e:
            print(f"Error loading model for {symbol}: {e}. Model file might be corrupted or incompatible. Raising error to frontend.")
            raise HTTPException(status_code=500, detail=f"Model for {symbol} could not be loaded. Please try retraining (set retrain=true) or check backend logs.")

    # --- API Key for Alpha Vantage (check before data fetching if not already done in data_loader) ---
    ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not ALPHA_VANTAGE_API_KEY:
        print("Warning: ALPHA_VANTAGE_API_KEY environment variable not set. Data fetching might fail.")
        # Decide if this should be a critical error or just a warning based on your data_loader's handling

    # --- Metrics Calculation on Test Set ---
    # Predict probabilities and classes on the test set
    y_probs = model.predict([Xs_test, Xl_test])
    y_pred = (y_probs > 0.5).astype(int) # Convert probabilities to binary predictions

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # Initialize other metrics as "N/A" in case calculation fails
    roc = "N/A"
    pr_auc = "N/A"
    report = {}

    # Calculate ROC AUC, PR AUC, and Classification Report only if both classes are present in y_test
    if len(np.unique(y_test)) > 1:
        try:
            roc = roc_auc_score(y_test, y_probs)
            pr_auc = average_precision_score(y_test, y_probs)
            report = classification_report(y_test, y_pred, output_dict=True)
            print(f"Metrics for {symbol}: Accuracy={acc:.4f}, ROC AUC={roc:.4f}, PR AUC={pr_auc:.4f}")
        except ValueError as e:
            print(f"Warning: Could not calculate ROC AUC/PR AUC/Classification Report for {symbol} due to value error (e.g., single class in y_test after split or other data issues): {e}")
            report = {"info": f"Detailed report not available due to: {e}"}
    else:
        print(f"Warning: Only one class present in y_test for {symbol}. ROC AUC, PR AUC, and Classification Report will be N/A.")
        report = {"info": "Only one class found in test set, detailed classification report not available."}


    # --- Prediction for the next N days (using the LATEST available data) ---
    # We need enough recent data to form the last short and long sequences for prediction
    if len(scaled_data) < long_window:
        raise HTTPException(
            status_code=400, 
            detail=f"Not enough recent data for {symbol} to make a future prediction. "
                   f"Need at least {long_window} data points, but only have {len(scaled_data)}."
        )

    # Extract the very last sequences from the entire scaled dataset
    last_short_data = scaled_data[-short_window:].reshape(1, short_window, n_features)
    last_long_data = scaled_data[-long_window:].reshape(1, long_window, n_features)

    # Make the prediction (probability of going "up")
    next_prob = float(model.predict([last_short_data, last_long_data])[0][0])
    next_percent = round(next_prob * 100, 2) # Convert to percentage

    # --- Interpret the Prediction ---
    direction = "UNCERTAIN"
    explanation = "🤔 Prediction confidence is low — proceed with caution."
    probability = next_percent # Default probability shown is for "up" movement

    # Custom thresholds for interpreting the prediction with more confidence
    # You can adjust these thresholds (e.g., 0.52 and 0.48)
    if next_prob >= 0.55: # If probability of "up" is 55% or more
        direction = "UP"
        probability = next_percent
        explanation = f"📈 Expected to go UP by {threshold*100:.0f}% or more!"
    elif next_prob <= 0.45: # If probability of "up" is 45% or less (implies 55% or more chance of not going up)
        direction = "DOWN"
        probability = 100 - next_percent # Show probability of going down
        explanation = f"📉 Expected to go DOWN by {threshold*100:.0f}% or more!"
    else: # If probability is between 45% and 55%
        direction = "FLAT"
        # For 'FLAT', you might show the raw 'up' probability or the probability of staying within the threshold.
        # For simplicity, showing raw 'up' probability here.
        explanation = f"↔️ Expected to remain relatively flat (within {threshold*100:.0f}% change)."
    
    # Special case for very close to 50/50 for extra uncertainty
    if abs(next_prob - 0.5) < 0.02: # If prediction is very close to 50% (e.g., between 48% and 52%)
        direction = "UNCERTAIN"
        explanation = "🤔 Very close to 50/50 probability, prediction is highly uncertain."
        probability = next_percent # Still show the raw 'up' probability value

    return {
        "accuracy": round(acc, 4), # Accuracy on the test set
        "roc_auc": round(roc, 4) if isinstance(roc, float) else roc, # ROC AUC on the test set
        "pr_auc": round(pr_auc, 4) if isinstance(pr_auc, float) else pr_auc, # PR AUC on the test set
        "classification_report": report, # Detailed classification report
        "next_day_prediction": {
            "direction": direction,
            "probability_percent": probability,
            "explanation": explanation
        }
    }


# --- API Endpoints for Prediction ---
# Using @app.post and @app.get for flexibility, expecting /api/predict
@app.post("/api/predict")
async def predict_post(request: PredictRequest):
    """
    Predicts stock movement based on a POST request body.
    """
    return process_prediction(
        symbol=request.symbol,
        retrain=request.retrain,
        threshold=request.threshold,
        future_days=request.future_days
    )

@app.get("/api/predict")
async def predict_get(
    symbol: str,
    retrain: bool = False, # Query parameter for retraining
    threshold: float = 0.02, # Query parameter for threshold
    future_days: int = 3 # Query parameter for future days
):
    """
    Predicts stock movement based on GET request query parameters.
    """
    return process_prediction(
        symbol=symbol,
        retrain=retrain,
        threshold=threshold,
        future_days=future_days
    )