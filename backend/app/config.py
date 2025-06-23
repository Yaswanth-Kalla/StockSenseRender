# backend/app/config.py

import os

# --- IMPORTANT ---
# Do NOT hardcode your API key here in production.
# This value will be fetched from an environment variable set in Railway.
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

# Define your top BSE stocks here
TOP_BSE_STOCKS = {
    "RELIANCE.BSE": "Reliance Industries",
    "TCS.BSE": "Tata Consultancy Services",
    "INFY.BSE": "Infosys",
    "HDFCBANK.BSE": "HDFC Bank",
    "ICICIBANK.BSE": "ICICI Bank",
    "SBIN.BSE": "State Bank of India",
    "TATAMOTORS.BSE": "Tata Motors",
    "BHARTIARTL.BSE": "Bharti Airtel",
    "LT.BSE": "Larsen & Toubro",
    "AXISBANK.BSE": "Axis Bank",
    # Add more as needed
}

# You can also move short_window and long_window here if you prefer a single source of truth
# for all hyperparameters, and import them into main.py.
# For now, if main.py defines them, that's fine too.
# short_window = 30
# long_window = 120
