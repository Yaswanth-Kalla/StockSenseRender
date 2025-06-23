import os

API_KEY = "4XR8Y3WX2AZHHL0E"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

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
}


short_window = 30
long_window = 120
