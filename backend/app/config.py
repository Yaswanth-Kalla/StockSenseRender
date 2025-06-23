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
    "SUNPHARMA.BSE": "Sun Pharmaceuticals",
    "BHARTIARTL.BSE": "Bharti Airtel",
    "LT.BSE": "Larsen & Toubro",
    "BAJFINANCE.BSE": "Bajaj Finance",
    "AXISBANK.BSE": "Axis Bank",
    "ITC.BSE": "ITC",
    "HINDUNILVR.BSE": "Hindustan Unilever",
    "ULTRACEMCO.BSE": "UltraTech Cement",
    "MARUTI.BSE": "Maruti Suzuki",
    "TATASTEEL.BSE": "Tata Steel",
    "JSWSTEEL.BSE": "JSW Steel",
    "ONGC.BSE": "ONGC",
    "NTPC.BSE": "NTPC",
    "POWERGRID.BSE": "Power Grid",
    "COALINDIA.BSE": "Coal India",
}


short_window = 30
long_window = 120
