// frontend/src/api/config.js
import axios from 'axios';

// Get the API base URL from environment variables.
// In local development, this comes from frontend/.env.
// In Render deployment, this comes from the environment variable set in Render's dashboard.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

// --- IMPORTANT: Add a check for the API_BASE_URL ---
// This helps during local development if you forget to set the .env file.
// When deployed on Render, VITE_API_BASE_URL *will* be set, so this warning won't appear.
if (!API_BASE_URL) {
    console.error("VITE_API_BASE_URL environment variable is not set. API calls might fail.");
    // You might want to throw an error or set a default local URL here for robustness,
    // but for deployment, ensuring the env var is set on Render is key.
    // Example for local fallback: API_BASE_URL = 'http://localhost:8000';
}

const API = axios.create({ baseURL: API_BASE_URL });

// --- IMPORTANT: Update API endpoints to match your backend ---
// Your FastAPI backend uses "/api/stocks" and "/api/predict"
export const getStocks = () => API.get('/api/stocks');
export const getStockInfo = (symbol) => API.get(`/api/stocks/${symbol}`);
export const predict = ({ symbol, retrain, threshold, future_days }) =>
  API.get('/api/predict', { // Changed to /api/predict
    params: { symbol, retrain, threshold, future_days }
  });

export default API;
