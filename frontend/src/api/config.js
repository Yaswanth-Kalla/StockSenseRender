import axios from 'axios';

// Correct base URL for your running FastAPI server
const API = axios.create({ baseURL: 'http://127.0.0.1:8000' });

export const getStocks = () => API.get('/stocks');
export const getStockInfo = (symbol) => API.get(`/stocks/${symbol}`);
export const predict = ({ symbol, retrain, threshold, future_days }) =>
  API.get('/predict', {
    params: { symbol, retrain, threshold, future_days }
  });

export default API;
