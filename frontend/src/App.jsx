import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import StockList from './pages/StockList';
import StockDetail from './pages/StockDetail';
import Predict from './pages/Predict';


const App = () => (
  <Routes>
    <Route path="/" element={<Home />} />
    <Route path="/stocks" element={<StockList />} />
    <Route path="/stocks/:stockId" element={<StockDetail />} />
    <Route path="/predict" element={<Predict />} />
  </Routes>
);

export default App;