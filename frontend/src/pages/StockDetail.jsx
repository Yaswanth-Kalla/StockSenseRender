import React, { useEffect, useState, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom'; // Import useNavigate
import {
  Container,
  Typography,
  CircularProgress,
  Alert,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Paper,
  Box,
  Button // Import Button
} from '@mui/material';
import Navbar from '../components/Navbar';
import { getStockInfo } from '../api/config';
import 'chartjs-adapter-date-fns';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  TimeScale,
  Tooltip,
  Legend,
} from 'chart.js';
import {
  CandlestickController,
  CandlestickElement,
} from 'chartjs-chart-financial';
import { Chart } from 'react-chartjs-2';
import { motion } from 'framer-motion';

ChartJS.register(
  CategoryScale,
  LinearScale,
  TimeScale,
  Tooltip,
  Legend,
  CandlestickController,
  CandlestickElement
);

const Candlestick = (props) => <Chart type="candlestick" {...props} />;

// Background image
import backgroundImage from '../public/stocks-6.jpg';

const StockDetail = () => {
  const { stockId } = useParams();
  const navigate = useNavigate(); // Initialize useNavigate
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getStockInfo(stockId)
      .then((res) => {
        setData(res.data);
        setLoading(false);
      })
      .catch(() => {
        setError('Failed to load stock data');
        setLoading(false);
      });
  }, [stockId]);

  const chartData = useMemo(() => {
    if (!data?.data) return { datasets: [] };

    const last60 = data.data.slice(-60);
    return {
      datasets: [
        {
          label: 'Candlestick Chart',
          data: last60.map((row) => ({
            x: new Date(row.date),
            o: row.Open,
            h: row.High,
            l: row.Low,
            c: row.Close,
          })),
          borderColor: 'rgba(0, 201, 255, 1)',
          upColor: '#00e676',
          downColor: '#ff1744',
          color: '#888888',
          barThickness: 10,
        },
      ],
    };
  }, [data]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: 'nearest',
        intersect: false,
        callbacks: {
          label: (ctx) => {
            const { o, h, l, c } = ctx.raw;
            return `O: ${o}   H: ${h}   L: ${l}   C: ${c}`;
          },
        },
        backgroundColor: '#222',
        titleColor: '#fff',
        bodyColor: '#e0f7fa',
        borderColor: '#00C9FF',
        borderWidth: 1,
      },
      title: {
        display: true,
        text: 'Candlestick - Last 60 Days',
        color: '#b0e0e6',
        font: { size: 18, weight: 'bold', family: `'Inter', sans-serif` },
      },
    },
    scales: {
      x: {
        type: 'time',
        time: { unit: 'day', tooltipFormat: 'MMM dd' },
        ticks: { color: '#e0f7fa' },
        grid: { color: 'rgba(255,255,255,0.1)' },
      },
      y: {
        ticks: { color: '#e0f7fa' },
        grid: { color: 'rgba(255,255,255,0.1)' },
      },
    },
  }), []);

  // Handler for the predict button click
  const handlePredictClick = () => {
    navigate('/predict', { state: { selectedStockSymbol: stockId } });
  };

  return (
    <>
      <Navbar />

      <Box
        sx={{
          minHeight: '100vh',
          backgroundImage: `url(${backgroundImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundAttachment: 'fixed',
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          color: '#fff',
        }}
      >
        <Box sx={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)', zIndex: 1 }} />

        <Container
          sx={{
            zIndex: 2, position: 'relative',
            py: 4, my: 4,
            backdropFilter: 'blur(6px)',
            backgroundColor: 'rgba(255,255,255,0.06)',
            borderRadius: 2, border: '1px solid rgba(255,255,255,0.15)',
            boxShadow: 3,
            maxWidth: { md: '1000px' },
            paddingTop: { xs: '90px', sm: '100px' },
            marginTop: { xs: '0px', sm: '0px' },
          }}
        >
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 10 }}>
              <CircularProgress color="inherit" />
            </Box>
          ) : error ? (
            <Alert severity="error">{error}</Alert>
          ) : (
            <>
              <motion.div
                initial={{ opacity: 0, y: -30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7 }}
              >
                <Typography
                  variant="h4"
                  sx={{
                    textAlign: 'center', mb: 3,
                    fontWeight: 'bold', fontFamily: `'Inter', sans-serif`,
                    color: '#e0f7fa'
                  }}
                >
                  {data.name} ({stockId})
                </Typography>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
              >
                <Paper elevation={6} sx={{
                  p: 2, mb: 4,
                  height: { xs: 300, md: 450 },
                  backgroundColor: 'rgba(255,255,255,0.08)',
                  backdropFilter: 'blur(8px)',
                  border: '1px solid rgba(255,255,255,0.15)',
                }}>
                  {chartData.datasets.length ? (
                    <Candlestick data={chartData} options={chartOptions} />
                  ) : (
                    <Typography>No chart data.</Typography>
                  )}
                </Paper>
              </motion.div>

              {/* Predict Button */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.3 }} // Slightly delayed animation
                style={{ textAlign: 'center', marginBottom: '24px' }} // Center the button
              >
                <Button
                  variant="contained"
                  onClick={handlePredictClick}
                  sx={{
                    backgroundColor: 'transparent',
                    backgroundImage: 'linear-gradient(45deg, rgba(0, 201, 255, 0.1), rgba(146, 254, 157, 0.05))',
                    backdropFilter: 'blur(8px) brightness(90%)',
                    color: '#fff',
                    fontWeight: 'bold',
                    padding: '12px 40px',
                    borderRadius: '10px',
                    fontSize: '1.1rem',
                    transition: 'all 0.3s ease-in-out',
                    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
                    border: '1px solid rgba(0, 201, 255, 0.4)',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      backgroundImage: 'linear-gradient(45deg, rgba(146, 254, 157, 0.15), rgba(0, 201, 255, 0.3))',
                      boxShadow: '0 8px 30px rgba(0, 0, 0, 0.6), 0 0 30px #00C9FF',
                      borderColor: '#00C9FF',
                    },
                    '&.Mui-disabled': { // Added for completeness, though probably not needed here
                      backgroundColor: 'rgba(255, 255, 255, 0.04)',
                      color: 'rgba(255, 255, 255, 0.4)',
                      boxShadow: 'none',
                      cursor: 'not-allowed',
                      borderColor: 'rgba(255, 255, 255, 0.05)',
                      backdropFilter: 'blur(5px) brightness(80%)',
                      backgroundImage: 'none',
                    }
                  }}
                >
                  Predict This Stock
                </Button>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.9 }}
              >
                <Typography variant="h5" sx={{
                  textAlign: 'center', mb: 2,
                  fontWeight: 'bold', color: '#e0f7fa'
                }}>
                  Historical Data (Last 30 Days)
                </Typography>

                <Paper elevation={6} sx={{
                  backgroundColor: 'rgba(255,255,255,0.08)',
                  backdropFilter: 'blur(8px)',
                  border: '1px solid rgba(255,255,255,0.15)',
                  overflowX: 'auto',
                }}>
                  <Table size="small" sx={{ minWidth: 500 }}>
                    <TableHead>
                      <TableRow>
                        {['Date', 'Open', 'High', 'Low', 'Close'].map((head) => (
                          <TableCell key={head} sx={{
                            color: '#a7d9ef', fontWeight: 'bold', whiteSpace: 'nowrap'
                          }}>{head}</TableCell>
                        ))}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {data.data.slice(-30).map((row, idx) => (
                        <TableRow
                          key={idx}
                          sx={{ '&:nth-of-type(odd)': { backgroundColor: 'rgba(255,255,255,0.05)' } }}
                        >
                          <TableCell sx={{ color: '#e0f7fa', whiteSpace: 'nowrap' }}>
                            {new Date(row.date).toLocaleDateString()}
                          </TableCell>
                          <TableCell sx={{ color: '#e0f7fa', whiteSpace: 'nowrap' }}>{row.Open}</TableCell>
                          <TableCell sx={{ color: '#e0f7fa', whiteSpace: 'nowrap' }}>{row.High}</TableCell>
                          <TableCell sx={{ color: '#e0f7fa', whiteSpace: 'nowrap' }}>{row.Low}</TableCell>
                          <TableCell sx={{ color: '#e0f7fa', whiteSpace: 'nowrap' }}>{row.Close}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Paper>
              </motion.div>
            </>
          )}
        </Container>
      </Box>
    </>
  );
};

export default StockDetail;