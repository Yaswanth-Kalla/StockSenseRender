import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom'; // Import useLocation
import {
  Container, Typography, Button, MenuItem,
  FormControlLabel,
  CircularProgress, Alert, Paper,
  Box, TextField,
  Switch
} from '@mui/material';
import { getStocks, predict } from '../api/config';
import Navbar from '../components/Navbar';
import { motion } from 'framer-motion';

import backgroundImage from '../public/stocks-6.jpg';

const MotionTypography = motion(Typography);
const MotionBox = motion(Box);
const MotionButton = motion(Button);
const MotionPaper = motion(Paper);

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.2
    }
  }
};

const itemVariants = {
  hidden: { y: 50, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      duration: 0.6,
      ease: "easeOut"
    }
  }
};

const Predict = () => {
  const location = useLocation(); // Initialize useLocation
  const { selectedStockSymbol } = location.state || {}; // Get state from navigation

  const [symbol, setSymbol] = useState('');
  const [retrain, setRetrain] = useState(false);
  const [stocks, setStocks] = useState({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const NAVBAR_HEIGHT = 85;

  useEffect(() => {
    getStocks()
      .then(res => {
        setStocks(res.data.stocks);
        // Set symbol from navigation state if available, otherwise default to first stock
        if (selectedStockSymbol && res.data.stocks[selectedStockSymbol]) {
          setSymbol(selectedStockSymbol);
        } else if (Object.keys(res.data.stocks).length > 0) {
          setSymbol(Object.keys(res.data.stocks)[0]);
        }
      })
      .catch(() => setError("Failed to fetch stock list"));
  }, [selectedStockSymbol]); // Depend on selectedStockSymbol to re-run if it changes

  const handleSubmit = () => {
    setLoading(true);
    setResult(null);
    setError('');

    if (!symbol) {
      setError("Please select a stock symbol.");
      setLoading(false);
      return;
    }

    predict({ symbol, retrain })
      .then(res => {
        setResult(res.data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Prediction API error:", err);
        const backendErrorMessage = err.response && err.response.data && err.response.data.message;

        if (backendErrorMessage && backendErrorMessage.includes("No pre-trained model found") && !retrain) {
          setError("No pre-trained model found. Please choose the 'Retrain Model' option.");
        } else {
          setError("Prediction failed. Please try again later or check your inputs.");
        }
        setLoading(false);
      });
  };

  const buttonWidth = 42;
  const buttonHeight = 24;
  const toggleDiameter = 19;
  const buttonToggleOffset = (buttonHeight - toggleDiameter) / 2;
  const toggleShadowOffset = 7;
  const toggleWider = 36;
  const colorGrey = '#cccccc';
  const colorGreen = '#4296f4';

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
          boxSizing: 'border-box',
          overflow: 'auto',
          '&::before': {
            content: '""',
            position: 'absolute',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.6)',
            zIndex: 0,
          }
        }}
      >
        <Container
          sx={{
            zIndex: 2,
            position: 'relative',
            py: 4,
            maxWidth: { xs: '95%', sm: '80%', md: '700px' },
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '16px',
            boxShadow: '0 12px 40px 0 rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            my: 4,
            paddingTop: `${NAVBAR_HEIGHT + 24}px`,
            paddingBottom: '48px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            minHeight: `calc(100vh - ${NAVBAR_HEIGHT + 64}px)`,
          }}
        >
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}
          >
            <MotionTypography
              variant="h2"
              gutterBottom
              sx={{
                fontWeight: 900,
                fontFamily: `'Poppins', sans-serif`,
                textAlign: 'center',
                mb: 4,
                fontSize: { xs: '2.5rem', sm: '3rem', md: '3.75rem' },
                color: '#fff',
                textShadow: '3px 3px 6px rgba(0,0,0,0.7)',
                lineHeight: 1.1,
              }}
              variants={itemVariants}
            >
              Predict Your Stock Moves 📈
            </MotionTypography>

            <MotionBox
              sx={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: 3,
                justifyContent: { xs: 'center', sm: 'center' },
                width: '100%',
                mb: 4,
              }}
              variants={itemVariants}
            >
              <TextField
                select
                label="Select Stock"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                variant="outlined"
                sx={{
                  minWidth: { xs: '90%', sm: 300 },
                  '& .MuiInputLabel-root': {
                    color: '#b0e0e6',
                    fontSize: '1.1rem',
                    '&.Mui-focused': { color: '#00C9FF' },
                  },
                  '& .MuiInputBase-input': {
                    color: '#fff',
                    padding: '14px 12px',
                    textShadow: '0 0 5px rgba(255,255,255,0.7)',
                  },
                  '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(255, 255, 255, 0.3)', transition: 'border-color 0.3s ease' },
                  '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(255, 255, 255, 0.6)' },
                  '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#00C9FF', borderWidth: '2px' },
                  '& .MuiSvgIcon-root': { color: '#b0e0e6', transition: 'color 0.3s ease' },
                }}
                InputLabelProps={{
                  shrink: true,
                }}
                SelectProps={{
                  MenuProps: {
                    PaperProps: {
                      sx: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        backdropFilter: 'blur(8px)',
                        borderRadius: '8px',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
                      },
                    },
                  },
                }}
              >
                {Object.entries(stocks).map(([key, name]) => (
                  <MenuItem
                    key={key}
                    value={key}
                    sx={{
                      backgroundColor: 'transparent',
                      color: '#fff',
                      '&:hover': {
                        backgroundColor: 'rgba(0, 201, 255, 0.2)',
                        color: '#00C9FF',
                      },
                      '&.Mui-selected': {
                        backgroundColor: 'rgba(0, 201, 255, 0.8)',
                        color: '#fff',
                        fontWeight: 'bold',
                        boxShadow: '0 0 15px rgba(0, 201, 255, 0.6)',
                        borderRadius: '4px',
                        transition: 'background-color 0.3s ease, box-shadow 0.3s ease',
                        '&:hover': {
                            backgroundColor: 'rgba(0, 201, 255, 0.9)',
                            boxShadow: '0 0 20px rgba(0, 201, 255, 0.8)',
                        }
                      },
                    }}
                  >
                    {name} ({key})
                  </MenuItem>
                ))}
              </TextField>

              <FormControlLabel
                control={
                  <Switch
                    checked={retrain}
                    onChange={(e) => setRetrain(e.target.checked)}
                    sx={{
                      width: buttonWidth,
                      height: buttonHeight,
                      padding: 0,
                      overflow: 'visible',

                      '& .MuiSwitch-switchBase': {
                        padding: `${buttonToggleOffset}px`,
                        transition: 'all 0.3s ease-in-out',
                        color: 'transparent',

                        '&.Mui-checked': {
                          transform: `translateX(${buttonWidth - toggleDiameter - buttonToggleOffset}px)`,
                          color: 'transparent',
                          '& + .MuiSwitch-track': {
                            backgroundColor: colorGreen,
                          },
                        },

                        '&.Mui-active': {
                            '& .MuiSwitch-thumb': {
                                width: toggleWider,
                            },
                        },
                        '&.Mui-checked.Mui-active': {
                            transform: `translateX(${buttonWidth - toggleWider - buttonToggleOffset}px)`,
                        }
                      },

                      '& .MuiSwitch-thumb': {
                        width: toggleDiameter,
                        height: toggleDiameter,
                        backgroundColor: '#fff',
                        borderRadius: '50%',
                        boxShadow: `${toggleShadowOffset}px 0 calc(${toggleShadowOffset}px * 4) rgba(0, 0, 0, 0.1)`,
                        transition: 'all 0.3s ease-in-out',
                      },

                      '& .MuiSwitch-switchBase.Mui-checked .MuiSwitch-thumb': {
                        boxShadow: `calc(${toggleShadowOffset}px * -1) 0 calc(${toggleShadowOffset}px * 4) rgba(0, 0, 0, 0.1)`,
                      },

                      '& .MuiSwitch-track': {
                        width: buttonWidth,
                        height: buttonHeight,
                        backgroundColor: colorGrey,
                        borderRadius: `${buttonHeight / 2}px`,
                        opacity: 1,
                        transition: 'all 0.3s ease-in-out',
                      },
                    }}
                  />
                }
                label={
                  <Typography sx={{ color: '#fff', pr: 1 }}>
                    Retrain Model
                  </Typography>
                }
                labelPlacement="start"
                sx={{
                  minWidth: { xs: '90%', sm: 'auto' },
                  mr: { xs: 0, sm: 2 },
                  '& .MuiFormControlLabel-label': { fontSize: '1.1rem' },
                  alignSelf: 'center',
                }}
              />
            </MotionBox>

            <MotionButton
              variant="contained"
              onClick={handleSubmit}
              disabled={loading || !symbol}
              sx={{
                mt: 2,
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
                '&.Mui-disabled': {
                  backgroundColor: 'rgba(255, 255, 255, 0.04)',
                  color: 'rgba(255, 255, 255, 0.4)',
                  boxShadow: 'none',
                  cursor: 'not-allowed',
                  borderColor: 'rgba(255, 255, 255, 0.05)',
                  backdropFilter: 'blur(5px) brightness(80%)',
                  backgroundImage: 'none',
                }
              }}
              variants={itemVariants}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Predict Stock Movement'}
            </MotionButton>

            {error && <Alert severity="error" sx={{ mt: 3, zIndex: 3, width: '100%' }}>{error}</Alert>}

            {result && (
              <MotionPaper
                elevation={6}
                sx={{
                  mt: 5,
                  p: 3,
                  backgroundColor: 'rgba(255, 255, 255, 0.08)',
                  backdropFilter: 'blur(10px) brightness(90%)',
                  borderRadius: '16px',
                  overflow: 'hidden',
                  border: '1px solid rgba(255, 255, 255, 0.15)',
                  boxShadow: '0 8px 30px rgba(0, 0, 0, 0.4)',
                  width: '100%',
                }}
                variants={itemVariants}
              >
                <Typography variant="h6" gutterBottom sx={{
                  color: '#e0f7fa',
                  fontWeight: 'bold',
                  fontFamily: `'Inter', sans-serif`,
                  textShadow: '0 0 5px rgba(0,0,0,0.5)',
                  mb: 2,
                }}>
                  📊 Prediction Results
                </Typography>

                <Box sx={{ mb: 3 }}>
                  <Typography sx={{ color: '#fff', mb: 1, fontSize: '1.1rem' }}>
                    ✅ Accuracy of the model: <strong style={{ color: '#92FE9D' }}>{result.accuracy.toFixed(4)}</strong>
                  </Typography>
                  <Typography sx={{ color: '#fff', mb: 1.5, fontSize: '1.2rem', fontWeight: 'bold' }}>
                    🔮 Prediction for Next 3 Trading Days: <strong style={{ color: result.next_day_prediction.direction === 'Up' ? '#92FE9D' : '#FF6B6B' }}>
                      {result.next_day_prediction.direction}
                    </strong>
                    &nbsp;with probability <strong style={{ color: '#00C9FF' }}>{(result.next_day_prediction.probability_percent).toFixed(2)}%</strong>
                  </Typography>
                </Box>
              </MotionPaper>
            )}
          </motion.div>
        </Container>
      </Box>
    </>
  );
};

export default Predict;