import React, { useEffect } from 'react';
import { Container, Typography, Button, Stack, Box, useTheme } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import '../App.css';

// *** NEW: Import the background image using Vite's asset handling ***
import homeBackgroundImage from '../public/stocks-6.jpg'; // Adjust path if your image is in a subfolder like public/images/

const headingText = "Predict Your Stock Moves 📈";
const subText = "Use LSTM-powered models to predict stock trends of top Indian companies.";

const Home = () => {
  const location = useLocation();
  const theme = useTheme();

  useEffect(() => {
    if (location.pathname === '/' && location.hash === '#about-section') {
      const timer = setTimeout(() => {
        const aboutSection = document.getElementById('about-section');
        if (aboutSection) {
          aboutSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 300);

      return () => clearTimeout(timer);
    }
  }, [location]);

  return (
    <Box>
      <Navbar />

      {/* Hero Section */}
      <Box
        sx={{
          minHeight: '100vh',
          // *** MODIFIED LINE: Use the imported image variable ***
          backgroundImage: `url(${homeBackgroundImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
        }}
      >
        {/* Overlay for readability */}
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            zIndex: 1,
          }}
        />

        {/* Hero Content */}
        <motion.div
          initial="hidden"
          animate="visible"
          style={{
            zIndex: 2,
            width: '100%',
            boxSizing: 'border-box',
            position: 'relative',
            paddingTop: '80px',
          }}
        >
          <Container sx={{ textAlign: 'center', color: '#fff', py: 4 }}>
            <Typography
              component="h1"
              sx={{
                fontWeight: 900,
                fontFamily: `'Poppins', sans-serif`,
                fontSize: { xs: '1.8rem', md: '2.8rem', lg: '3.8rem' },
                mb: 3,
                textShadow: '3px 3px 6px rgba(0,0,0,0.7)',
                lineHeight: 1.1,
              }}
            >
              {headingText.split(" ").map((word, index) => (
                <motion.span
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.15, duration: 0.5 }}
                  style={{ marginRight: 8, display: 'inline-block' }}
                >
                  {word}
                </motion.span>
              ))}
            </Typography>

            <Typography
              variant="h6"
              sx={{
                mt: 2,
                fontFamily: `'Inter', sans-serif`,
                fontSize: { xs: '1rem', md: '1.25rem' },
                color: '#ddd',
              }}
            >
              {subText.split(" ").map((word, index) => (
                <motion.span
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    delay: headingText.split(" ").length * 0.15 + index * 0.05,
                    duration: 0.4
                  }}
                  style={{ marginRight: 5, display: 'inline-block' }}
                >
                  {word}
                </motion.span>
              ))}
            </Typography>

            <Stack direction="row" justifyContent="center" spacing={2} mt={4}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  delay: headingText.split(" ").length * 0.15 + subText.split(" ").length * 0.05 + 0.3,
                }}
              >
                <Button
                  variant="contained"
                  color="primary"
                  component={Link}
                  to="/stocks"
                  sx={{
                    fontWeight: 600, px: 4, py: 1.5,
                    transition: 'all 0.3s ease-in-out',
                    '&:hover': {
                      boxShadow: `0 0 0 3px ${theme.palette.primary.main}, 0 4px 10px rgba(0,0,0,0.3)`,
                      transform: 'translateY(-2px)',
                    },
                  }}
                >
                  View Stocks
                </Button>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  delay: headingText.split(" ").length * 0.15 + subText.split(" ").length * 0.05 + 0.6,
                }}
              >
                <Button
                  variant="contained"
                  color="secondary"
                  component={Link}
                  to="/predict"
                  sx={{
                    fontWeight: 600, px: 4, py: 1.5,
                    transition: 'all 0.3s ease-in-out',
                    '&:hover': {
                      boxShadow: `0 0 0 3px ${theme.palette.secondary.main}, 0 4px 10px rgba(0,0,0,0.3)`,
                      transform: 'translateY(-2px)',
                    },
                  }}
                >
                  Make Prediction
                </Button>
              </motion.div>
            </Stack>
          </Container>
        </motion.div>
      </Box>

      {/* Technical Explanation Section - Target for 'About' link */}
      <motion.div
        id="about-section"
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
      >
        <Box
          sx={{
            background: `
              linear-gradient(135deg, #1a2a3a 0%, #0a1a2a 100%),
              radial-gradient(circle at center, rgba(30, 60, 90, 0.3) 0%, transparent 70%)
            `,
            backgroundBlendMode: 'overlay',
            color: '#e0f7fa',
            py: 6,
            px: 3,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '400px',
          }}
        >
          <Box
            sx={{
              maxWidth: '1000px',
              width: '100%',
              display: 'flex',
              flexDirection: { xs: 'column', md: 'row' },
              gap: { xs: 4, md: 6 },
              backgroundColor: 'rgba(255,255,255,0.05)',
              backdropFilter: 'blur(6px)',
              borderRadius: 2,
              border: '1px solid rgba(255,255,255,0.15)',
              p: 4,
              alignItems: 'center',
            }}
          >
            {/* Left Side: Text Content */}
            <Box sx={{ flex: 1, textAlign: { xs: 'center', md: 'left' } }}>
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 'bold',
                  mb: 2,
                  fontFamily: `'Poppins', sans-serif`,
                }}
              >
                🧠 How Our Stock Predictor Works
              </Typography>
              <Typography
                sx={{
                  mb: 2,
                  fontFamily: `'Inter', sans-serif`,
                }}
              >
                Our system uses an advanced <strong>LSTM (Long Short-Term Memory)</strong> model trained on recent market data to forecast the next move in stock prices.
              </Typography>
              <Typography
                sx={{
                  mb: 2,
                  fontFamily: `'Inter', sans-serif`,
                }}
              >
                ✅ It focuses on price changes of <strong>at least 2%</strong> over the next <strong>3 days</strong>. This threshold is ideal for balancing accuracy and actionability.
              </Typography>
              <Typography
                sx={{
                  mb: 2,
                  fontFamily: `'Inter', sans-serif`,
                }}
              >
                📊 It leverages <strong>technical indicators</strong> like <strong>MACD</strong>, <strong>RSI</strong>, and <strong>SMA20</strong> to make reliable predictions.
              </Typography>
              <Typography
                sx={{
                  mb: 2,
                  fontFamily: `'Inter', sans-serif`,
                }}
              >
                ⚙️ Predictions are based on real-time data fetched via API, and model logic is optimized for each stock using a dual-window Bi-LSTM setup.
              </Typography>
            </Box>

            {/* Right Side: Image */}
            <Box
              sx={{
                flex: { xs: 'none', md: 0.7 },
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                width: { xs: '100%', md: 'auto' },
                minHeight: { xs: '200px', md: '300px' },
              }}
            >
              <Box
                component="img"
                src="https://media.istockphoto.com/id/1867035079/photo/analytics-and-data-management-systems-business-analytics-and-data-management-systems-to-make.jpg?s=612x612&w=0&k=20&c=tFcJnBIWlkPhIumrPtkSJwFRNDMtdVfJ1CYbfUlx5fE="
                alt="AI Model Diagram or Data Visualization"
                sx={{
                  maxWidth: '100%',
                  height: 'auto',
                  borderRadius: 1,
                  boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
                  border: '1px solid rgba(255,255,255,0.1)',
                }}
              />
            </Box>
          </Box>
        </Box>
      </motion.div>
    </Box>
  );
};

export default Home;
