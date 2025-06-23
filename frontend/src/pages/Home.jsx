import React, { useEffect } from 'react';
import { Container, Typography, Button, Stack, Box, useTheme } from '@mui/material'; // Import useTheme
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import '../App.css'; // Global styles for general app styling (if any)

const headingText = "Predict Your Stock Moves 📈";
const subText = "Use LSTM-powered models to predict stock trends of top Indian companies.";

const Home = () => {
  const location = useLocation(); // Get the current location object
  const theme = useTheme(); // Access the theme object

  // useEffect for smooth scrolling to sections based on URL hash
  useEffect(() => {
    // Only attempt to scroll if on the home page and hash matches
    if (location.pathname === '/' && location.hash === '#about-section') {
      const timer = setTimeout(() => {
        const aboutSection = document.getElementById('about-section');
        if (aboutSection) {
          aboutSection.scrollIntoView({ behavior: 'smooth', block: 'start' }); // block: 'start' ensures it scrolls to the top of the element
        }
      }, 300); // Increased delay to 300ms for robustness

      return () => clearTimeout(timer); // Cleanup timer on unmount or re-render
    }
  }, [location]); // Re-run effect when the location object (including hash) changes

  return (
    <Box>
      <Navbar />

      {/* Hero Section */}
      <Box
        sx={{
          minHeight: '100vh',
          backgroundImage: "url('/src/public/stocks-6.jpg')", // Make sure this path is correct
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
            paddingTop: '80px', // Keep safe from fixed navbar
          }}
        >
          <Container sx={{ textAlign: 'center', color: '#fff', py: 4 }}>
            {/* Main Heading with Framer Motion animations */}
            <Typography
              component="h1"
              sx={{
                fontWeight: 900, // Matched with Predict.jsx heading
                fontFamily: `'Poppins', sans-serif`, // Poppins font for Hero Heading
                // Decreased font size by ~20%
                fontSize: { xs: '1.8rem', md: '2.8rem', lg: '3.8rem' },
                mb: 3,
                textShadow: '3px 3px 6px rgba(0,0,0,0.7)', // Add a bit more shadow for prominence
                lineHeight: 1.1, // Adjust line height if needed
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

            {/* Sub Text with Framer Motion animations */}
            <Typography
              variant="h6"
              sx={{
                mt: 2,
                fontFamily: `'Inter', sans-serif`, // Inter font for Hero Subtext
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

            {/* Action Buttons */}
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
                      // Highlight with primary color
                      boxShadow: `0 0 0 3px ${theme.palette.primary.main}, 0 4px 10px rgba(0,0,0,0.3)`,
                      transform: 'translateY(-2px)', // Subtle lift
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
                      // Highlight with secondary color
                      boxShadow: `0 0 0 3px ${theme.palette.secondary.main}, 0 4px 10px rgba(0,0,0,0.3)`,
                      transform: 'translateY(-2px)', // Subtle lift
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
        id="about-section" // This ID allows direct navigation from Navbar
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
                  fontFamily: `'Poppins', sans-serif`, // Poppins font for Technical Section Heading
                }}
              >
                🧠 How Our Stock Predictor Works
              </Typography>
              <Typography
                sx={{
                  mb: 2,
                  fontFamily: `'Inter', sans-serif`, // Inter font for Technical Section Body
                }}
              >
                Our system uses an advanced <strong>LSTM (Long Short-Term Memory)</strong> model trained on recent market data to forecast the next move in stock prices.
              </Typography>
              <Typography
                sx={{
                  mb: 2,
                  fontFamily: `'Inter', sans-serif`, // Inter font for Technical Section Body
                }}
              >
                ✅ It focuses on price changes of <strong>at least 2%</strong> over the next <strong>3 days</strong>. This threshold is ideal for balancing accuracy and actionability.
              </Typography>
              <Typography
                sx={{
                  mb: 2,
                  fontFamily: `'Inter', sans-serif`, // Inter font for Technical Section Body
                }}
              >
                📊 It leverages <strong>technical indicators</strong> like <strong>MACD</strong>, <strong>RSI</strong>, and <strong>SMA20</strong> to make reliable predictions.
              </Typography>
              <Typography
                sx={{
                  mb: 2,
                  fontFamily: `'Inter', sans-serif`, // Inter font for Technical Section Body
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
                src="https://media.istockphoto.com/id/1867035079/photo/analytics-and-data-management-systems-business-analytics-and-data-management-systems-to-make.jpg?s=612x612&w=0&k=20&c=tFcJnBIWlkPhIumrPtkSJwFRNDMtdVfJ1CYbfUlx5fE=" // *** REMEMBER TO REPLACE WITH YOUR ACTUAL IMAGE PATH ***
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