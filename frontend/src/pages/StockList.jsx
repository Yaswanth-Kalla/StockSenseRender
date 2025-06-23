import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  CircularProgress,
  List,
  ListItemButton,
  ListItemText,
  Box,
  TextField,
  InputAdornment,
  IconButton
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ClearIcon from '@mui/icons-material/Clear';
import { Link } from 'react-router-dom';
import { getStocks } from '../api/config';
import Navbar from '../components/Navbar';
import { motion, AnimatePresence } from 'framer-motion';

// Import the background image
import backgroundImage from '../public/stocks-6.jpg';

// Create motion-enabled Material-UI components
const MotionTypography = motion(Typography);
const MotionList = motion(List);
const MotionListItemButton = motion(ListItemButton);
const MotionBox = motion(Box);

// Define animation variants for staggered children effect
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2
    }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      duration: 0.4,
      ease: "easeOut"
    }
  }
};

const StockList = () => {
  const [stocks, setStocks] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [isSearchActive, setIsSearchActive] = useState(false);
  const [isInputFocused, setIsInputFocused] = useState(false);

  const NAVBAR_HEIGHT = 85;

  useEffect(() => {
    getStocks()
      .then(res => {
        setStocks(res.data.stocks);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch stocks:", err);
        setLoading(false);
      });
  }, []);

  const filteredStocks = stocks
    ? Object.entries(stocks).filter(([key, name]) =>
        name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        key.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : [];

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
          justifyContent: 'flex-start',
          color: '#fff',
          paddingTop: `${NAVBAR_HEIGHT + 24}px`,
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
          component={motion.div}
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          sx={{
            zIndex: 2,
            position: 'relative',
            py: 4,
            maxWidth: { xs: '95%', sm: '80%', md: '700px' },
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            backdropFilter: 'blur(5px)',
            borderRadius: '12px',
            boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.2)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            my: 4,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '300px', width: '100%' }}>
              <CircularProgress color="inherit" />
            </Box>
          ) : (
            <>
              {/* Overall container for Title and Search Area. This handles hover.
                  No longer needs position: 'relative' as the button is relative to a nested Box. */}
              <Box
                sx={{
                  width: '100%',
                  display: 'flex',
                  flexDirection: 'column', // Stack title line and search bar vertically
                  alignItems: 'center', // Center content horizontally
                  mb: 3, // Margin below this entire search component area
                }}
                onMouseEnter={() => setIsSearchActive(true)}
                onMouseLeave={() => {
                  // Only deactivate if input is NOT focused AND search term is empty
                  if (!isInputFocused && searchTerm === '') {
                    setIsSearchActive(false);
                  }
                }}
              >
                {/* NEW: This Box specifically defines the line for the Title and Button */}
                <Box
                  sx={{
                    width: '100%', // Takes full width to allow centering
                    display: 'flex',
                    justifyContent: 'center', // Centers the Typography within this line
                    alignItems: 'center', // Vertically aligns content in this line
                    position: 'relative', // IMPORTANT: This is the new relative parent for the absolute button
                    minHeight: '40px', // Ensures a consistent height for the title line
                  }}
                >
                  {/* Title - centered within this new inner Box */}
                  <MotionTypography
                    variant="h4"
                    sx={{
                      fontWeight: 'bold',
                      fontFamily: `'Poppins', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"`,
                      textAlign: 'center',
                      mb: 0,
                      textShadow: '0 0 8px rgba(0,0,0,0.7)',
                    }}
                    variants={itemVariants}
                  >
                    💲Available Stocks
                  </MotionTypography>

                  {/* Search Button - now positioned absolutely relative to the new inner Box */}
                  <AnimatePresence mode="wait" initial={false}>
                    {!isSearchActive && (
                      <motion.div
                        key="searchButton"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        transition={{ duration: 0.2 }}
                        sx={{
                          position: 'absolute', // Absolute positioning relative to its new parent
                          right: 0, // Aligns to the right edge of its new parent
                          top: '50%', // Aligns to the vertical middle of its new parent
                          transform: 'translateY(-50%)', // Adjusts for the button's own height to perfectly center
                          zIndex: 1,
                        }}
                      >
                        <IconButton
                          sx={{ color: 'rgba(255,255,255,0.7)', '&:hover': { color: '#00bcd4' } }}
                          aria-label="Open search bar"
                        >
                          <SearchIcon />
                        </IconButton>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </Box>

                {/* Expanded Search Bar (appears below the new title/button line) */}
                <AnimatePresence initial={false}>
                  {isSearchActive && (
                    <motion.div
                      key="expandedSearchBar"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: '56px' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3, ease: "easeOut" }}
                      style={{ width: '100%' }}
                      sx={{ mt: 1 }} // Added a small margin-top for separation
                    >
                      <TextField
                        id="stock-search-input"
                        variant="outlined"
                        fullWidth
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        onFocus={() => setIsInputFocused(true)}
                        onBlur={() => {
                          setIsInputFocused(false);
                          if (searchTerm === '') {
                            setIsSearchActive(false);
                          }
                        }}
                        InputProps={{
                          startAdornment: (
                            <InputAdornment position="start">
                              <SearchIcon sx={{ color: 'rgba(255,255,255,0.7)' }} />
                            </InputAdornment>
                          ),
                          placeholder: "Search Stocks (e.g., RELIANCE, TCS)",
                          endAdornment: searchTerm && (
                            <InputAdornment position="end">
                              <IconButton
                                onClick={() => setSearchTerm('')}
                                sx={{ color: 'rgba(255,255,255,0.7)', '&:hover': { color: '#fff' } }}
                                aria-label="Clear search"
                              >
                                <ClearIcon />
                              </IconButton>
                            </InputAdornment>
                          ),
                          sx: {
                              color: '#fff',
                              fontFamily: `'Inter', sans-serif`,
                              backgroundColor: 'rgba(255, 255, 255, 0.08)',
                              borderRadius: '8px',
                              '&::placeholder': {
                                  color: 'rgba(255,255,255,0.5)',
                                  opacity: 1,
                              },
                          },
                        }}
                        sx={{
                          height: '100%',
                          '& .MuiOutlinedInput-root': {
                            height: '100%',
                            borderRadius: '8px',
                            '& fieldset': {
                              borderColor: 'rgba(255,255,255,0.2)',
                              transition: 'border-color 0.3s ease',
                            },
                            '&:hover fieldset': {
                              borderColor: 'rgba(255,255,255,0.5)',
                            },
                            '&.Mui-focused fieldset': {
                              borderColor: '#00bcd4',
                              boxShadow: '0 0 8px rgba(0, 188, 212, 0.5)',
                            },
                          },
                        }}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </Box>

              {filteredStocks.length === 0 && !loading && (
                <MotionTypography variants={itemVariants} sx={{ textAlign: 'center', color: 'rgba(255,255,255,0.8)' }}>
                  No stocks found matching your search.
                </MotionTypography>
              )}

              {filteredStocks.length > 0 && (
                <MotionList
                  sx={{ mt: 2, width: '100%' }}
                  variants={containerVariants}
                >
                  {filteredStocks.map(([key, name]) => (
                    <MotionListItemButton
                      key={key}
                      component={Link}
                      to={`/stocks/${key}`}
                      sx={{
                        mb: 1,
                        borderRadius: '8px',
                        backgroundColor: 'rgba(255, 255, 255, 0.08)',
                        transition: 'background-color 0.3s ease, transform 0.2s ease',
                        '&:hover': {
                          backgroundColor: 'rgba(255, 255, 255, 0.15)',
                          transform: 'translateY(-2px)',
                        },
                      }}
                      variants={itemVariants}
                    >
                      <ListItemText
                        primary={
                          <Typography
                            variant="h6"
                            sx={{
                              color: '#fff',
                              fontFamily: `'Inter', sans-serif`,
                              fontWeight: 500,
                              textShadow: '0 0 5px rgba(0,0,0,0.5)',
                            }}
                          >
                            {name}
                          </Typography>
                        }
                        secondary={
                          <Typography
                            variant="body2"
                            sx={{
                              color: '#b0e0e6',
                              fontFamily: `'Inter', sans-serif`,
                              opacity: 0.8,
                              textShadow: '0 0 3px rgba(0,0,0,0.3)',
                            }}
                          >
                            ({key})
                          </Typography>
                        }
                      />
                    </MotionListItemButton>
                  ))}
                </MotionList>
              )}
            </>
          )}
        </Container>
      </Box>
    </>
  );
};

export default StockList;