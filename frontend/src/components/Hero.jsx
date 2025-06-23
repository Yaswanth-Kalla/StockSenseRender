import { Box, Heading, Text, Button, Stack } from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';

const Hero = () => (
  <Box bg="blue.600" color="white" py={20} textAlign="center">
    <Heading fontSize="4xl">Predict Your Stock Moves 📈</Heading>
    <Text fontSize="xl" mt={4}>
      Harness the power of LSTM models to forecast the next price move for top Indian stocks.
    </Text>
    <Stack direction="row" justify="center" mt={6} spacing={4}>
      <Button
        as={RouterLink}
        to="/stocks"
        colorScheme="green"
        size="lg"
        transition="all 0.3s ease-in-out"
        _hover={{
          // Enhanced green glow for the outline
          boxShadow: '0 0 25px 8px rgba(72, 187, 120, 0.9)', // Increased blur, spread, and opacity
          // Optional: Add a subtle text shadow for a shining text effect
          textShadow: '0 0 5px rgba(255, 255, 255, 0.7)',
          transform: 'scale(1.02)',
        }}
      >
        View Stocks
      </Button>
      <Button
        as={RouterLink}
        to="/predict"
        colorScheme="yellow"
        size="lg"
        transition="all 0.3s ease-in-out"
        _hover={{
          // Enhanced yellow glow for the outline
          boxShadow: '0 0 25px 8px rgba(236, 201, 75, 0.9)', // Increased blur, spread, and opacity
          // Optional: Add a subtle text shadow for a shining text effect
          textShadow: '0 0 5px rgba(255, 255, 255, 0.7)',
          transform: 'scale(1.02)',
        }}
      >
        Make Prediction
      </Button>
    </Stack>
  </Box>
);

export default Hero;