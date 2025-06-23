import { Box, Heading, Text, Button } from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';

const StockCard = ({ symbol, name }) => (
  <Box p={4} borderWidth="1px" borderRadius="md">
    <Heading size="md">{name}</Heading>
    <Text>{symbol}</Text>
    <Button mt={4} as={RouterLink} to={`/stocks/${symbol}`} colorScheme="teal">
      Details
    </Button>
  </Box>
);

export default StockCard;
