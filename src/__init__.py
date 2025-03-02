"""Zero Waste Data Cleaning Project."""

import logging
import os

# Set up logging
log_file = os.path.join('logs', 'pipeline.log')
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('src')
logger.info('Initializing Zero Waste Data Cleaning Project v1.0.0')
