import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Application settings
APP_NAME = "FraudWatch"
APP_VERSION = "1.0.0"

# Database settings
DATABASE_URL = os.environ.get("DATABASE_URL")

# Model training settings
TRAINING_EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Feature definitions
FEATURE_COLUMNS = [
    'amount',
    'payment_method',
    'card_present',
    'hour_of_day',
    'day_of_week',
    'weekend',
    'ip_address',
    'device_id',
    'latitude',
    'longitude'
]

NUMERIC_COLUMNS = [
    'amount',
    'hour_of_day',
    'day_of_week',
    'weekend',
    'latitude',
    'longitude'
]

CATEGORICAL_COLUMNS = [
    'payment_method',
    'card_present',
    'device_id',
    'ip_address'
]

# Model hyperparameters
AUTOENCODER_LAYERS = [64, 32, 16, 32, 64]
AUTOENCODER_LATENT_DIM = 16
AUTOENCODER_LEARNING_RATE = 0.001

GBM_NUM_TREES = 200
GBM_MAX_DEPTH = 8
GBM_MIN_EXAMPLES = 20
GBM_LEARNING_RATE = 0.1

# Hybrid model configuration
AUTOENCODER_WEIGHT = 0.7
GBM_WEIGHT = 0.3
FRAUD_THRESHOLD = 0.5

# API configuration
API_RATE_LIMIT = 100  # requests per minute
API_TIMEOUT = 30  # seconds

# Security settings
PASSWORD_MIN_LENGTH = 8
SESSION_TIMEOUT = 3600  # 1 hour
BCRYPT_LOG_ROUNDS = 12

# Monitoring settings
ALERT_EMAIL_SENDER = "alerts@fraudwatch.example.com"
METRICS_RETENTION_DAYS = 90
