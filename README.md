# FraudWatch: Enterprise Fraud Detection System

FraudWatch is an enterprise-grade real-time fraud detection system using TensorFlow with a hybrid model architecture, designed for banking and financial services. The system employs machine learning to detect suspicious transactions and provide compliance officers with explainable fraud detection results.

## Features

- **Real-time fraud detection**: Score transactions as they occur using a hybrid ML model
- **Hybrid architecture**: Combines unsupervised anomaly detection (Autoencoder) with supervised classification (Gradient Boosted Trees)
- **Explainable AI**: Provides detailed explanations for fraud determinations to aid compliance reviews
- **User interface**: Complete web interface for monitoring, alerts, and transaction reviews
- **API**: RESTful API for integration with other systems
- **Comprehensive analytics**: Monitor system performance and fraud patterns

## System Requirements

- Python 3.11+
- PostgreSQL database
- TensorFlow 2.x
- Flask web framework

## Getting Started

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - `DATABASE_URL`: PostgreSQL connection string
   - `SESSION_SECRET`: Secret key for session security

### Initialization

Initialize the database with sample data:

```bash
python initialize_data.py
```

This creates:
- Admin user (username: admin, password: admin123)
- Sample model version
- Test transaction data (mix of legitimate and fraudulent)

### Running the Application

Start the application with:

```bash
gunicorn --bind 0.0.0.0:5000 main:app
```

Or for development:

```bash
python main.py
```

Access the web interface at: http://localhost:5000

### Using the API

FraudWatch provides a RESTful API for scoring transactions and retrieving information:

#### Score a Transaction

```
POST /api/v1/score
```

Request body:
```json
{
  "transaction_id": "tx_123456789",
  "amount": 1500.00,
  "merchant_id": "merch_12345",
  "timestamp": "2023-05-01T12:34:56Z",
  "customer_id": "cust_987654",
  "payment_method": "card",
  "card_present": false,
  "additional_features": {
    "ip_address": "192.168.1.1",
    "device_id": "dev_abcdef",
    "location": {
      "latitude": 37.7749,
      "longitude": -122.4194
    }
  }
}
```

## Architecture

FraudWatch uses a hybrid model architecture that combines:

1. **Unsupervised anomaly detection** (Autoencoder): Identifies unusual patterns that don't match legitimate transaction patterns
2. **Supervised classification** (Gradient Boosted Trees): Classifies transactions based on known fraud patterns

This hybrid approach provides excellent performance for both known fraud patterns and novel fraud strategies.

### Components

- **Data Pipeline**: Preprocessing, feature engineering, and data validation
- **Model Architecture**: Implementation of autoencoder and GBM models
- **API Service**: RESTful interface for transaction scoring
- **Web Interface**: Dashboard, transaction review, and administration
- **Monitoring**: System performance and model drift detection
- **Security**: Authentication, authorization, and audit logging

## Development

### Project Structure

```
├── api.py                # API endpoints
├── app.py                # Flask application setup
├── config.py             # Configuration settings
├── data_pipeline.py      # Data preprocessing pipeline
├── fin_data.py           # Sample data generation
├── initialize_data.py    # Database initialization
├── main.py               # Application entry point
├── ml_pipeline.py        # Machine learning training pipeline
├── model_architecture.py # Model implementation
├── models.py             # Database models
├── monitoring.py         # System monitoring endpoints
├── routes.py             # Web UI routes
├── security.py           # Authentication and authorization
├── static/               # Static assets (CSS, JS, images)
├── templates/            # HTML templates
├── tests/                # Unit and integration tests
└── utils.py              # Utility functions
```

### Testing

Run unit tests with:

```bash
python -m unittest discover tests
```

## Compliance and Security

FraudWatch is designed to meet requirements for:
- GDPR
- PCI DSS 
- Sarbanes-Oxley (SOX)
- CCPA

## License

Copyright © 2025 FraudWatch Enterprise Fraud Detection