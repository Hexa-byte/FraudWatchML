import logging
import time
from datetime import datetime
import json
from flask import Blueprint, request, jsonify, current_app
from models import Transaction, AuditLog, db, User
from utils import get_active_model, log_audit, validate_request

# Import data_pipeline without tensorflow dependency
from data_pipeline import DataPipeline

# Configure logging
logger = logging.getLogger(__name__)

# Create API blueprint
api_bp = Blueprint('api', __name__)

# Initialize data pipeline
data_pipeline = DataPipeline()

# Global model cache
_model_cache = None

def get_model():
    """Get or initialize the fraud detection model
    
    Returns:
        HybridFraudModel: Initialized hybrid model
    """
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    try:
        # Lazy import to avoid loading TensorFlow at startup
        from model_architecture import HybridFraudModel, AutoencoderModel, GradientBoostedTreesModel
        from config import FEATURE_COLUMNS
        
        # Get active model version from database
        model_version = get_active_model()
        
        if not model_version:
            logger.error("No active model found in database")
            return None
        
        # Load autoencoder and GBM models
        autoencoder = AutoencoderModel(input_dim=len(FEATURE_COLUMNS), 
                                      model_path=model_version.autoencoder_path)
        
        gbm = GradientBoostedTreesModel(model_path=model_version.gbm_path)
        
        # Create hybrid model
        _model_cache = HybridFraudModel(autoencoder=autoencoder, 
                                        gbm=gbm, 
                                        autoencoder_weight=0.7)
        
        logger.info(f"Loaded model version {model_version.version}")
        return _model_cache
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

@api_bp.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

@api_bp.route('/score', methods=['POST'])
def score_transaction():
    """Score a transaction for fraud
    
    Request body:
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
    
    Returns:
        JSON response with fraud score and details
    """
    start_time = time.time()
    
    # Get request data
    data = request.get_json()
    
    # Validate request data
    validation_result = validate_request(data)
    if not validation_result['valid']:
        return jsonify({
            'error': 'Invalid request data',
            'details': validation_result['errors']
        }), 400
    
    # Extract transaction data
    transaction_id = data.get('transaction_id')
    
    # Check if transaction already exists
    existing_transaction = Transaction.query.filter_by(id=transaction_id).first()
    if existing_transaction:
        return jsonify({
            'error': 'Transaction already processed',
            'transaction_id': transaction_id,
            'fraud_score': existing_transaction.fraud_score,
            'is_fraudulent': existing_transaction.is_fraud
        }), 409
    
    # Get the model
    model = get_model()
    if not model:
        return jsonify({
            'error': 'Model not available',
            'transaction_id': transaction_id
        }), 500
    
    try:
        # Lazy import libraries only when needed
        import pandas as pd
        import tensorflow as tf
        
        # Prepare data for the model
        df = pd.DataFrame([data])
        
        # Add location data if available
        if 'additional_features' in data and 'location' in data['additional_features']:
            df['latitude'] = data['additional_features']['location'].get('latitude')
            df['longitude'] = data['additional_features']['location'].get('longitude')
        
        # Add other additional features
        if 'additional_features' in data:
            for key, value in data['additional_features'].items():
                if key != 'location' and not isinstance(value, dict):
                    df[key] = value
        
        # Preprocess the data
        processed_df = data_pipeline.preprocess_data(df)
        
        # Convert to TensorFlow dataset
        features = processed_df.drop(['transaction_id'], errors='ignore')
        tf_data = tf.convert_to_tensor(features.values, dtype=tf.float32)
        
        # Make prediction
        fraud_score, is_fraud = model.predict_fraud(tf_data)
        
        # Get explanation
        explanation = model.explain_prediction(tf_data[0:1], features.columns.tolist())
        
        # Build the response
        response = {
            'transaction_id': transaction_id,
            'fraud_score': float(fraud_score[0]),
            'is_fraudulent': bool(is_fraud[0]),
            'confidence': min(0.5 + abs(float(fraud_score[0]) - 0.5) * 2, 0.99),
            'explanation': explanation,
            'processing_time_ms': int((time.time() - start_time) * 1000)
        }
        
        # Save transaction to database
        try:
            transaction = Transaction(
                id=transaction_id,
                amount=data.get('amount'),
                timestamp=datetime.fromisoformat(data.get('timestamp').replace('Z', '+00:00')),
                customer_id=data.get('customer_id'),
                merchant_id=data.get('merchant_id'),
                payment_method=data.get('payment_method'),
                card_present=data.get('card_present', False),
                ip_address=data.get('additional_features', {}).get('ip_address'),
                device_id=data.get('additional_features', {}).get('device_id'),
                latitude=df.get('latitude', [None])[0],
                longitude=df.get('longitude', [None])[0],
                fraud_score=float(fraud_score[0]),
                is_fraud=bool(is_fraud[0])
            )
            
            db.session.add(transaction)
            db.session.commit()
            
            # Log the transaction
            log_audit(
                action="score_transaction",
                resource_type="transaction",
                resource_id=transaction_id,
                details=json.dumps(response)
            )
            
        except Exception as e:
            logger.error(f"Error saving transaction: {e}")
            db.session.rollback()
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing transaction: {e}")
        return jsonify({
            'error': 'Error processing transaction',
            'details': str(e),
            'transaction_id': transaction_id
        }), 500

@api_bp.route('/transactions/<transaction_id>/review', methods=['POST'])
def review_transaction(transaction_id):
    """Review a transaction (confirm/reject fraud flag)
    
    Request body:
    {
        "is_fraud": true,
        "reviewer_id": 123,
        "notes": "Confirmed fraud case"
    }
    
    Returns:
        JSON response with updated transaction details
    """
    # Get request data
    data = request.get_json()
    
    # Validate request
    if 'is_fraud' not in data or 'reviewer_id' not in data:
        return jsonify({
            'error': 'Missing required fields',
            'required': ['is_fraud', 'reviewer_id']
        }), 400
    
    # Get transaction
    transaction = Transaction.query.get(transaction_id)
    if not transaction:
        return jsonify({
            'error': 'Transaction not found',
            'transaction_id': transaction_id
        }), 404
    
    # Verify reviewer exists
    reviewer = User.query.get(data['reviewer_id'])
    if not reviewer:
        return jsonify({
            'error': 'Reviewer not found',
            'reviewer_id': data['reviewer_id']
        }), 404
    
    try:
        # Update transaction
        transaction.reviewed = True
        transaction.review_result = data['is_fraud']
        transaction.reviewed_by = data['reviewer_id']
        transaction.reviewed_at = datetime.utcnow()
        
        db.session.commit()
        
        # Log the review
        log_audit(
            action="review_transaction",
            resource_type="transaction",
            resource_id=transaction_id,
            details=json.dumps(data),
            user_id=data['reviewer_id']
        )
        
        return jsonify({
            'transaction_id': transaction_id,
            'reviewed': True,
            'is_fraud': transaction.review_result,
            'reviewed_by': transaction.reviewed_by,
            'reviewed_at': transaction.reviewed_at.isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error reviewing transaction: {e}")
        db.session.rollback()
        return jsonify({
            'error': 'Error reviewing transaction',
            'details': str(e),
            'transaction_id': transaction_id
        }), 500

@api_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """Get information about the active model"""
    model_version = get_active_model()
    
    if not model_version:
        return jsonify({
            'error': 'No active model found'
        }), 404
    
    return jsonify({
        'version': model_version.version,
        'created_at': model_version.created_at.isoformat(),
        'precision': model_version.precision,
        'recall': model_version.recall,
        'active': model_version.active
    })

@api_bp.route('/transactions/recent', methods=['GET'])
def get_recent_transactions():
    """Get recent transactions
    
    Query parameters:
        limit (int): Maximum number of transactions to return (default: 100)
        offset (int): Offset for pagination (default: 0)
        fraud_only (bool): Whether to return only fraud transactions (default: false)
        
    Returns:
        JSON response with transaction list
    """
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    fraud_only = request.args.get('fraud_only', 'false').lower() == 'true'
    
    # Build query
    query = Transaction.query
    if fraud_only:
        query = query.filter(Transaction.is_fraud == True)
    
    # Get total count
    total_count = query.count()
    
    # Get transactions
    transactions = query.order_by(Transaction.timestamp.desc()) \
                       .limit(limit).offset(offset).all()
    
    # Build response
    result = {
        'transactions': [],
        'total_count': total_count,
        'limit': limit,
        'offset': offset
    }
    
    for tx in transactions:
        result['transactions'].append({
            'id': tx.id,
            'amount': tx.amount,
            'timestamp': tx.timestamp.isoformat(),
            'customer_id': tx.customer_id,
            'merchant_id': tx.merchant_id,
            'payment_method': tx.payment_method,
            'fraud_score': tx.fraud_score,
            'is_fraud': tx.is_fraud,
            'reviewed': tx.reviewed,
            'review_result': tx.review_result
        })
    
    return jsonify(result)
