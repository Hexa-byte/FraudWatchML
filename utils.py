import logging
import json
import numpy as np
from datetime import datetime
from flask import request, current_app, g
from models import ModelVersion, AuditLog, db
from sqlalchemy import desc

logger = logging.getLogger(__name__)

def get_active_model():
    """Get the currently active model version
    
    Returns:
        ModelVersion: Active model version or None
    """
    # Check if we already have it in the request context
    if hasattr(g, 'active_model'):
        return g.active_model
    
    # Query the active model
    active_model = ModelVersion.query.filter_by(active=True).order_by(desc(ModelVersion.created_at)).first()
    
    # Store in request context
    g.active_model = active_model
    
    return active_model

def log_audit(action, resource_type, resource_id, details=None, user_id=None, ip_address=None):
    """Log an action to the audit log
    
    Args:
        action (str): The action being performed
        resource_type (str): Type of resource (user, transaction, etc.)
        resource_id: ID of the resource
        details (str, optional): Additional details about the action
        user_id (int, optional): User ID performing the action
        ip_address (str, optional): IP address of the request
    """
    try:
        # Get IP address from request if not provided
        if ip_address is None and request:
            ip_address = request.remote_addr
        
        # Create audit log entry
        audit_log = AuditLog(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id),
            details=details,
            ip_address=ip_address
        )
        
        db.session.add(audit_log)
        db.session.commit()
        
    except Exception as e:
        logger.error(f"Error logging audit entry: {e}")
        db.session.rollback()

def validate_request(data):
    """Validate transaction request data
    
    Args:
        data (dict): Request data to validate
        
    Returns:
        dict: Validation result
    """
    errors = []
    
    # Check required fields
    required_fields = ['transaction_id', 'amount', 'timestamp', 'customer_id', 'merchant_id', 'payment_method']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate transaction_id format
    if 'transaction_id' in data and not data['transaction_id'].strip():
        errors.append("transaction_id cannot be empty")
    
    # Validate amount
    if 'amount' in data:
        try:
            amount = float(data['amount'])
            if amount <= 0:
                errors.append("amount must be greater than 0")
        except (ValueError, TypeError):
            errors.append("amount must be a valid number")
    
    # Validate timestamp
    if 'timestamp' in data:
        try:
            datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        except (ValueError, TypeError):
            errors.append("timestamp must be a valid ISO 8601 date string")
    
    # Validate customer_id
    if 'customer_id' in data and not data['customer_id'].strip():
        errors.append("customer_id cannot be empty")
    
    # Validate merchant_id
    if 'merchant_id' in data and not data['merchant_id'].strip():
        errors.append("merchant_id cannot be empty")
    
    # Validate payment_method
    valid_payment_methods = ['card', 'bank_transfer', 'wallet', 'crypto', 'other']
    if 'payment_method' in data and data['payment_method'] not in valid_payment_methods:
        errors.append(f"payment_method must be one of: {', '.join(valid_payment_methods)}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def generate_shap_values(model, data):
    """Generate SHAP values for model explanation
    
    Args:
        model: Trained model
        data: Input data
        
    Returns:
        dict: SHAP values and feature names
    """
    # In a real system, this would use the SHAP library
    # This is a simplified placeholder implementation
    feature_names = data.columns.tolist()
    num_features = len(feature_names)
    
    # Generate random values as placeholder
    # In a real system, these would be actual SHAP values
    shap_values = np.random.randn(1, num_features)
    
    # Normalize to -1 to 1 range
    shap_values = shap_values / np.max(np.abs(shap_values))
    
    # Convert to list for JSON serialization
    shap_list = shap_values.tolist()[0]
    
    # Create result dictionary
    result = {
        'shap_values': dict(zip(feature_names, shap_list)),
        'base_value': 0.5,  # Placeholder base value
    }
    
    return result

def normalize_score(score, min_val=0.0, max_val=1.0):
    """Normalize a score to a specific range
    
    Args:
        score (float): Score to normalize
        min_val (float): Minimum value in range
        max_val (float): Maximum value in range
        
    Returns:
        float: Normalized score
    """
    # Clip to range
    score = max(min(score, max_val), min_val)
    
    # Normalize to 0-1
    return (score - min_val) / (max_val - min_val)

def format_currency(amount):
    """Format a currency amount
    
    Args:
        amount (float): Amount to format
        
    Returns:
        str: Formatted currency string
    """
    return f"${amount:,.2f}"
