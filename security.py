import os
import logging
import secrets
import hashlib
from functools import wraps
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, redirect, url_for, render_template, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import User, AuditLog, db
from utils import log_audit

logger = logging.getLogger(__name__)

# Create security blueprint
security_bp = Blueprint('security', __name__)

# Initialize flask-login
login_manager = LoginManager()

def init_app(app):
    """Initialize security for app
    
    Args:
        app: Flask application
    """
    login_manager.init_app(app)
    login_manager.login_view = 'security.login'
    login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    """Load user for flask-login
    
    Args:
        user_id: User ID to load
        
    Returns:
        User object or None
    """
    return User.query.get(int(user_id))

def rbac_required(roles):
    """Role-based access control decorator
    
    Args:
        roles: List of allowed roles
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if current_user.role not in roles:
                log_audit(
                    action="access_denied",
                    resource_type="endpoint",
                    resource_id=request.path,
                    user_id=current_user.id,
                    details=f"Role {current_user.role} not in {roles}"
                )
                return jsonify({'error': 'Access denied'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@security_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login endpoint"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            
            # Update last login time
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            # Log the login
            log_audit(
                action="login",
                resource_type="user",
                resource_id=user.id,
                user_id=user.id,
                details=f"User {username} logged in",
                ip_address=request.remote_addr
            )
            
            next_page = request.args.get('next', url_for('monitoring.dashboard'))
            return redirect(next_page)
        
        flash('Invalid username or password', 'danger')
        
        # Log failed login attempt
        log_audit(
            action="login_failed",
            resource_type="user",
            resource_id=username,
            details=f"Failed login attempt for {username}",
            ip_address=request.remote_addr
        )
    
    return render_template('login.html')

@security_bp.route('/logout')
@login_required
def logout():
    """User logout endpoint"""
    username = current_user.username
    user_id = current_user.id
    
    logout_user()
    
    # Log the logout
    log_audit(
        action="logout",
        resource_type="user",
        resource_id=user_id,
        user_id=user_id,
        details=f"User {username} logged out",
        ip_address=request.remote_addr
    )
    
    return redirect(url_for('security.login'))

@security_bp.route('/profile')
@login_required
def profile():
    """User profile page"""
    return render_template('profile.html', user=current_user)

@security_bp.route('/users', methods=['GET'])
@rbac_required(['admin'])
def list_users():
    """Admin endpoint to list users"""
    users = User.query.all()
    
    user_list = []
    for user in users:
        user_list.append({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'is_active': user.is_active,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None
        })
    
    return jsonify({'users': user_list})

@security_bp.route('/users', methods=['POST'])
@rbac_required(['admin'])
def create_user():
    """Admin endpoint to create a new user"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['username', 'email', 'password', 'role']
    for field in required_fields:
        if field not in data:
            return jsonify({
                'error': f'Missing required field: {field}'
            }), 400
    
    # Validate role
    allowed_roles = ['admin', 'analyst', 'viewer']
    if data['role'] not in allowed_roles:
        return jsonify({
            'error': f'Invalid role. Must be one of: {", ".join(allowed_roles)}'
        }), 400
    
    # Check if user already exists
    existing_user = User.query.filter(
        (User.username == data['username']) | (User.email == data['email'])
    ).first()
    
    if existing_user:
        return jsonify({
            'error': 'User with this username or email already exists'
        }), 409
    
    # Create new user
    try:
        user = User(
            username=data['username'],
            email=data['email'],
            role=data['role'],
            is_active=data.get('is_active', True)
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        # Log the action
        log_audit(
            action="create_user",
            resource_type="user",
            resource_id=user.id,
            user_id=current_user.id,
            details=f"User {data['username']} created with role {data['role']}"
        )
        
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at.isoformat()
        }), 201
    
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        db.session.rollback()
        return jsonify({
            'error': 'Error creating user',
            'details': str(e)
        }), 500

@security_bp.route('/users/<int:user_id>', methods=['PUT'])
@rbac_required(['admin'])
def update_user(user_id):
    """Admin endpoint to update a user"""
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    
    # Prevent updating own role (admin can't demote themselves)
    if current_user.id == user_id and 'role' in data and data['role'] != 'admin':
        return jsonify({
            'error': 'Cannot change your own admin role'
        }), 403
    
    try:
        # Update user fields
        for field in ['email', 'role', 'is_active']:
            if field in data:
                setattr(user, field, data[field])
        
        # Update password if provided
        if 'password' in data:
            user.set_password(data['password'])
        
        db.session.commit()
        
        # Log the action
        log_audit(
            action="update_user",
            resource_type="user",
            resource_id=user_id,
            user_id=current_user.id,
            details=f"User {user.username} updated"
        )
        
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'is_active': user.is_active
        })
    
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        db.session.rollback()
        return jsonify({
            'error': 'Error updating user',
            'details': str(e)
        }), 500

@security_bp.route('/audit-logs', methods=['GET'])
@rbac_required(['admin'])
def get_audit_logs():
    """Admin endpoint to get audit logs"""
    # Get parameters
    days = request.args.get('days', 7, type=int)
    limit = request.args.get('limit', 100, type=int)
    user_id = request.args.get('user_id', type=int)
    action = request.args.get('action')
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Build query
    query = AuditLog.query.filter(AuditLog.timestamp >= start_date)
    
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    
    if action:
        query = query.filter(AuditLog.action == action)
    
    # Get logs
    logs = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    
    # Format response
    log_list = []
    for log in logs:
        log_list.append({
            'id': log.id,
            'timestamp': log.timestamp.isoformat(),
            'user_id': log.user_id,
            'action': log.action,
            'resource_type': log.resource_type,
            'resource_id': log.resource_id,
            'details': log.details,
            'ip_address': log.ip_address
        })
    
    return jsonify({'audit_logs': log_list})

@security_bp.route('/api-tokens', methods=['POST'])
@rbac_required(['admin', 'analyst'])
def create_api_token():
    """Create an API token for automated access"""
    # Generate a secure token
    token = secrets.token_hex(32)
    
    # Store token hash (in a real system you'd store this in the database)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    
    # Set an expiration (30 days)
    expires = datetime.utcnow() + timedelta(days=30)
    
    # In a real system, you'd store this in the database
    # For now, store in session (this is not suitable for production!)
    if 'api_tokens' not in session:
        session['api_tokens'] = []
    
    session['api_tokens'].append({
        'token_hash': token_hash,
        'user_id': current_user.id,
        'created_at': datetime.utcnow().isoformat(),
        'expires_at': expires.isoformat()
    })
    
    # Log the action
    log_audit(
        action="create_api_token",
        resource_type="api_token",
        resource_id=token_hash[:8],  # Store just a prefix for identification
        user_id=current_user.id,
        details="API token created",
        ip_address=request.remote_addr
    )
    
    return jsonify({
        'token': token,  # Only returned once!
        'expires_at': expires.isoformat()
    })

def encrypt_sensitive_data(data):
    """Encrypt sensitive data for storage
    
    Args:
        data: Data to encrypt
        
    Returns:
        Encrypted data
    """
    # In a real system, you'd use proper encryption (AES-256)
    # This is a placeholder implementation
    return f"ENCRYPTED:{data}"

def decrypt_sensitive_data(encrypted_data):
    """Decrypt sensitive data
    
    Args:
        encrypted_data: Encrypted data
        
    Returns:
        Decrypted data
    """
    # In a real system, you'd use proper decryption
    # This is a placeholder implementation
    if encrypted_data.startswith("ENCRYPTED:"):
        return encrypted_data[10:]
    return encrypted_data
