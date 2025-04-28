from datetime import datetime
from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    """User model for authentication and access control"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    role = db.Column(db.String(20), nullable=False, default='analyst')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Transaction(db.Model):
    """Transaction data model"""
    id = db.Column(db.String(64), primary_key=True)
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    customer_id = db.Column(db.String(64), nullable=False)
    merchant_id = db.Column(db.String(64), nullable=False)
    payment_method = db.Column(db.String(20), nullable=False)
    card_present = db.Column(db.Boolean, default=False)
    ip_address = db.Column(db.String(45), nullable=True)
    device_id = db.Column(db.String(64), nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    
    # Model prediction results
    fraud_score = db.Column(db.Float, nullable=True)
    is_fraud = db.Column(db.Boolean, nullable=True)
    reviewed = db.Column(db.Boolean, default=False)
    review_result = db.Column(db.Boolean, nullable=True)
    reviewed_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<Transaction {self.id}>'

class ModelVersion(db.Model):
    """Model version tracking"""
    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(32), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    active = db.Column(db.Boolean, default=False)
    autoencoder_path = db.Column(db.String(256), nullable=False)
    gbm_path = db.Column(db.String(256), nullable=False)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    def __repr__(self):
        return f'<ModelVersion {self.version}>'

class AuditLog(db.Model):
    """Audit logging for compliance"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    action = db.Column(db.String(32), nullable=False)
    resource_type = db.Column(db.String(32), nullable=False)
    resource_id = db.Column(db.String(64), nullable=False)
    details = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    
    def __repr__(self):
        return f'<AuditLog {self.id}>'

class AlertConfig(db.Model):
    """Configuration for fraud alerts"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    description = db.Column(db.Text, nullable=True)
    condition = db.Column(db.Text, nullable=False) # JSON condition
    threshold = db.Column(db.Float, nullable=False)
    active = db.Column(db.Boolean, default=True)
    notification_emails = db.Column(db.Text, nullable=True) # Comma-separated list
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<AlertConfig {self.name}>'

class ModelMetrics(db.Model):
    """Daily model performance metrics"""
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    model_version_id = db.Column(db.Integer, db.ForeignKey('model_version.id'), nullable=False)
    transactions_processed = db.Column(db.Integer, default=0)
    avg_latency_ms = db.Column(db.Float, nullable=True)
    fraud_detected = db.Column(db.Integer, default=0)
    false_positives = db.Column(db.Integer, default=0)
    false_negatives = db.Column(db.Integer, default=0)
    kl_divergence = db.Column(db.Float, nullable=True) # Model drift metric
    
    def __repr__(self):
        return f'<ModelMetrics {self.date}>'
