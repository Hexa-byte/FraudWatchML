import logging
from flask import render_template, redirect, url_for, flash, request
from app import app
from flask_login import login_required, current_user
from models import Transaction, AlertConfig, db
from security import rbac_required
from utils import get_active_model

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Home page route"""
    return redirect(url_for('monitoring.dashboard'))

@app.route('/transactions')
@login_required
def transactions():
    """Transaction list page"""
    # Get query parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    fraud_only = request.args.get('fraud_only', 'false').lower() == 'true'
    
    # Build query
    query = Transaction.query
    if fraud_only:
        query = query.filter(Transaction.is_fraud == True)
    
    # Get paginated transactions
    transactions = query.order_by(Transaction.timestamp.desc()).paginate(
        page=page, per_page=per_page
    )
    
    return render_template(
        'transaction_review.html',
        transactions=transactions,
        fraud_only=fraud_only
    )

@app.route('/transactions/<transaction_id>')
@login_required
def transaction_detail(transaction_id):
    """Transaction detail page"""
    transaction = Transaction.query.get_or_404(transaction_id)
    
    return render_template(
        'transaction_detail.html',
        transaction=transaction
    )

@app.route('/transactions/<transaction_id>/review', methods=['POST'])
@login_required
@rbac_required(['admin', 'analyst'])
def review_transaction_form(transaction_id):
    """Review a transaction from the web interface"""
    transaction = Transaction.query.get_or_404(transaction_id)
    
    # Get form data
    is_fraud = request.form.get('is_fraud') == 'true'
    notes = request.form.get('notes', '')
    
    # Update transaction
    transaction.reviewed = True
    transaction.review_result = is_fraud
    transaction.reviewed_by = current_user.id
    
    db.session.commit()
    
    flash(f"Transaction {transaction_id} marked as {'fraudulent' if is_fraud else 'legitimate'}", 'success')
    return redirect(url_for('transactions'))

@app.route('/alerts')
@login_required
@rbac_required(['admin', 'analyst'])
def alerts():
    """Alert configuration page"""
    alert_configs = AlertConfig.query.all()
    
    return render_template(
        'alerts.html',
        alert_configs=alert_configs
    )

@app.route('/alerts/new', methods=['GET', 'POST'])
@login_required
@rbac_required(['admin', 'analyst'])
def new_alert():
    """Create a new alert configuration"""
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description', '')
        condition = request.form.get('condition')
        threshold = request.form.get('threshold', type=float)
        notification_emails = request.form.get('notification_emails', '')
        
        # Validate inputs
        if not name or not condition or threshold is None:
            flash('Please fill in all required fields', 'danger')
            return render_template('alert_form.html')
        
        # Create alert config
        alert_config = AlertConfig(
            name=name,
            description=description,
            condition=condition,
            threshold=threshold,
            notification_emails=notification_emails,
            created_by=current_user.id
        )
        
        db.session.add(alert_config)
        db.session.commit()
        
        flash(f"Alert configuration '{name}' created", 'success')
        return redirect(url_for('alerts'))
    
    return render_template('alert_form.html')

@app.route('/model/info')
@login_required
def model_info():
    """Model information page"""
    model_version = get_active_model()
    
    if not model_version:
        flash('No active model found', 'warning')
        return redirect(url_for('monitoring.dashboard'))
    
    return render_template(
        'model_info.html',
        model=model_version
    )

@app.errorhandler(404)
def page_not_found(e):
    """404 error handler"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """500 error handler"""
    logger.error(f"Server error: {e}")
    return render_template('500.html'), 500
