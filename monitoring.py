import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, render_template, current_app
from sqlalchemy import func, text
from models import Transaction, ModelMetrics, ModelVersion, db
from utils import get_active_model

logger = logging.getLogger(__name__)

# Create monitoring blueprint
monitoring_bp = Blueprint('monitoring', __name__)

@monitoring_bp.route('/dashboard', methods=['GET'])
def dashboard():
    """Render the monitoring dashboard"""
    return render_template('dashboard.html')

@monitoring_bp.route('/metrics/summary', methods=['GET'])
def get_metrics_summary():
    """Get summary metrics for the dashboard
    
    Returns:
        JSON with summary metrics
    """
    # Get time range from query parameters (default to last 30 days)
    days = request.args.get('days', 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    try:
        # Get overall stats
        total_transactions = Transaction.query.filter(Transaction.timestamp >= start_date).count()
        flagged_transactions = Transaction.query.filter(
            Transaction.timestamp >= start_date,
            Transaction.is_fraud == True
        ).count()
        
        # Calculate fraud rate
        fraud_rate = (flagged_transactions / total_transactions) * 100 if total_transactions > 0 else 0
        
        # Get false positive rate from reviewed transactions
        reviewed = Transaction.query.filter(
            Transaction.timestamp >= start_date,
            Transaction.reviewed == True
        ).count()
        
        false_positives = Transaction.query.filter(
            Transaction.timestamp >= start_date,
            Transaction.is_fraud == True,
            Transaction.reviewed == True,
            Transaction.review_result == False
        ).count()
        
        false_positive_rate = (false_positives / reviewed) * 100 if reviewed > 0 else 0
        
        # Get current model metrics
        model_version = get_active_model()
        
        # Get daily metrics for charts
        daily_metrics_query = text("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as tx_count,
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count
            FROM transaction
            WHERE timestamp >= :start_date
            GROUP BY DATE(timestamp)
            ORDER BY date
        """)
        
        daily_result = db.session.execute(daily_metrics_query, {'start_date': start_date})
        daily_stats = []
        
        for row in daily_result:
            daily_stats.append({
                'date': row.date.strftime('%Y-%m-%d'),
                'tx_count': row.tx_count,
                'fraud_count': row.fraud_count,
                'fraud_rate': (row.fraud_count / row.tx_count) * 100 if row.tx_count > 0 else 0
            })
        
        # Get payment method breakdown
        payment_method_query = text("""
            SELECT 
                payment_method,
                COUNT(*) as tx_count,
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count
            FROM transaction
            WHERE timestamp >= :start_date
            GROUP BY payment_method
        """)
        
        payment_result = db.session.execute(payment_method_query, {'start_date': start_date})
        payment_stats = []
        
        for row in payment_result:
            payment_stats.append({
                'payment_method': row.payment_method,
                'tx_count': row.tx_count,
                'fraud_count': row.fraud_count,
                'fraud_rate': (row.fraud_count / row.tx_count) * 100 if row.tx_count > 0 else 0
            })
        
        # Get model performance metrics if available
        model_performance = []
        if model_version:
            metrics_query = ModelMetrics.query.filter(
                ModelMetrics.model_version_id == model_version.id,
                ModelMetrics.date >= start_date.date()
            ).order_by(ModelMetrics.date)
            
            for metric in metrics_query:
                model_performance.append({
                    'date': metric.date.strftime('%Y-%m-%d'),
                    'transactions': metric.transactions_processed,
                    'avg_latency_ms': metric.avg_latency_ms,
                    'false_positives': metric.false_positives,
                    'false_negatives': metric.false_negatives,
                    'kl_divergence': metric.kl_divergence
                })
        
        # Build response
        response = {
            'summary': {
                'total_transactions': total_transactions,
                'flagged_transactions': flagged_transactions,
                'fraud_rate': fraud_rate,
                'false_positive_rate': false_positive_rate,
                'reviewed_transactions': reviewed,
                'false_positives': false_positives
            },
            'model': {
                'version': model_version.version if model_version else None,
                'precision': model_version.precision if model_version else None,
                'recall': model_version.recall if model_version else None
            },
            'daily_stats': daily_stats,
            'payment_method_stats': payment_stats,
            'model_performance': model_performance
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating metrics summary: {e}")
        return jsonify({
            'error': 'Error generating metrics summary',
            'details': str(e)
        }), 500

@monitoring_bp.route('/metrics/model-drift', methods=['GET'])
def get_model_drift_metrics():
    """Get model drift metrics
    
    Returns:
        JSON with model drift metrics
    """
    # Get time range from query parameters (default to last 30 days)
    days = request.args.get('days', 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    try:
        # Get active model
        model_version = get_active_model()
        if not model_version:
            return jsonify({
                'error': 'No active model found'
            }), 404
        
        # Get model drift metrics
        metrics = ModelMetrics.query.filter(
            ModelMetrics.model_version_id == model_version.id,
            ModelMetrics.date >= start_date.date()
        ).order_by(ModelMetrics.date).all()
        
        # Format response
        drift_metrics = []
        for metric in metrics:
            drift_metrics.append({
                'date': metric.date.strftime('%Y-%m-%d'),
                'kl_divergence': metric.kl_divergence,
                'transactions_processed': metric.transactions_processed
            })
        
        return jsonify({
            'model_version': model_version.version,
            'drift_metrics': drift_metrics
        })
    
    except Exception as e:
        logger.error(f"Error retrieving model drift metrics: {e}")
        return jsonify({
            'error': 'Error retrieving model drift metrics',
            'details': str(e)
        }), 500

@monitoring_bp.route('/metrics/high-value', methods=['GET'])
def get_high_value_transaction_metrics():
    """Get metrics for high-value transactions
    
    Returns:
        JSON with high-value transaction metrics
    """
    # Get parameters
    days = request.args.get('days', 30, type=int)
    threshold = request.args.get('threshold', 10000, type=float)  # Default $10,000
    start_date = datetime.utcnow() - timedelta(days=days)
    
    try:
        # Get high-value transactions
        high_value_txs = Transaction.query.filter(
            Transaction.timestamp >= start_date,
            Transaction.amount >= threshold
        ).all()
        
        # Calculate statistics
        total_count = len(high_value_txs)
        flagged_count = sum(1 for tx in high_value_txs if tx.is_fraud)
        
        # Get daily breakdown
        daily_query = text("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as count,
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as flagged
            FROM transaction
            WHERE timestamp >= :start_date AND amount >= :threshold
            GROUP BY DATE(timestamp)
            ORDER BY date
        """)
        
        daily_result = db.session.execute(daily_query, {
            'start_date': start_date,
            'threshold': threshold
        })
        
        daily_breakdown = []
        for row in daily_result:
            daily_breakdown.append({
                'date': row.date.strftime('%Y-%m-%d'),
                'count': row.count,
                'flagged': row.flagged,
                'rate': (row.flagged / row.count) * 100 if row.count > 0 else 0
            })
        
        return jsonify({
            'total_count': total_count,
            'flagged_count': flagged_count,
            'flag_rate': (flagged_count / total_count) * 100 if total_count > 0 else 0,
            'threshold': threshold,
            'daily_breakdown': daily_breakdown
        })
    
    except Exception as e:
        logger.error(f"Error retrieving high-value transaction metrics: {e}")
        return jsonify({
            'error': 'Error retrieving high-value transaction metrics',
            'details': str(e)
        }), 500

@monitoring_bp.route('/transactions/merchant-risk', methods=['GET'])
def get_merchant_risk_metrics():
    """Get risk metrics by merchant
    
    Returns:
        JSON with merchant risk metrics
    """
    # Get parameters
    days = request.args.get('days', 30, type=int)
    limit = request.args.get('limit', 10, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    try:
        # Get merchant risk metrics
        merchant_query = text("""
            SELECT 
                merchant_id,
                COUNT(*) as tx_count,
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
                AVG(amount) as avg_amount,
                MAX(fraud_score) as max_fraud_score
            FROM transaction
            WHERE timestamp >= :start_date
            GROUP BY merchant_id
            HAVING COUNT(*) > 5
            ORDER BY (SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*)) DESC
            LIMIT :limit
        """)
        
        merchant_result = db.session.execute(merchant_query, {
            'start_date': start_date,
            'limit': limit
        })
        
        merchant_metrics = []
        for row in merchant_result:
            merchant_metrics.append({
                'merchant_id': row.merchant_id,
                'tx_count': row.tx_count,
                'fraud_count': row.fraud_count,
                'fraud_rate': (row.fraud_count / row.tx_count) * 100,
                'avg_amount': row.avg_amount,
                'max_fraud_score': row.max_fraud_score
            })
        
        return jsonify({
            'merchant_risk_metrics': merchant_metrics
        })
    
    except Exception as e:
        logger.error(f"Error retrieving merchant risk metrics: {e}")
        return jsonify({
            'error': 'Error retrieving merchant risk metrics',
            'details': str(e)
        }), 500

@monitoring_bp.route('/metrics/update-daily', methods=['POST'])
def update_daily_metrics():
    """Admin endpoint to calculate and store daily metrics
    
    This would typically be scheduled as a cron job
    
    Returns:
        JSON response with operation status
    """
    if request.headers.get('X-API-Key') != os.environ.get('ADMIN_API_KEY'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Get the date to update (default to yesterday)
        target_date_str = request.args.get('date')
        if target_date_str:
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        else:
            target_date = (datetime.utcnow() - timedelta(days=1)).date()
        
        # Get active model
        model_version = get_active_model()
        if not model_version:
            return jsonify({'error': 'No active model found'}), 404
        
        # Check if metrics already exist for this date
        existing_metrics = ModelMetrics.query.filter_by(
            date=target_date,
            model_version_id=model_version.id
        ).first()
        
        if existing_metrics:
            return jsonify({
                'warning': f'Metrics for {target_date} already exist',
                'metrics_id': existing_metrics.id
            }), 200
        
        # Calculate metrics for the day
        # 1. Get all transactions for the day
        start_datetime = datetime.combine(target_date, datetime.min.time())
        end_datetime = datetime.combine(target_date, datetime.max.time())
        
        transactions = Transaction.query.filter(
            Transaction.timestamp >= start_datetime,
            Transaction.timestamp <= end_datetime
        ).all()
        
        if not transactions:
            return jsonify({'warning': f'No transactions found for {target_date}'}), 200
        
        # 2. Calculate metrics
        total_count = len(transactions)
        fraud_count = sum(1 for tx in transactions if tx.is_fraud)
        
        # Calculate average latency if available
        latencies = [tx.processing_time_ms for tx in transactions if hasattr(tx, 'processing_time_ms') and tx.processing_time_ms]
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        
        # Calculate false positives and negatives for reviewed transactions
        reviewed_txs = [tx for tx in transactions if tx.reviewed]
        false_positives = sum(1 for tx in reviewed_txs if tx.is_fraud and not tx.review_result)
        false_negatives = sum(1 for tx in reviewed_txs if not tx.is_fraud and tx.review_result)
        
        # Calculate KL divergence as a model drift metric (simplified)
        # In a real system, you'd calculate this properly using feature distributions
        kl_divergence = 0.05  # Placeholder value
        
        # 3. Create metrics record
        metrics = ModelMetrics(
            date=target_date,
            model_version_id=model_version.id,
            transactions_processed=total_count,
            avg_latency_ms=avg_latency,
            fraud_detected=fraud_count,
            false_positives=false_positives,
            false_negatives=false_negatives,
            kl_divergence=kl_divergence
        )
        
        db.session.add(metrics)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'date': target_date.strftime('%Y-%m-%d'),
            'metrics_id': metrics.id,
            'transactions_processed': total_count
        })
    
    except Exception as e:
        logger.error(f"Error updating daily metrics: {e}")
        db.session.rollback()
        return jsonify({
            'error': 'Error updating daily metrics',
            'details': str(e)
        }), 500
