import os
import unittest
import json
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add parent directory to import path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db
from models import Transaction, User, ModelVersion
from api import get_model

class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        # Configure test database
        self.db_fd, self.db_path = tempfile.mkstemp()
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{self.db_path}'
        app.config['TESTING'] = True
        
        # Create test client
        self.client = app.test_client()
        
        # Create database tables
        with app.app_context():
            db.create_all()
            
            # Create test user
            test_user = User(
                username='testuser',
                email='test@example.com',
                role='analyst'
            )
            test_user.set_password('password')
            
            # Create test model version
            test_model = ModelVersion(
                version='test_version',
                created_at=datetime.utcnow(),
                active=True,
                autoencoder_path='test_autoencoder_path',
                gbm_path='test_gbm_path',
                precision=0.95,
                recall=0.98
            )
            
            # Add to database
            db.session.add(test_user)
            db.session.add(test_model)
            db.session.commit()
    
    def tearDown(self):
        """Clean up after tests"""
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get('/api/v1/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
    
    @patch('api.get_model')
    def test_score_transaction_validation(self, mock_get_model):
        """Test transaction scoring with invalid data"""
        # Invalid request (missing required fields)
        response = self.client.post(
            '/api/v1/score',
            json={'amount': 100.0}
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('details', data)
    
    @patch('api.get_model')
    def test_score_transaction_duplicate(self, mock_get_model):
        """Test scoring a duplicate transaction"""
        # Create a transaction first
        with app.app_context():
            transaction = Transaction(
                id='tx_12345',
                amount=100.0,
                timestamp=datetime.utcnow(),
                customer_id='cust_123',
                merchant_id='merch_456',
                payment_method='card',
                fraud_score=0.1,
                is_fraud=False
            )
            db.session.add(transaction)
            db.session.commit()
        
        # Try to score the same transaction
        response = self.client.post(
            '/api/v1/score',
            json={
                'transaction_id': 'tx_12345',
                'amount': 100.0,
                'timestamp': datetime.utcnow().isoformat(),
                'customer_id': 'cust_123',
                'merchant_id': 'merch_456',
                'payment_method': 'card'
            }
        )
        
        self.assertEqual(response.status_code, 409)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['transaction_id'], 'tx_12345')
    
    @patch('api.get_model')
    def test_score_transaction_model_unavailable(self, mock_get_model):
        """Test scoring when model is unavailable"""
        # Mock get_model to return None
        mock_get_model.return_value = None
        
        # Try to score a transaction
        response = self.client.post(
            '/api/v1/score',
            json={
                'transaction_id': 'tx_new_123',
                'amount': 100.0,
                'timestamp': datetime.utcnow().isoformat(),
                'customer_id': 'cust_123',
                'merchant_id': 'merch_456',
                'payment_method': 'card'
            }
        )
        
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Model not available')
    
    @patch('api.get_model')
    @patch('data_pipeline.DataPipeline.preprocess_data')
    @patch('tensorflow.convert_to_tensor')
    def test_score_transaction_success(self, mock_convert_to_tensor, mock_preprocess_data, mock_get_model):
        """Test successful transaction scoring"""
        # Mock model and preprocessed data
        mock_model = MagicMock()
        mock_model.predict_fraud.return_value = (
            [0.25],  # fraud_score
            [False]  # is_fraud
        )
        mock_model.explain_prediction.return_value = {
            'key_factors': [
                {'feature': 'amount', 'importance': 0.8},
                {'feature': 'card_present', 'importance': 0.5}
            ]
        }
        mock_get_model.return_value = mock_model
        
        # Mock preprocessing
        mock_preprocess_data.return_value = pd.DataFrame({
            'amount': [100.0],
            'card_present': [False]
        })
        
        # Mock tensor conversion
        mock_convert_to_tensor.return_value = tf.constant([[100.0, 0.0]])
        
        # Score a transaction
        transaction_data = {
            'transaction_id': 'tx_new_456',
            'amount': 100.0,
            'timestamp': datetime.utcnow().isoformat(),
            'customer_id': 'cust_123',
            'merchant_id': 'merch_456',
            'payment_method': 'card',
            'card_present': False,
            'additional_features': {
                'ip_address': '192.168.1.1',
                'device_id': 'device_abc'
            }
        }
        
        response = self.client.post(
            '/api/v1/score',
            json=transaction_data
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check response fields
        self.assertEqual(data['transaction_id'], 'tx_new_456')
        self.assertEqual(data['fraud_score'], 0.25)
        self.assertEqual(data['is_fraudulent'], False)
        self.assertIn('confidence', data)
        self.assertIn('explanation', data)
        self.assertIn('processing_time_ms', data)
        
        # Verify model was called with proper data
        mock_model.predict_fraud.assert_called_once()
    
    def test_get_model_info(self):
        """Test model info endpoint"""
        response = self.client.get('/api/v1/model/info')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check model info fields
        self.assertEqual(data['version'], 'test_version')
        self.assertIn('created_at', data)
        self.assertEqual(data['precision'], 0.95)
        self.assertEqual(data['recall'], 0.98)
        self.assertEqual(data['active'], True)
    
    def test_get_recent_transactions(self):
        """Test recent transactions endpoint"""
        # Add some transactions
        with app.app_context():
            for i in range(5):
                transaction = Transaction(
                    id=f'tx_recent_{i}',
                    amount=100.0 * (i + 1),
                    timestamp=datetime.utcnow(),
                    customer_id='cust_123',
                    merchant_id='merch_456',
                    payment_method='card',
                    fraud_score=0.1 * (i + 1),
                    is_fraud=i >= 3  # Last 2 are fraud
                )
                db.session.add(transaction)
            db.session.commit()
        
        # Get recent transactions
        response = self.client.get('/api/v1/transactions/recent?limit=3')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check response structure
        self.assertIn('transactions', data)
        self.assertIn('total_count', data)
        self.assertEqual(data['limit'], 3)
        self.assertEqual(data['offset'], 0)
        
        # Check transaction count
        self.assertEqual(len(data['transactions']), 3)
        
        # Test fraud_only filter
        response = self.client.get('/api/v1/transactions/recent?fraud_only=true')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Should only return fraud transactions
        self.assertGreaterEqual(len(data['transactions']), 2)
        for tx in data['transactions']:
            self.assertTrue(tx['is_fraud'])
    
    def test_review_transaction_invalid(self):
        """Test transaction review with invalid data"""
        # Missing required fields
        response = self.client.post(
            '/api/v1/transactions/tx_12345/review',
            json={'notes': 'Test review'}
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('required', data)
    
    def test_review_transaction_not_found(self):
        """Test reviewing a non-existent transaction"""
        response = self.client.post(
            '/api/v1/transactions/nonexistent_tx/review',
            json={
                'is_fraud': True,
                'reviewer_id': 1,
                'notes': 'Test review'
            }
        )
        
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Transaction not found')
    
    def test_review_transaction_success(self):
        """Test successful transaction review"""
        # Create a transaction first
        with app.app_context():
            transaction = Transaction(
                id='tx_to_review',
                amount=500.0,
                timestamp=datetime.utcnow(),
                customer_id='cust_123',
                merchant_id='merch_456',
                payment_method='card',
                fraud_score=0.8,
                is_fraud=True,
                reviewed=False
            )
            db.session.add(transaction)
            db.session.commit()
        
        # Review the transaction
        response = self.client.post(
            '/api/v1/transactions/tx_to_review/review',
            json={
                'is_fraud': True,
                'reviewer_id': 1,
                'notes': 'Confirmed fraud transaction'
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check response
        self.assertEqual(data['transaction_id'], 'tx_to_review')
        self.assertEqual(data['reviewed'], True)
        self.assertEqual(data['is_fraud'], True)
        self.assertEqual(data['reviewed_by'], 1)
        self.assertIn('reviewed_at', data)
        
        # Verify transaction was updated in database
        with app.app_context():
            updated_tx = Transaction.query.get('tx_to_review')
            self.assertEqual(updated_tx.reviewed, True)
            self.assertEqual(updated_tx.review_result, True)
            self.assertEqual(updated_tx.reviewed_by, 1)
            self.assertIsNotNone(updated_tx.reviewed_at)


class TestModelCache(unittest.TestCase):
    """Test the model caching functionality"""
    
    @patch('api._model_cache')
    @patch('api.get_active_model')
    @patch('model_architecture.AutoencoderModel')
    @patch('model_architecture.GradientBoostedTreesModel')
    @patch('model_architecture.HybridFraudModel')
    def test_get_model_cache(self, mock_hybrid, mock_gbm, mock_autoencoder, mock_get_active_model, mock_cache):
        """Test model cache functionality"""
        # Set mock_cache to None to force model loading
        mock_cache.return_value = None
        
        # Mock active model
        mock_model_version = MagicMock()
        mock_model_version.autoencoder_path = 'test_ae_path'
        mock_model_version.gbm_path = 'test_gbm_path'
        mock_model_version.version = 'test_version'
        mock_get_active_model.return_value = mock_model_version
        
        # Mock component models
        mock_ae_instance = MagicMock()
        mock_gbm_instance = MagicMock()
        mock_hybrid_instance = MagicMock()
        
        mock_autoencoder.return_value = mock_ae_instance
        mock_gbm.return_value = mock_gbm_instance
        mock_hybrid.return_value = mock_hybrid_instance
        
        # Call get_model
        with app.app_context():
            model = get_model()
        
        # Verify model was created correctly
        mock_get_active_model.assert_called_once()
        mock_autoencoder.assert_called_once()
        mock_gbm.assert_called_once()
        mock_hybrid.assert_called_once_with(
            autoencoder=mock_ae_instance,
            gbm=mock_gbm_instance,
            autoencoder_weight=0.7
        )
        
        self.assertEqual(model, mock_hybrid_instance)
    
    @patch('api._model_cache')
    def test_get_model_from_cache(self, mock_cache):
        """Test getting model from cache"""
        # Set mock_cache to a non-None value
        mock_model = MagicMock()
        mock_cache.__bool__.return_value = True
        mock_cache.__nonzero__ = mock_cache.__bool__  # For Python 2 compatibility
        api._model_cache = mock_model
        
        # Call get_model
        with app.app_context():
            model = get_model()
        
        # Verify cache was used
        self.assertEqual(model, mock_model)


if __name__ == '__main__':
    unittest.main()
