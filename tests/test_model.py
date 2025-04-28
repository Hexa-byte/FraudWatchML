import os
import unittest
import tensorflow as tf
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Add parent directory to import path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_architecture import AutoencoderModel, GradientBoostedTreesModel, HybridFraudModel
from data_pipeline import DataPipeline

class TestAutoencoderModel(unittest.TestCase):
    """Test cases for the Autoencoder model"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a small dataset for testing
        self.input_dim = 10
        self.latent_dim = 4
        self.model = AutoencoderModel(input_dim=self.input_dim, latent_dim=self.latent_dim)
        
        # Create random test data
        self.test_data = np.random.random((5, self.input_dim))
    
    def test_model_initialization(self):
        """Test that the model initializes correctly"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.encoder)
        self.assertIsNotNone(self.model.decoder)
        
        # Check model architecture
        self.assertEqual(self.model.model.input_shape, (None, self.input_dim))
        self.assertEqual(self.model.model.output_shape, (None, self.input_dim))
        
        # Check encoder architecture
        self.assertEqual(self.model.encoder.input_shape, (None, self.input_dim))
        self.assertEqual(self.model.encoder.output_shape, (None, self.latent_dim))
        
        # Check decoder architecture
        self.assertEqual(self.model.decoder.input_shape, (None, self.latent_dim))
        self.assertEqual(self.model.decoder.output_shape, (None, self.input_dim))
    
    def test_reconstruction(self):
        """Test that the autoencoder can reconstruct inputs"""
        # Get reconstruction
        reconstructed = self.model.model.predict(self.test_data)
        
        # Check shape
        self.assertEqual(reconstructed.shape, self.test_data.shape)
        
        # Test that reconstruction error is computed correctly
        error = self.model.compute_reconstruction_error(self.test_data)
        self.assertEqual(error.shape, (5,))  # Should be one error value per sample
    
    @patch('os.makedirs')
    @patch('tensorflow.keras.models.Model.save')
    def test_save_model(self, mock_save, mock_makedirs):
        """Test model saving functionality"""
        # Configure mocks
        mock_makedirs.return_value = None
        mock_save.return_value = None
        
        # Call save_model
        model_path = self.model.save_model(model_dir='test_models')
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with('test_models', exist_ok=True)
        
        # Verify save called
        mock_save.assert_called_once()
        
        # Check path format
        self.assertTrue(model_path.startswith('test_models/autoencoder_'))
        self.assertTrue(model_path.endswith('.h5'))

    @patch('tensorflow.keras.models.load_model')
    def test_load_model(self, mock_load_model):
        """Test model loading functionality"""
        # Configure mock
        mock_model = MagicMock()
        mock_model.input = tf.keras.layers.Input(shape=(self.input_dim,))
        mock_model.layers = [
            MagicMock(name='encoder_input'),
            MagicMock(name='encoder_dense_1'),
            MagicMock(name='encoder_dense_2'),
            MagicMock(name='encoder_dense_3'),
            MagicMock(name='decoder_dense_1'),
            MagicMock(name='decoder_dense_2'),
            MagicMock(name='decoder_output')
        ]
        mock_load_model.return_value = mock_model
        
        # Call load_model
        model = AutoencoderModel(input_dim=self.input_dim, latent_dim=self.latent_dim)
        model.load_model('test_path.h5')
        
        # Verify load called
        mock_load_model.assert_called_once_with('test_path.h5')
        
        # Check that model was updated
        self.assertEqual(model.model, mock_model)


class TestGradientBoostedTreesModel(unittest.TestCase):
    """Test cases for the GBM model"""
    
    @patch('tensorflow_decision_forests.keras.GradientBoostedTreesModel')
    def setUp(self, mock_gbm):
        """Set up test environment"""
        # Configure mock
        self.mock_gbm_instance = MagicMock()
        mock_gbm.return_value = self.mock_gbm_instance
        
        # Create model
        self.model = GradientBoostedTreesModel()
        
        # Create random test data
        self.features = np.random.random((5, 10))
        self.labels = np.random.randint(0, 2, size=(5,))
        
        # Create TensorFlow dataset
        self.dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels)).batch(2)
    
    def test_model_initialization(self):
        """Test that the model initializes correctly"""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.model, self.mock_gbm_instance)
    
    def test_train(self):
        """Test model training functionality"""
        # Configure mock
        self.mock_gbm_instance.fit.return_value = MagicMock()
        
        # Call train
        history = self.model.train(self.dataset)
        
        # Verify fit called
        self.mock_gbm_instance.fit.assert_called_once()
        
        # Check arguments
        args, kwargs = self.mock_gbm_instance.fit.call_args
        self.assertEqual(args[0], self.dataset)
    
    @patch('os.makedirs')
    def test_save_model(self, mock_makedirs):
        """Test model saving functionality"""
        # Configure mocks
        mock_makedirs.return_value = None
        self.mock_gbm_instance.save.return_value = None
        
        # Call save_model
        model_path = self.model.save_model(model_dir='test_models')
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with('test_models', exist_ok=True)
        
        # Verify save called
        self.mock_gbm_instance.save.assert_called_once()
        
        # Check path format
        self.assertTrue(model_path.startswith('test_models/gbm_'))
    
    @patch('tensorflow_decision_forests.keras.GradientBoostedTreesModel.load')
    def test_load_model(self, mock_load):
        """Test model loading functionality"""
        # Configure mock
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Call load_model
        model = GradientBoostedTreesModel()
        model.load_model('test_path')
        
        # Verify load called
        mock_load.assert_called_once_with('test_path')
        
        # Check that model was updated
        self.assertEqual(model.model, mock_model)


class TestHybridFraudModel(unittest.TestCase):
    """Test cases for the hybrid fraud model"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock autoencoder
        self.mock_autoencoder = MagicMock()
        self.mock_autoencoder.compute_reconstruction_error.return_value = np.array([0.1, 0.8, 0.3])
        
        # Create mock GBM
        self.mock_gbm = MagicMock()
        self.mock_gbm.predict.return_value = np.array([0.2, 0.7, 0.4])
        
        # Create hybrid model
        self.model = HybridFraudModel(
            autoencoder=self.mock_autoencoder,
            gbm=self.mock_gbm,
            autoencoder_weight=0.7
        )
        
        # Create test data
        self.test_data = np.random.random((3, 10))
    
    def test_model_initialization(self):
        """Test that the model initializes correctly"""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.autoencoder, self.mock_autoencoder)
        self.assertEqual(self.model.gbm, self.mock_gbm)
        self.assertEqual(self.model.autoencoder_weight, 0.7)
        self.assertEqual(self.model.gbm_weight, 0.3)
    
    def test_predict_fraud(self):
        """Test fraud prediction functionality"""
        # Get predictions
        scores, is_fraud = self.model.predict_fraud(self.test_data, threshold=0.5)
        
        # Verify calls to component models
        self.mock_autoencoder.compute_reconstruction_error.assert_called_once()
        self.mock_gbm.predict.assert_called_once()
        
        # Check results
        expected_scores = 0.7 * np.array([0.1, 0.8, 0.3]) + 0.3 * np.array([0.2, 0.7, 0.4])
        expected_is_fraud = expected_scores > 0.5
        
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-5)
        np.testing.assert_array_equal(is_fraud, expected_is_fraud)
    
    def test_explain_prediction(self):
        """Test prediction explanation functionality"""
        # Configure mock GBM for inspector
        inspector = MagicMock()
        inspector.variable_importances.return_value = {
            "MEAN_DECREASE_IN_ACCURACY": {
                "feature1": 0.3,
                "feature2": 0.2,
                "feature3": 0.1,
                "feature4": 0.05,
                "feature5": 0.01
            }
        }
        self.mock_gbm.model.make_inspector.return_value = inspector
        
        # Get explanation
        explanation = self.model.explain_prediction(self.test_data[0:1], ["feature1", "feature2", "feature3", "feature4", "feature5"])
        
        # Verify inspector called
        self.mock_gbm.model.make_inspector.assert_called_once()
        inspector.variable_importances.assert_called_once()
        
        # Check explanation format
        self.assertIn("key_factors", explanation)
        self.assertEqual(len(explanation["key_factors"]), 5)
        
        # Check factors are sorted by importance
        factors = explanation["key_factors"]
        self.assertEqual(factors[0]["feature"], "feature1")
        self.assertEqual(factors[1]["feature"], "feature2")


class TestDataPipeline(unittest.TestCase):
    """Test cases for the data pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.pipeline = DataPipeline()
        
        # Create sample DataFrame
        self.sample_data = pd.DataFrame({
            'transaction_id': ['tx_1', 'tx_2', 'tx_3', 'tx_4', 'tx_5'],
            'amount': [100.0, 500.0, 2000.0, 50.0, 1000.0],
            'timestamp': pd.date_range(start='2023-01-01', periods=5),
            'customer_id': ['cust_1', 'cust_2', 'cust_1', 'cust_3', 'cust_2'],
            'merchant_id': ['merch_1', 'merch_2', 'merch_3', 'merch_2', 'merch_1'],
            'payment_method': ['card', 'bank_transfer', 'card', 'wallet', 'card'],
            'card_present': [True, False, False, False, True],
            'is_fraud': [0, 0, 1, 0, 0]
        })
    
    def test_preprocess_data(self):
        """Test data preprocessing functionality"""
        # Preprocess data
        processed_df = self.pipeline.preprocess_data(self.sample_data, training=True)
        
        # Check that new features were added
        self.assertIn('hour_of_day', processed_df.columns)
        self.assertIn('day_of_week', processed_df.columns)
        self.assertIn('weekend', processed_df.columns)
        
        # Check categorical encoding
        self.assertIn('payment_method_card', processed_df.columns)
        self.assertIn('payment_method_bank_transfer', processed_df.columns)
        self.assertNotIn('payment_method', processed_df.columns)
        
        # Check outlier detection
        self.assertIn('mahalanobis_distance', processed_df.columns)
        self.assertIn('is_outlier', processed_df.columns)
    
    def test_handle_class_imbalance(self):
        """Test class imbalance handling functionality"""
        # Add more non-fraud examples to create imbalance
        additional_data = pd.DataFrame({
            'transaction_id': ['tx_6', 'tx_7', 'tx_8', 'tx_9', 'tx_10'],
            'amount': [200.0, 300.0, 400.0, 600.0, 700.0],
            'timestamp': pd.date_range(start='2023-01-06', periods=5),
            'customer_id': ['cust_1', 'cust_2', 'cust_3', 'cust_1', 'cust_2'],
            'merchant_id': ['merch_1', 'merch_2', 'merch_1', 'merch_3', 'merch_2'],
            'payment_method': ['card', 'bank_transfer', 'wallet', 'card', 'bank_transfer'],
            'card_present': [True, False, False, True, False],
            'is_fraud': [0, 0, 0, 0, 0]
        })
        
        imbalanced_df = pd.concat([self.sample_data, additional_data])
        
        # Original ratio: 1 fraud out of 10 (10%)
        self.assertEqual(imbalanced_df['is_fraud'].sum(), 1)
        self.assertEqual(len(imbalanced_df), 10)
        
        # Handle class imbalance
        balanced_df = self.pipeline.handle_class_imbalance(imbalanced_df)
        
        # Verify that fraud ratio has increased
        fraud_ratio = balanced_df['is_fraud'].mean()
        self.assertGreater(fraud_ratio, 0.1)  # Should be higher than original 10%
    
    def test_create_tf_dataset(self):
        """Test TensorFlow dataset creation"""
        # Create dataset
        dataset = self.pipeline.create_tf_dataset(self.sample_data, batch_size=2)
        
        # Check that it's a TensorFlow dataset
        self.assertIsInstance(dataset, tf.data.Dataset)
        
        # Get one batch
        for features, labels in dataset.take(1):
            # Check batch shape
            self.assertEqual(labels.shape[0], 2)  # Batch size
            
            # Check features
            self.assertIsInstance(features, dict)


if __name__ == '__main__':
    unittest.main()
